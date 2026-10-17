"""The seven forecasting models: AR, Naive, ARDL, VAR (time series); ANN, k-NN, SVR (ML).

Protocol (see README "Interpretation of the paper"):
  * Time-series models are estimated on Jan-1984..Dec-2013 and forecast the 12 months of 2014.
      AR / Naive / VAR : dynamic (iterated multi-step) forecasts, no 2014 information used.
      ARDL             : conditional forecast - uses the realised 2014 regressors.
  * ML models use lagged inflation + the six predictors (t and t-1). By default they are
    one-step-ahead using realised 2014 data; ``recursive=True`` feeds back their own forecasts.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.ardl import ardl_select_order
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR

from . import config as C

warnings.filterwarnings("ignore")


def split(df: pd.DataFrame):
    return df.loc[: C.TRAIN_END], df.loc[C.TEST_START : C.TEST_END]


# =============================================================== time-series models
def forecast_ar(train, test, target):
    res = AutoReg(train[target], lags=C.AR_LAGS[target], trend="c").fit()
    return np.asarray(res.forecast(len(test)))


def forecast_naive(train, test, target):
    """Eq. (2): Y_t = a1 * Y_{t-1} + e_t  (OLS, no constant), iterated forward."""
    res = AutoReg(train[target], lags=1, trend="n").fit()
    return np.asarray(res.forecast(len(test)))


def forecast_var(train, test, target):
    cols = [target] + C.PREDICTORS
    res = VAR(train[cols]).fit(C.VAR_LAGS)
    fc = res.forecast(train[cols].values[-C.VAR_LAGS:], steps=len(test))
    return fc[:, 0]


def forecast_ardl(train, test, target):
    """Eq. (3): ARDL with per-variable lag orders chosen by BIC (Schwarz)."""
    sel = ardl_select_order(
        train[target], maxlag=C.ARDL_MAX_LAG, exog=train[C.PREDICTORS],
        maxorder=C.ARDL_MAX_ORDER, trend="c", ic="bic",
    )
    res = sel.model.fit()
    return np.asarray(res.forecast(steps=len(test), exog=test[C.PREDICTORS]))


# =============================================================== machine-learning models
def make_features(df: pd.DataFrame, target: str, y_lags: int = C.ML_Y_LAGS) -> pd.DataFrame:
    X = {f"{target}_l{k}": df[target].shift(k) for k in range(1, y_lags + 1)}
    for c in C.PREDICTORS:
        X[c] = df[c]
        X[f"{c}_l1"] = df[c].shift(1)
    return pd.DataFrame(X, index=df.index)


class _Ensemble:
    """Average of several fitted regressors (used for ANN random restarts)."""
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


def _scaled(estimator):
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), estimator), transformer=StandardScaler()
    )


def fit_knn(X, y, tune_k=False):
    k = C.KNN_K
    if tune_k:
        gs = GridSearchCV(_scaled(KNeighborsRegressor()),
                          {"regressor__kneighborsregressor__n_neighbors": list(range(1, 16))},
                          cv=TimeSeriesSplit(5), scoring="neg_root_mean_squared_error").fit(X, y)
        return gs.best_estimator_
    return _scaled(KNeighborsRegressor(n_neighbors=k)).fit(X, y)


def _mlp(activation, seed):
    return _scaled(MLPRegressor(
        hidden_layer_sizes=C.ANN_HIDDEN, activation=activation, solver="sgd",
        learning_rate_init=C.ANN_LR, momentum=C.ANN_MOMENTUM, max_iter=1000,
        random_state=seed))


def fit_ann(X, y, n_seeds=C.ANN_N_SEEDS):
    # choose activation function on the last 36 months of the training sample
    cut = len(X) - 36
    best, best_err = None, np.inf
    for act in C.ANN_ACTIVATIONS:
        m = _mlp(act, C.RANDOM_STATE).fit(X.iloc[:cut], y.iloc[:cut])
        err = np.sqrt(np.mean((y.iloc[cut:] - m.predict(X.iloc[cut:])) ** 2))
        if err < best_err:
            best, best_err = act, err
    models = [_mlp(best, C.RANDOM_STATE + s).fit(X, y) for s in range(n_seeds)]
    return _Ensemble(models)


def fit_svr(X, y):
    grid = {f"regressor__svr__{k}": v for k, v in C.SVR_GRID.items()}
    gs = GridSearchCV(_scaled(SVR(kernel="rbf")), grid, cv=TimeSeriesSplit(5),
                      scoring="neg_root_mean_squared_error", n_jobs=-1).fit(X, y)
    return gs.best_estimator_


def _ml_predict(model, full, target, test_index, recursive):
    if not recursive:
        feats = make_features(full, target).loc[test_index]
        return np.asarray(model.predict(feats))
    work = full.copy()
    work.loc[test_index, target] = np.nan
    preds = []
    for t in test_index:
        row = make_features(work, target).loc[[t]]
        p = float(model.predict(row)[0])
        work.loc[t, target] = p
        preds.append(p)
    return np.asarray(preds)


def forecast_ml(name, df, target, recursive=False, tune_k=False, n_seeds=C.ANN_N_SEEDS):
    train, test = split(df)
    feats = make_features(df, target)
    Xtr = feats.loc[train.index].dropna()
    ytr = train.loc[Xtr.index, target]
    if name == "k-NN":
        model = fit_knn(Xtr, ytr, tune_k)
    elif name == "ANN":
        model = fit_ann(Xtr, ytr, n_seeds)
    elif name == "SVR":
        model = fit_svr(Xtr, ytr)
    else:
        raise ValueError(name)
    return _ml_predict(model, df, target, test.index, recursive)


# =============================================================== driver
def run_all(df, models=None, recursive_ml=False, tune_k=False, n_seeds=C.ANN_N_SEEDS, verbose=True):
    """Return {(inflation, model): np.ndarray(12)} of 2014 forecasts."""
    models = models or C.MODEL_ORDER
    train, test = split(df)
    ts = {"AR": forecast_ar, "Naive": forecast_naive, "ARDL": forecast_ardl, "VAR": forecast_var}
    out = {}
    for target in C.INFLATION_SERIES:
        for m in models:
            if verbose:
                print(f"  fitting {m:6s} on {target}")
            if m in ts:
                out[(target, m)] = ts[m](train, test, target)
            else:
                out[(target, m)] = forecast_ml(m, df, target, recursive_ml, tune_k, n_seeds)
    return out
