import numpy as np
import pytest

from inflation_forecasting import config as C
from inflation_forecasting.data import hp_gap, make_synthetic, monthly_inflation
from inflation_forecasting.evaluation import r2_table, rmse, rmse_table, winners
from inflation_forecasting.models import forecast_ar, forecast_ardl, forecast_ml, forecast_naive, forecast_var, split

import pandas as pd


@pytest.fixture(scope="module")
def df():
    return make_synthetic(seed=1)


def test_inflation_transform():
    p = pd.Series([100.0, 101.0, 102.01])
    out = monthly_inflation(p)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(1200 * np.log(1.01))


def test_hp_gap_zero_mean_ish():
    s = pd.Series(np.cumsum(np.random.default_rng(0).normal(size=200)))
    assert abs(hp_gap(s).mean()) < 1e-6


def test_split_sizes(df):
    train, test = split(df)
    assert len(test) == 12 and train.index[-1] < test.index[0]


@pytest.mark.parametrize("fn", [forecast_ar, forecast_naive, forecast_var, forecast_ardl])
def test_ts_models_shape(df, fn):
    train, test = split(df)
    fc = fn(train, test, "CPI")
    assert fc.shape == (12,) and np.isfinite(fc).all()


@pytest.mark.parametrize("name", ["k-NN", "SVR", "ANN"])
@pytest.mark.parametrize("recursive", [False, True])
def test_ml_models_shape(df, name, recursive):
    fc = forecast_ml(name, df, "Core-CPI", recursive=recursive, n_seeds=1)
    assert fc.shape == (12,) and np.isfinite(fc).all()


def test_rmse_and_tables(df):
    assert rmse([1, 2, 3], [1, 2, 5], 3) == pytest.approx(np.sqrt(4 / 3))
    test = df.loc[C.TEST_START:]
    fc = {(i, m): test[i].values + 1.0 for i in C.INFLATION_SERIES for m in C.MODEL_ORDER}
    t = rmse_table(df, fc)
    assert np.allclose(t.values, 1.0)
    assert np.allclose(r2_table(df, fc).values, 1.0)
    assert len(winners(t)) == 16
