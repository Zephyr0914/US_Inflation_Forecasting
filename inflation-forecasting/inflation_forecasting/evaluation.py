"""RMSE (Eq. 6), R^2 (Eqs. 7-8) and win-count summaries (paper Tables 3-4 and Sec. 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def rmse(actual, forecast, h):
    a, f = np.asarray(actual)[:h], np.asarray(forecast)[:h]
    return float(np.sqrt(np.mean((a - f) ** 2)))


def rmse_table(df, forecasts, models=None) -> pd.DataFrame:
    """Rows = (Horizon, Inflation); columns = models. RMSE over the first h months of 2014."""
    models = models or C.MODEL_ORDER
    test = df.loc[C.TEST_START : C.TEST_END]
    rows = {}
    for h in C.HORIZONS:
        for infl in C.INFLATION_SERIES:
            rows[(h, infl)] = {m: rmse(test[infl], forecasts[(infl, m)], h) for m in models}
    t = pd.DataFrame(rows).T[models]
    t.index.names = ["Horizon", "Inflation"]
    return t


def r2_table(df, forecasts, models=None) -> pd.DataFrame:
    """R^2 of the regression of actual on forecast values over the 12-month horizon."""
    models = models or C.MODEL_ORDER
    test = df.loc[C.TEST_START : C.TEST_END]
    rows = {}
    for infl in C.INFLATION_SERIES:
        rows[infl] = {}
        for m in models:
            f = np.asarray(forecasts[(infl, m)])
            rows[infl][m] = float(np.corrcoef(test[infl].values, f)[0, 1] ** 2) if np.std(f) > 0 else 0.0
    return pd.DataFrame(rows).T[models]


def winners(rmse_df: pd.DataFrame) -> pd.DataFrame:
    best = rmse_df.idxmin(axis=1).rename("Best model").to_frame()
    best["Family"] = np.where(best["Best model"].isin(C.ML_MODELS), "Machine learning", "Time series")
    best["Type"] = np.where(best["Best model"].isin(C.UNIVARIATE_MODELS), "Univariate", "Multivariate")
    return best


def summarize(rmse_df: pd.DataFrame) -> str:
    w = winners(rmse_df)
    n = len(w)
    ml, ts = (w["Family"] == "Machine learning").sum(), (w["Family"] == "Time series").sum()
    mv, uv = (w["Type"] == "Multivariate").sum(), (w["Type"] == "Univariate").sum()
    lines = [
        f"Out of {n} conditions: ML best in {ml}, time-series best in {ts}",
        f"Multivariate best in {mv}, univariate best in {uv}",
        "Wins per model: " + ", ".join(f"{k}={v}" for k, v in w["Best model"].value_counts().items()),
        "(Paper: ML 7 / TS 9; multivariate 14 / univariate 2)",
    ]
    return "\n".join(lines)


def to_markdown(df: pd.DataFrame, bold_min=True, fmt="{:.2f}") -> str:
    """Markdown table; the row-wise best value is starred like in the paper."""
    d = df.reset_index()
    idx_cols = list(df.index.names) if df.index.names[0] else ["index"]
    val_cols = list(df.columns)
    best = df.idxmin(axis=1) if bold_min else df.idxmax(axis=1)
    lines = ["| " + " | ".join(idx_cols + val_cols) + " |", "|" + "---|" * (len(idx_cols) + len(val_cols))]
    for i, (_, row) in enumerate(d.iterrows()):
        cells = [str(row[c]) for c in idx_cols]
        for c in val_cols:
            s = fmt.format(row[c])
            cells.append(s + "*" if best.iloc[i] == c else s)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
