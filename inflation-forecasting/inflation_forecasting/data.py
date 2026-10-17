"""Data download (FRED), transformation and a synthetic generator for offline testing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter

from . import config as C


# --------------------------------------------------------------------------- download
def _read_fred(code: str, cache: bool = True, refresh: bool = False) -> pd.Series:
    """Download one FRED series as CSV (no API key needed) and cache it in data/."""
    C.DATA_DIR.mkdir(exist_ok=True)
    path = C.DATA_DIR / f"{code}.csv"
    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        if cache:
            df.to_csv(path)
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    s.name = code
    if code in C.PCT_CHANGE_CODES:
        s = 100 * s.pct_change()
    return s


def load_raw(refresh: bool = False) -> pd.DataFrame:
    """Download all raw series; try fallback codes when the first one fails."""
    out = {}
    for name, codes in C.FRED_CODES.items():
        last_err = None
        for code in codes:
            try:
                out[name] = _read_fred(code, refresh=refresh)
                if code != codes[0]:
                    print(f"[data] {name}: using fallback series {code}")
                break
            except Exception as e:  # network / missing series
                last_err = e
        else:
            raise RuntimeError(f"Could not download {name} from {codes}: {last_err}")
    raw = pd.concat(out, axis=1)
    return raw.loc[C.DOWNLOAD_START : C.TEST_END]


# --------------------------------------------------------------------------- transforms
def monthly_inflation(price: pd.Series) -> pd.Series:
    """Y_t = 1200 * [log P_t - log P_{t-1}]  (annualised monthly inflation, paper Sec. 2.1)."""
    return 1200 * np.log(price).diff()


def hp_gap(series: pd.Series, lamb: int = C.HP_LAMBDA) -> pd.Series:
    """Gap = series minus its Hodrick-Prescott trend (paper footnote 2)."""
    cycle, _trend = hpfilter(series.dropna(), lamb=lamb)
    return cycle.reindex(series.index)


def build_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw FRED levels -> modelling dataset (4 inflation series + 6 predictors)."""
    df = pd.DataFrame(index=raw.index)
    for name in C.INFLATION_SERIES:
        df[name] = monthly_inflation(raw[name])
    df = df.loc[C.SAMPLE_START :]
    r = raw.loc[C.SAMPLE_START :]
    for name in ["UNEM", "IP", "WORK", "HS"]:
        df[name] = hp_gap(r[name])
    df["INC"] = r["INC"]
    df["SPREAD"] = r["GS5"] - r["TB3MS"]
    df = df.asfreq("MS").dropna()
    return df


def load_dataset(refresh: bool = False) -> pd.DataFrame:
    return build_dataset(load_raw(refresh=refresh))


# --------------------------------------------------------------------------- synthetic
def make_synthetic(seed: int = 0) -> pd.DataFrame:
    """Random data with the same shape/columns as the real dataset.

    ONLY for smoke-testing the pipeline offline - results are meaningless economically.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(C.SAMPLE_START, C.TEST_END, freq="MS")
    n = len(idx)
    base = np.zeros(n)
    for t in range(1, n):
        base[t] = 0.7 * base[t - 1] + rng.normal(0, 1)
    d = {}
    for name, (scale, mean, phi) in {
        "CPI": (3.0, 2.7, 0.2), "Core-CPI": (2.4, 2.7, 0.6),
        "PCE": (6.0, 5.2, 0.1), "Core-PCE": (6.5, 5.5, 0.1),
    }.items():
        e = rng.normal(0, scale, n) + 0.3 * scale * base
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + e[t]
        d[name] = mean + y
    d["UNEM"] = 0.37 * (-base + rng.normal(0, 0.3, n))
    d["IP"] = 1.5 * (base + rng.normal(0, 0.5, n))
    d["WORK"] = 700 * (base + rng.normal(0, 0.5, n))
    d["HS"] = 100 * (base + rng.normal(0, 0.8, n))
    d["INC"] = 0.25 + 0.4 * rng.normal(size=n) + 0.1 * base
    d["SPREAD"] = 1.4 + 0.8 * np.sin(np.arange(n) / 30) + rng.normal(0, 0.1, n)
    return pd.DataFrame(d, index=idx)
