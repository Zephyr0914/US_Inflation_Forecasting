"""Figures 1-4 of the paper: actual vs forecast inflation in 2014 (12-month horizon)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C


def plot_forecasts(df, forecasts, out_dir=None, models=None):
    out_dir = out_dir or (C.RESULTS_DIR / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    models = models or C.MODEL_ORDER
    test = df.loc[C.TEST_START : C.TEST_END]
    paths = []
    for i, infl in enumerate(C.INFLATION_SERIES, 1):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(test.index, test[infl], "k-o", lw=2.5, label="Actual")
        for m in models:
            ax.plot(test.index, forecasts[(infl, m)], marker=".", lw=1.2, label=m)
        ax.set_title(f"Fig. {i}: Forecasted and actual {infl} inflation, 2014 (h = 12)")
        ax.set_ylabel("Annualised monthly inflation (%)")
        ax.grid(alpha=0.3)
        ax.legend(ncol=4, fontsize=8)
        fig.tight_layout()
        p = out_dir / f"fig{i}_{infl.lower().replace('-', '_')}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths
