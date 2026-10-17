"""Run the full experiment:  python main.py   (or  python main.py --synthetic  to test offline)."""
import argparse

import pandas as pd

from inflation_forecasting import config as C
from inflation_forecasting.data import load_dataset, make_synthetic
from inflation_forecasting.evaluation import r2_table, rmse_table, summarize, to_markdown, winners
from inflation_forecasting.models import run_all
from inflation_forecasting.plotting import plot_forecasts


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic", action="store_true", help="use random data (offline smoke test)")
    p.add_argument("--refresh", action="store_true", help="re-download FRED data")
    p.add_argument("--ml-recursive", action="store_true",
                   help="ML models feed back their own forecasts (default: one-step with realised data)")
    p.add_argument("--tune-k", action="store_true", help="tune k of k-NN by CV instead of k=3")
    p.add_argument("--seeds", type=int, default=C.ANN_N_SEEDS, help="ANN random restarts to average")
    return p.parse_args()


def main():
    a = parse()
    C.RESULTS_DIR.mkdir(exist_ok=True)
    df = make_synthetic() if a.synthetic else load_dataset(refresh=a.refresh)
    print(f"Dataset: {df.index[0]:%Y-%m} -> {df.index[-1]:%Y-%m}, {len(df)} obs")
    print(df.describe().T[["mean", "50%", "max", "min", "std"]].round(2))

    print("\nFitting models ...")
    fc = run_all(df, recursive_ml=a.ml_recursive, tune_k=a.tune_k, n_seeds=a.seeds)

    rm, r2 = rmse_table(df, fc), r2_table(df, fc)
    tag = "synthetic_" if a.synthetic else ""
    rm.round(4).to_csv(C.RESULTS_DIR / f"{tag}table3_rmse.csv")
    r2.round(4).to_csv(C.RESULTS_DIR / f"{tag}table4_r2.csv")
    pd.DataFrame({f"{i}|{m}": v for (i, m), v in fc.items()}, index=df.loc[C.TEST_START:].index) \
        .to_csv(C.RESULTS_DIR / f"{tag}forecasts_2014.csv")
    (C.RESULTS_DIR / f"{tag}tables.md").write_text(
        "## Table 3 - RMSE (2014)\n\n" + to_markdown(rm) +
        "\n\n## Table 4 - R2, h = 12\n\n" + to_markdown(r2, bold_min=False) + "\n")
    plot_forecasts(df, fc, C.RESULTS_DIR / f"{tag}figures")

    print("\nTable 3 - RMSE\n", rm.round(2))
    print("\nTable 4 - R2 (h=12)\n", r2.round(2))
    print("\nBest model per condition\n", winners(rm))
    print("\n" + summarize(rm))
    print(f"\nSaved outputs to {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()
