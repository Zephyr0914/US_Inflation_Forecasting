# US_Inflation_Forecasting

## What it does

Forecasts four US inflation measures (**CPI, Core-CPI, PCE, Core-PCE**) for the 12 months of 2014 at horizons of
3, 6, 9 and 12 months, using models estimated on Jan 1984 – Dec 2013:

| Family | Model | Type |
|---|---|---|
| Time series | AR (BIC lags: 2 / 12 / 1 / 5) | univariate |
| Time series | Naïve (AR(1) without constant, iterated) | univariate |
| Time series | VAR(1) | multivariate |
| Time series | ARDL (lag orders by BIC) | multivariate |
| Machine learning | k-NN (k = 3) | multivariate |
| Machine learning | ANN (1 hidden layer, 3 neurons, lr 0.3, momentum 0.2) | multivariate |
| Machine learning | SVR (RBF kernel, grid-searched) | multivariate |

Predictors: unemployment, industrial production, nonfarm payrolls, housing starts (HP-filter gaps),
real consumption growth and the term spread (5Y − 3M). Accuracy is measured by RMSE (Table 3) and the R² of
actual-on-forecast regressions (Table 4).

## Quick start

```bash
git clone https://github.com/<you>/inflation-forecasting.git
cd inflation-forecasting
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py                 # downloads data from FRED (no API key) and runs everything
python main.py --synthetic     # offline smoke test with random data (results are meaningless)
pytest                         # unit tests
```

Options: `--refresh` (re-download), `--ml-recursive`, `--tune-k`, `--seeds N`.

Outputs go to `results/`: `table3_rmse.csv`, `table4_r2.csv`, `tables.md`, `forecasts_2014.csv`, and
`figures/fig1…fig4.png` (the analogues of the paper's Figures 1–4). The console also prints the win counts
that the paper reports (ML best in 7 of 16 conditions, time series in 9; multivariate in 14, univariate in 2).

## Project layout

```
├── main.py                       # entry point
├── inflation_forecasting/
│   ├── config.py                 # sample dates, FRED codes, hyper-parameters
│   ├── data.py                   # FRED download, transformations, synthetic data
│   ├── models.py                 # AR, Naïve, VAR, ARDL, k-NN, ANN, SVR
│   ├── evaluation.py             # RMSE, R², winner tables
│   └── plotting.py               # forecast figures
├── tests/test_pipeline.py
├── data/                         # cached FRED CSVs (git-ignored)
├── results/                      # generated tables and figures
└── .github/workflows/tests.yml   # CI
```

## Interpretation of the paper (read this before comparing numbers)

The paper leaves several details open, so these are my choices. All are easy to change in `config.py` / `models.py`.

1. **Horizon h** = RMSE over the first *h* months of 2014 (Eq. 6).
2. **Forecast protocol.** AR, Naïve and VAR produce dynamic multi-step forecasts from Dec 2013. ARDL uses the realised 2014 regressors (its contemporaneous terms need them). ML models use lagged inflation plus predictors at *t* and *t−1* and, by default, **one-step-ahead forecasts using realised 2014 data** — this matches the paper's figures, where ML forecasts track actual data, but it means ML models get more information than AR/Naïve/VAR. Use `--ml-recursive` for a stricter comparison.
3. **Core-PCE series.** Paper Table 5 gives the same FRED code (`DPCERL1M225NBEA`) for Core-PCE and for INC, which looks like a typo. I use `PCEPILFE` (PCE price index excl. food & energy) for Core-PCE. The paper's PCE series is *nominal* PCE spending (code `PCE`), whose growth is not a price inflation measure; I keep it for fidelity — set `"PCE": ["PCEPI"]` in `config.py` for a true price index.
4. **Core-CPI** uses `USACPICORMINMEI`, falling back to `CPILFESL` if unavailable.
5. **Hyper-parameter search** (ANN activation, SVR grid) uses time-ordered validation inside the training sample only. ANN results average several random initialisations.
6. **HP filter** gaps are computed on the full sample, as in the paper's footnote 2. This is a two-sided filter and leaks a little future information into the 2014 predictors.

## Expect differences from the published tables

- FRED data have been revised since 2016, and some series codes may have changed.
- The paper's VAR RMSEs (8–16) are far larger than other models', probably from scaling or specification
  details not reported; this code will not reproduce those values.
- The test set is **12 observations**. RMSE rankings on so few points are noisy — treat "model X wins" as weak evidence.
  Adding a Diebold–Mariano test or a rolling-origin evaluation would be a natural extension.

## Citation

```bibtex
@article{ulke2018comparison,
  title   = {A comparison of time series and machine learning models for inflation forecasting: empirical evidence from the {USA}},
  author  = {{\"U}lke, Volkan and Sahin, Afsin and Subasi, Abdulhamit},
  journal = {Neural Computing and Applications},
  volume  = {30}, pages = {1519--1527}, year = {2018},
  doi     = {10.1007/s00521-016-2766-x}
}
```
