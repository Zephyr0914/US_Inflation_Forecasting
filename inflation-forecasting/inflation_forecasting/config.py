"""Central configuration: sample periods, FRED series codes, model settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# ---- Sample (paper: Jan 1984 - Dec 2013 training, 2014 pseudo out-of-sample) ----
DOWNLOAD_START = "1983-12-01"   # one extra month so Jan-1984 inflation exists
SAMPLE_START = "1984-01-01"
TRAIN_END = "2013-12-01"
TEST_START = "2014-01-01"
TEST_END = "2014-12-01"
HORIZONS = [3, 6, 9, 12]

# ---- Variables ----
INFLATION_SERIES = ["CPI", "Core-CPI", "PCE", "Core-PCE"]
PREDICTORS = ["UNEM", "IP", "WORK", "HS", "INC", "SPREAD"]
GAP_VARIABLES = ["UNEM", "IP", "WORK", "HS"]  # non-stationary -> HP gap (paper footnote 2)
HP_LAMBDA = 129_600  # standard value for monthly data (Ravn-Uhlig)

# FRED codes, tried in order (first that downloads wins).
# NOTE: Table 5 of the paper lists DPCERL1M225NBEA for BOTH Core-PCE and INC, which looks
# like a copy error. Here Core-PCE uses PCEPILFE (PCE price index excl. food & energy).
FRED_CODES = {
    "CPI": ["CPIAUCSL"],
    "Core-CPI": ["USACPICORMINMEI", "CPILFESL"],   # CPILFESL = fallback if OECD series unavailable
    "PCE": ["PCE"],                                # paper: nominal PCE (use "PCEPI" for a true price index)
    "Core-PCE": ["PCEPILFE"],
    "UNEM": ["UNRATE"],
    "IP": ["INDPRO"],
    "INC": ["DPCERL1M225NBEA", "PCEC96"],          # PCEC96 fallback -> converted to % change
    "WORK": ["PAYEMS"],
    "HS": ["HOUST"],
    "GS5": ["GS5"],
    "TB3MS": ["TB3MS"],
}
PCT_CHANGE_CODES = {"PCEC96"}  # level series that must be converted to monthly % change

# ---- Model settings (paper Section 3) ----
AR_LAGS = {"CPI": 2, "Core-CPI": 12, "PCE": 1, "Core-PCE": 5}  # BIC-selected in the paper
VAR_LAGS = 1                    # Schwarz criterion in the paper
ARDL_MAX_LAG = 3                # max lags of the dependent variable searched (BIC)
ARDL_MAX_ORDER = 2              # max lag order of each regressor searched (BIC)
KNN_K = 3                       # paper's best k
ANN_HIDDEN = (3,)               # one hidden layer, three neurons
ANN_LR = 0.3                    # learning rate
ANN_MOMENTUM = 0.2
ANN_ACTIVATIONS = ["logistic", "tanh", "relu"]
ANN_N_SEEDS = 5                 # average several random initialisations (reduces variance)
SVR_GRID = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.1],
    "epsilon": [0.01, 0.1, 0.5],
}
ML_Y_LAGS = 3                   # lags of inflation used as ML features
RANDOM_STATE = 42

MODEL_ORDER = ["AR", "Naive", "ARDL", "VAR", "ANN", "k-NN", "SVR"]
TIME_SERIES_MODELS = ["AR", "Naive", "ARDL", "VAR"]
ML_MODELS = ["ANN", "k-NN", "SVR"]
UNIVARIATE_MODELS = ["AR", "Naive"]   # all others count as multivariate (as in the paper)
