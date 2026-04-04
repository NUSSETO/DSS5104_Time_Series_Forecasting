"""
Central configuration for DSS5104 CA2 — Deep Learning Time-Series Forecasting.

All paths, hyperparameters, dataset configs, and random seeds are defined here
so that the entire experiment is controlled from a single place.
"""

import os
from pathlib import Path

# MPS fallback: DeepAR's Student-T sampling uses aten::_standard_gamma which
# is not implemented on Apple MPS. This env var makes PyTorch fall back to CPU
# for unsupported ops only, while all other ops keep running on MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]  # CA2/
DATA_DIR = ROOT / "Data"
CODE_DIR = ROOT / "Code"
RESULTS_DIR = CODE_DIR / "results"

# ─── Random Seeds (3 seeds) ──────────────────────────────────────────────────
SEEDS = [42, 123, 456]

# ─── Common DL Hyperparameters ───────────────────────────────────────────────
MAX_STEPS = 400           # reduced for MacBook Air (MPS); increase to 1000+ on GPU server
BATCH_SIZE = 16           # reduced for 16GB RAM
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 5
VAL_CHECK_STEPS = 50      # how often to check validation loss

# ─── M4 Dataset Config (Monthly subset) ──────────────────────────────────────
M4_CONFIG = {
    "name": "M4",
    "freq": "ME",  # pandas month-end frequency
    "season_length": 12,
    "horizon": 18,  # official M4 Monthly forecast horizon
    "input_size": 36,  # 2× horizon lookback
    "train_csv": DATA_DIR / "M4" / "Monthly-train.csv",
    "test_csv": DATA_DIR / "M4" / "Monthly-test.csv",
    "info_csv": DATA_DIR / "M4" / "m4_info.csv",
    "n_series_sample": 500,   # reduced for 16GB MacBook Air (use 1000+ on GPU server)
    "walk_forward_windows": 2,
    "max_train_size": 144,    # 12 years of monthly history to strictly bound training
}

# ─── M5 Dataset Config ───────────────────────────────────────────────────────
M5_CONFIG = {
    "name": "M5",
    "freq": "D",  # daily
    "season_length": 7,
    "horizon": 28,
    "input_size": 56,  # 2× horizon lookback
    "sales_csv": DATA_DIR / "M5" / "sales_train_evaluation.csv",
    "calendar_csv": DATA_DIR / "M5" / "calendar.csv",
    "n_series_sample": 200,   # reduced for 16GB MacBook Air (use 500+ on GPU server)
    "walk_forward_windows": 2,
    "max_train_size": 365,    # 1 year of daily history to strictly bound training
}

# ─── Traffic Dataset Config ──────────────────────────────────────────────────
TRAFFIC_CONFIG = {
    "name": "Traffic",
    "freq": "h",  # hourly
    "season_length": 24,
    "horizon": 24,  # 24-hour-ahead prediction
    "input_size": 168,  # 7 days lookback
    "data_file": DATA_DIR / "Traffic.tsf",
    "n_series_sample": 50,    # reduced for 16GB MacBook Air (use 100+ on GPU server)
    "walk_forward_windows": 2,  # reduced from 3 to save time
    "max_train_size": 672,   # 4 weeks (24*28) of hourly history 
}

# ─── Model-specific Hyperparameters ──────────────────────────────────────────
PATCHTST_PARAMS = {
    "patch_len": 16,
    "stride": 8,
    "n_heads": 4,
    "hidden_size": 128,
    "encoder_layers": 3,
}

NBEATS_PARAMS = {
    "stack_types": ["trend", "seasonality"],
    "n_blocks": [2, 2],             # reduced from [3,3] for memory
    "mlp_units": [[256, 256], [256, 256]],  # reduced from 512 for memory
}

TIDE_PARAMS = {
    "hidden_size": 256,
    "decoder_output_dim": 32,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
}

DEEPAR_PARAMS = {
    "hidden_size": 64,
    "n_layers": 2,
}

TIMESNET_PARAMS = {
    "top_k": 5,
    "num_kernels": 6,
    "d_model": 64,
    "d_ff": 64,
    "e_layers": 2,
}

LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
}
