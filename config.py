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
ROOT = Path(__file__).resolve().parent      # CA2/
DATA_DIR = ROOT / "Data"
RESULTS_DIR = ROOT / "results"

# ─── Random Seeds (3 seeds) ──────────────────────────────────────────────────
SEEDS = [42, 123, 456]

# ─── Shared DL Training Hyperparameters ──────────────────────────────────────
MAX_STEPS = 400      # reduced for MacBook Air (MPS); increase to 1000+ on GPU server
BATCH_SIZE = 16      # reduced for 16 GB RAM
EARLY_STOP_PATIENCE = 5
VAL_CHECK_STEPS = 50  # validate every 50 steps

# ─── Per-model Learning Rates ────────────────────────────────────────────────
# Transformer-based and MLP-encoder models (PatchTST, TiDE, TimesNet) are
# sensitive to high LRs — 1e-4 is the standard recommendation from each
# paper (Nie et al. 2023, Das et al. 2023, Wu et al. 2023).
# MLP residual (N-BEATS) and RNN (DeepAR) models are stable at 1e-3.
# DLinear is a single linear layer — 1e-3 is fine.
LR_TRANSFORMER = 1e-4   # PatchTST, TimesNet, TiDE
LR_MLP = 1e-3           # N-BEATS, DLinear
LR_RNN = 1e-3           # DeepAR

# ─── M4 Dataset Config (Monthly subset) ──────────────────────────────────────
M4_CONFIG = {
    "name": "M4",
    "freq": "ME",         # pandas month-end frequency
    "season_length": 12,
    "horizon": 18,        # official M4 Monthly forecast horizon
    "input_size": 36,     # 2× horizon lookback
    "train_csv": DATA_DIR / "M4" / "Monthly-train.csv",
    "test_csv":  DATA_DIR / "M4" / "Monthly-test.csv",
    "info_csv":  DATA_DIR / "M4" / "m4_info.csv",
    "n_series_sample": 500,     # reduced for 16 GB MacBook Air (use 1000+ on GPU server)
    "walk_forward_windows": 2,
    "max_train_size": 144,      # 12 years of monthly history
}

# ─── M5 Dataset Config ───────────────────────────────────────────────────────
M5_CONFIG = {
    "name": "M5",
    "freq": "D",          # daily
    "season_length": 7,
    "horizon": 28,
    "input_size": 56,     # 2× horizon lookback
    "sales_csv":    DATA_DIR / "M5" / "sales_train_evaluation.csv",
    "calendar_csv": DATA_DIR / "M5" / "calendar.csv",
    "n_series_sample": 200,     # reduced for 16 GB MacBook Air (use 500+ on GPU server)
    "walk_forward_windows": 2,
    "max_train_size": 365,      # 1 year of daily history
}

# ─── Traffic Dataset Config ──────────────────────────────────────────────────
TRAFFIC_CONFIG = {
    "name": "Traffic",
    "freq": "h",          # hourly
    "season_length": 24,
    "horizon": 24,        # 24-hour-ahead prediction
    "input_size": 168,    # 7 days lookback (7 × 24)
    "data_file": DATA_DIR / "Traffic.tsf",
    "n_series_sample": 50,      # reduced for 16 GB MacBook Air (use 100+ on GPU server)
    "walk_forward_windows": 2,
    "max_train_size": 672,      # 4 weeks (24 × 28) of hourly history
}

# ─── PatchTST Hyperparameters ─────────────────────────────────────────────────
# Defaults follow Nie et al. (2023). patch_len and stride are overridden
# per-dataset in pipelines/run_patchtst.py to align patches with seasonal units.
# n_heads raised from 4 → 8: paper typically uses 16; 8 is a reasonable
# memory-constrained compromise that still provides diverse attention patterns.
PATCHTST_PARAMS = {
    "patch_len": 16,        # default; overridden per-dataset in run_patchtst.py
    "stride": 8,            # default; overridden per-dataset in run_patchtst.py
    "n_heads": 8,           # ↑ from 4; Nie et al. use 16, we use 8 for memory
    "hidden_size": 128,     # paper default
    "encoder_layers": 3,    # paper default
}

# ─── N-BEATS Hyperparameters ──────────────────────────────────────────────────
# Interpretable N-BEATS-I (Oreshkin et al. 2019): trend + seasonality stacks
# with fixed polynomial / Fourier bases — provides explainability valued in the
# report. n_blocks=[3,3] and mlp_units=[[512,512],[512,512]] both match the
# original paper. mlp_units was previously reduced for memory; now that each
# model runs in its own pipeline, the full paper default is feasible.
NBEATS_PARAMS = {
    "stack_types": ["trend", "seasonality"],
    "n_blocks": [3, 3],                            # original paper default
    "mlp_units": [[512, 512], [512, 512]],          # ↑ restored to paper default
}

# ─── TiDE Hyperparameters ─────────────────────────────────────────────────────
# Das et al. (2023) paper default is hidden_size=128; we use 256 for extra
# capacity. All other params match paper defaults.
TIDE_PARAMS = {
    "hidden_size": 256,          # 2× paper default for extra capacity
    "decoder_output_dim": 32,    # paper default
    "num_encoder_layers": 2,     # paper default
    "num_decoder_layers": 2,     # paper default
}

# ─── DeepAR Hyperparameters ───────────────────────────────────────────────────
# hidden_size raised from 64 → 128, matching NeuralForecast's own documented
# default and the standard configuration in Salinas et al. (2020).
DEEPAR_PARAMS = {
    "hidden_size": 128,   # ↑ from 64; NeuralForecast default & paper standard
    "n_layers": 2,        # paper default
}

# ─── TimesNet Hyperparameters ─────────────────────────────────────────────────
# Wu et al. (2023): top_k=5 is the paper default for forecasting tasks.
# TimesNet uses FFT to automatically identify dominant frequency components,
# so top_k does NOT need to vary by dataset frequency — the FFT handles that.
TIMESNET_PARAMS = {
    "top_k": 5,          # paper default for forecasting; FFT auto-detects periods
    "num_kernels": 6,    # inception block kernel sizes
    "d_model": 64,       # paper default
    "d_ff": 64,          # paper default
    "e_layers": 2,       # paper default
}

# ─── LightGBM Hyperparameters ────────────────────────────────────────────────
# Conservative configuration suitable for time-series lag-feature regression.
# num_leaves=31 is the LightGBM default; lr=0.05 balances speed and stability
# at n_estimators=500 (within the recommended 100–500 range).
LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
}
