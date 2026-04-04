# DSS5104 CA2 — Deep Learning for Time-Series Forecasting

## Overview

This repository benchmarks **9 forecasting models** across **3 datasets** using a rigorous walk-forward evaluation protocol with 3 random seeds. Each model has its own independent pipeline that can be run separately.

### Models

| Model | Type | Architecture |
|-------|------|-------------|
| PatchTST | Deep Learning | Transformer (patch-based) |
| N-BEATS | Deep Learning | Deep residual (basis expansion) |
| TiDE | Deep Learning | MLP encoder-decoder |
| DeepAR | Deep Learning | RNN (autoregressive, probabilistic) |
| DLinear | Deep Learning | Single linear layer |
| TimesNet | Deep Learning | CNN (FFT → 2D tensors) |
| Seasonal Naive | Baseline | Repeat last seasonal pattern |
| AutoARIMA | Baseline | Classical statistical model |
| LightGBM | Baseline | Gradient-boosted trees with lag features |

### Datasets

| Dataset | Type | Domain | Series |
|---------|------|--------|--------|
| M4 Monthly | Univariate | Mixed (finance, demographics, etc.) | 500 sampled (default) |
| M5 | Hierarchical | Retail sales (Walmart) | 200 sampled (default) |
| Traffic | Multivariate | Transportation (SF road occupancy) | 50 sampled (default) |

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Data

Place the raw data in `../Data/`:
- `Data/M4/` — M4 competition CSVs
- `Data/M5/` — Walmart sales data
- `Data/Traffic.tsf` — Traffic TSF file

### 3. Run Experiments

Each model has its own pipeline that runs across all 3 datasets. Pipelines are fully independent — run any subset in any order.

**Run a single model** (e.g. PatchTST on M4 + M5 + Traffic):
```bash
cd Code_v2
python pipelines/run_patchtst.py
```

**Smoke test** (quick validation with minimal settings, ~5 min per model):
```bash
python pipelines/run_patchtst.py --smoke-test
python pipelines/run_seasonal_naive.py --smoke-test
```

**Run all models sequentially**:
```bash
python pipelines/run_all.py              # full run
python pipelines/run_all.py --smoke-test # quick validation
```

**Available pipelines** (each covers all 3 datasets):
```bash
python pipelines/run_seasonal_naive.py
python pipelines/run_auto_arima.py
python pipelines/run_lightgbm.py
python pipelines/run_patchtst.py
python pipelines/run_nbeats.py
python pipelines/run_tide.py
python pipelines/run_deepar.py
python pipelines/run_dlinear.py
python pipelines/run_timesnet.py
```

### 4. Aggregate & Plot Results

Run after one or more pipelines have completed (partial results are fine):
```bash
python analysis/aggregate_results.py
python analysis/plot_results.py
```

Results are saved to `results/` as flat CSVs (one per model per dataset, e.g. `PatchTST_M4.csv`).

## Project Structure

```
Code_v2/
├── config.py               # Central configuration (paths, seeds, hyperparameters)
├── requirements.txt        # Dependencies
├── data_prep/              # Dataset loading & formatting
│   ├── m4_prep.py
│   ├── m5_prep.py
│   └── traffic_prep.py
├── models/                 # Individual model definitions (one file per model)
│   ├── __init__.py         # ModelSpec dataclass
│   ├── seasonal_naive.py
│   ├── auto_arima.py
│   ├── lightgbm.py
│   ├── patchtst.py
│   ├── nbeats.py
│   ├── tide.py
│   ├── deepar.py
│   ├── dlinear.py
│   └── timesnet.py
├── evaluation/             # Evaluation engine
│   ├── walk_forward.py     # Sliding-window walk-forward driver (single-model)
│   ├── metrics.py          # MAE, MASE computation
│   └── timing.py           # Training time tracker
├── pipelines/              # Per-model pipeline scripts
│   ├── run_model.py        # Shared pipeline utility
│   ├── run_seasonal_naive.py
│   ├── run_auto_arima.py
│   ├── run_lightgbm.py
│   ├── run_patchtst.py
│   ├── run_nbeats.py
│   ├── run_tide.py
│   ├── run_deepar.py
│   ├── run_dlinear.py
│   ├── run_timesnet.py
│   └── run_all.py          # Orchestrator (runs all 9 sequentially)
├── analysis/               # Post-experiment analysis
│   ├── aggregate_results.py # Summary tables
│   └── plot_results.py      # Figures for report
└── results/                # Output (auto-created, flat per-model CSVs)
```

## Experimental Protocol

- **Walk-forward validation**: Sliding window with fixed-size training windows
- **Metrics**: MAE (primary) + MASE (scale-free, official M4 metric)
- **Seeds**: 3 random seeds per ML/DL model; mean ± std reported
- **Preprocessing**: Per-series standard normalization (handled by neuralforecast)

## Hardware

Report your hardware in the final report. Example:
- GPU: NVIDIA RTX 3090 / Apple M-series (MPS) / CPU only
- RAM: 16 GB+
- Estimated total runtime: 6-11 GPU hours
