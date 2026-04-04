# DSS5104 CA2 — Deep Learning for Time-Series Forecasting

## Overview

This repository benchmarks **9 forecasting models** across **3 datasets** using a rigorous walk-forward evaluation protocol with 5 random seeds.

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

**Smoke test** (quick validation, ~5 min):
```bash
cd Code
python runners/run_all.py --smoke-test
```

**Full run** (~6-11 GPU hours):
```bash
cd Code
python runners/run_all.py
```

**Individual datasets**:
```bash
python runners/run_m4.py
python runners/run_m5.py
python runners/run_traffic.py
```

### 4. Aggregate Results

```bash
python analysis/aggregate_results.py
python analysis/plot_results.py
```

Results are saved to `Code/results/`.

## Project Structure

```
Code/
├── config.py               # Central configuration
├── requirements.txt         # Dependencies
├── data_prep/               # Dataset loading & formatting
│   ├── m4_prep.py
│   ├── m5_prep.py
│   └── traffic_prep.py
├── models/                  # Model definitions
│   ├── baselines.py         # Seasonal Naive, AutoARIMA
│   ├── lgbm_model.py        # LightGBM
│   ├── dl_models.py         # PatchTST, N-BEATS, TiDE, DeepAR, DLinear
│   └── timesnet_model.py    # TimesNet
├── evaluation/              # Evaluation engine
│   ├── walk_forward.py      # Sliding-window walk-forward driver
│   ├── metrics.py           # MAE, MASE computation
│   └── timing.py            # Training time tracker
├── runners/                 # End-to-end experiment scripts
│   ├── run_m4.py
│   ├── run_m5.py
│   ├── run_traffic.py
│   └── run_all.py           # Master orchestrator
├── analysis/                # Post-experiment analysis
│   ├── aggregate_results.py # Summary tables
│   └── plot_results.py      # Figures for report
└── results/                 # Output (auto-created)
    ├── m4/
    ├── m5/
    ├── traffic/
    └── plots/
```

## Experimental Protocol

- **Walk-forward validation**: Sliding window with fixed-size training windows
- **Metrics**: MAE (primary) + MASE (scale-free, official M4 metric)
- **Seeds**: 5 random seeds per DL model; mean ± std reported
- **Preprocessing**: Per-series standard normalization (handled by neuralforecast)

## Hardware

Report your hardware in the final report. Example:
- GPU: NVIDIA RTX 3090 / Apple M-series (MPS) / CPU only
- RAM: 16 GB+
- Estimated total runtime: 6-11 GPU hours
