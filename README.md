# DSS5104 CA2 — Deep Learning for Time-Series Forecasting

This project benchmarks **9 forecasting models** across **3 datasets** using a rigorous sliding-window walk-forward evaluation protocol with 3 random seeds. The central research question is: *when, if ever, is the added complexity of a deep learning model justified over classical or linear baselines?*

---

## Models

| Model | Category | Architecture |
|-------|----------|-------------|
| PatchTST | Deep Learning | Transformer (patch-based input) |
| N-BEATS | Deep Learning | Deep residual with basis expansion (trend + seasonality stacks) |
| TiDE | Deep Learning | MLP encoder-decoder |
| DeepAR | Deep Learning | RNN (LSTM, autoregressive, probabilistic output) |
| **DLinear** | **Linear Baseline** | **Single linear layer mapping past → future (Zeng et al., 2023)** |
| **TimesNet** | **Deep Learning (Additional)** | **CNN with FFT-based period detection and 2D convolution** |
| Seasonal Naive | Classical Baseline | Repeat last observed seasonal cycle |
| AutoARIMA | Classical Baseline | Auto-selected ARIMA via information criteria (statsforecast) |
| LightGBM | Classical Baseline (ML) | Gradient-boosted trees with hand-crafted lag features |

> **TimesNet** is the additional model of choice (one extra beyond the four required). It was selected for its convolutional inductive bias: it uses FFT to identify dominant periodic components, reshapes the 1D time series into 2D tensors aligned to those periods, and applies inception-style 2D convolutions. This provides a distinctly different architectural perspective (CNN) compared to the Transformer, MLP, and RNN models also evaluated.

> **DLinear** is the critical linear baseline. It applies a single linear layer directly from past window to all future steps — included to ground Transformer comparisons following Zeng et al. (2023), who showed it can match or beat complex architectures on standard benchmarks.

---

## Datasets

| Dataset | Type | Domain | Series Sampled | Frequency | Forecast Horizon |
|---------|------|--------|----------------|-----------|-----------------|
| M4 Monthly | Univariate | Mixed (finance, demographics, industry, etc.) | 500 | Monthly | 18 steps |
| M5 | Hierarchical | Retail sales (Walmart) | 200 | Daily | 28 steps |
| Traffic | Multivariate | SF Bay Area road occupancy | 50 | Hourly | 24 steps |

Series counts are reduced from the full datasets due to hardware constraints (16 GB MacBook Air M4). Increase `n_series_sample` in `config.py` to run at full scale on a GPU server.

### Data Download

The datasets are **not included** in this repository. Download them from the following sources and place them as described:

- **M4 Monthly** — Download from the [Kaggle M4 Forecasting Competition Dataset](https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset). Place all CSVs in `Data/M4/`.
- **M5** — Download from [Kaggle M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data). Place all CSVs in `Data/M5/`.
- **Traffic** — Download `traffic_hourly_dataset.tsf` from the [Monash Time Series Repository](https://zenodo.org/records/4656132). Rename it to `Traffic.tsf` and place it in `Data/`.

Expected directory layout after downloading:

```
Data/
├── M4/
│   ├── Monthly-train.csv
│   ├── Monthly-test.csv
│   └── m4_info.csv          # (and other M4 frequency CSVs, unused)
├── M5/
│   ├── sales_train_evaluation.csv
│   ├── calendar.csv
│   ├── sell_prices.csv
│   ├── sample_submission.csv
│   └── sales_train_validation.csv
└── Traffic.tsf
```

---

## Setup

### Requirements

- **Python 3.12** (tested; other 3.10+ versions may work but are not guaranteed)
- Apple Silicon Mac with MPS backend, or a CUDA-enabled GPU, or CPU-only (slower)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Apple Silicon / MPS Note

On Apple Silicon Macs, DeepAR uses a Student-t distribution whose sampling operation (`aten::_standard_gamma`) is not yet implemented on MPS. The following environment variable is **set automatically** in `config.py` to fall back to CPU for that operation only:

```python
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
```

No manual action is needed, but be aware that DeepAR will partially run on CPU even on an MPS machine.

---

## Running Experiments

Each model has its own independent pipeline that runs across all 3 datasets. Pipelines can be run in any order and are fully independent of each other.

### Run a Single Model

```bash
python pipelines/run_patchtst.py       # PatchTST on M4 + M5 + Traffic
python pipelines/run_nbeats.py         # N-BEATS on M4 + M5 + Traffic
python pipelines/run_tide.py
python pipelines/run_deepar.py
python pipelines/run_dlinear.py
python pipelines/run_timesnet.py
python pipelines/run_seasonal_naive.py
python pipelines/run_auto_arima.py
python pipelines/run_lightgbm.py
```

### Run All Models Sequentially

```bash
python pipelines/run_all.py
```

### Smoke Test (Quick Validation, ~5 min per model)

Runs with minimal series and steps to verify the pipeline end-to-end before committing to a full run:

```bash
python pipelines/run_patchtst.py --smoke-test
python pipelines/run_all.py --smoke-test
```

### Reproduce All Report Results (Full Sequence)

To reproduce every result in the final report from scratch:

```bash
# Step 1: Run all model pipelines
python pipelines/run_all.py

# Step 2: Aggregate results into summary tables
python analysis/aggregate_results.py

# Step 3: Generate all figures
python analysis/plot_results.py
```

All output CSVs are saved to `results/` and all figures to `results/plots/`.

---

## Experimental Protocol

### Preprocessing

- All series are converted to **long format** (`unique_id`, `ds`, `y`).
- **Per-series standard normalization** (zero mean, unit variance) is applied by the NeuralForecast library internally during model fitting. No global normalization is applied, preserving the individual scale of each series.
- **No differencing or detrending** is applied manually. Models are expected to learn any necessary transformations.
- **Missing values / insufficient history**: series with fewer observations than `input_size` are filtered out before training in each walk-forward window (see `evaluation/walk_forward.py`).
- **LightGBM feature engineering**: lag features are constructed from the lagged target values of each series using the `mlforecast` library. No additional hand-crafted features (e.g., calendar features) are used.

### Walk-Forward Evaluation

A **sliding-window** approach is used with **2 windows** per dataset. In each window, the training set is capped at a maximum history length (`max_train_size` in `config.py`) and the test set is the most recent `horizon` steps. Windows move backward in time so test sets never overlap. Sliding window was chosen over expanding window to keep training time bounded on constrained hardware, and to test model adaptability across different time periods rather than simply accumulating more data.

### Metrics

- **MAE** (Mean Absolute Error): primary accuracy metric, computed per series and averaged. Scale-dependent — used for within-dataset comparisons only.
- **MASE** (Mean Absolute Scaled Error): scale-free metric that normalises MAE by the in-sample naive seasonal error. MASE < 1 indicates better-than-naive performance. Used for cross-dataset comparison and as the official M4 competition metric.

RMSE and MAPE are not reported: RMSE over-penalises outliers in ways not always meaningful for business forecasting, and MAPE is undefined when actuals are zero (common in M5 retail data).

### Seeds

- ML/DL models: **3 random seeds** (42, 123, 456) × 2 windows = **6 runs** per model per dataset.
- Statistical baselines (Seasonal Naive, AutoARIMA): deterministic — run with 2 seeds for logging consistency but results are seed-independent.
- All reported metrics are **mean ± standard deviation** across seeds and windows.

---

## Hardware and Runtime

| Item | Details |
|------|---------|
| Machine | Apple MacBook Air M4 |
| RAM | 16 GB unified memory |
| Backend | Apple MPS (GPU) with automatic CPU fallback for unsupported ops |
| Python | 3.12 |
| Total wall-clock runtime | ~4.4 hours (all 9 models × 3 datasets × 3 seeds × 2 windows) |

**Per-model total training time** (all datasets combined):

| Model | Total Time (s) |
|-------|---------------|
| LightGBM | ~9 |
| Seasonal Naive | ~37 |
| DLinear | ~83 |
| TiDE | ~209 |
| N-BEATS | ~224 |
| PatchTST | ~819 |
| DeepAR | ~2,004 |
| AutoARIMA | ~1,319 |
| TimesNet | ~10,994 |

Note: TimesNet's cost (~3 hours alone) dominates total runtime. On a GPU server, absolute times will differ but relative ordering is expected to hold.

---

## Key Findings

- **N-BEATS** is the most consistently strong model, achieving the best MAE and MASE on both M4 and M5, and ranking second on Traffic. Its explicit basis expansion for trend and seasonality provides a strong inductive bias without the overhead of attention mechanisms.
- **DLinear (linear baseline)** ranks first on Traffic, supporting Zeng et al.'s finding that simple linear mappings are sufficient for strongly periodic, regular time series.
- **AutoARIMA** remains competitive on M4 (second-best overall), demonstrating the strength of classical statistical methods on large univariate benchmarks.
- **DeepAR** consistently underperforms across all three datasets, suggesting its autoregressive RNN structure is poorly suited to these benchmarks.
- **TimesNet** is computationally the most expensive model (~10,994s total) but does not deliver commensurate accuracy gains — its cost-effectiveness is low compared to N-BEATS.
- **LightGBM** is the fastest model but struggles on M5 (MASE 2.11), where sparse retail demand patterns make lag features unreliable.

---

## Project Structure

```
CA2/
├── config.py                   # Central configuration (paths, seeds, hyperparameters)
├── requirements.txt            # Python dependencies
├── data_prep/                  # Dataset loading and formatting
│   ├── m4_prep.py              # M4 Monthly loading and long-format conversion
│   ├── m5_prep.py              # M5 loading and long-format conversion
│   └── traffic_prep.py         # Traffic .tsf loading and long-format conversion
├── models/                     # Individual model definitions (one file per model)
│   ├── __init__.py             # ModelSpec dataclass
│   ├── seasonal_naive.py
│   ├── auto_arima.py
│   ├── lightgbm.py
│   ├── patchtst.py
│   ├── nbeats.py
│   ├── tide.py
│   ├── deepar.py
│   ├── dlinear.py
│   └── timesnet.py
├── evaluation/                 # Evaluation engine
│   ├── walk_forward.py         # Sliding-window walk-forward driver
│   ├── metrics.py              # MAE and MASE computation
│   └── timing.py               # Training time and GPU memory tracker
├── pipelines/                  # Per-model pipeline scripts (one per model)
│   ├── run_model.py            # Shared pipeline utility used by all pipelines
│   ├── run_seasonal_naive.py
│   ├── run_auto_arima.py
│   ├── run_lightgbm.py
│   ├── run_patchtst.py
│   ├── run_nbeats.py
│   ├── run_tide.py
│   ├── run_deepar.py
│   ├── run_dlinear.py
│   ├── run_timesnet.py
│   └── run_all.py              # Orchestrator: runs all 9 pipelines sequentially
├── analysis/                   # Post-experiment aggregation and visualisation
│   ├── aggregate_results.py    # Produces summary_table.csv, pivot tables
│   └── plot_results.py         # Produces all figures in results/plots/
├── results/                    # Auto-created; flat CSVs and plots (excluded from git)
│   ├── <Model>_<Dataset>.csv   # Per-model, per-dataset raw results
│   ├── summary_table.csv       # Aggregated mean ± std per model × dataset
│   ├── pivot_mae.csv
│   ├── pivot_mase.csv
│   ├── computational_costs.csv
│   └── plots/                  # Figures used in the final report
└── Data/                       # Raw datasets — not included in repo (see Data Download)
```

---

## References

- Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers* (PatchTST). ICLR 2023.
- Oreshkin, B. N. et al. (2019). *N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting*. arXiv:1905.10437.
- Das, A. et al. (2023). *Long-term Forecasting with TiDE: Time-series Dense Encoder*. arXiv:2304.08424.
- Salinas, D. et al. (2020). *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks*. International Journal of Forecasting.
- Wu, H. et al. (2023). *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis*. ICLR 2023.
- Zeng, A. et al. (2023). *Are Transformers Effective for Time Series Forecasting?* (DLinear). AAAI 2023.
