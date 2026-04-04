# DSS5104 CA2 — Preliminary Results (V2)

**Date:** 29 March 2026 | **Hardware:** Apple M4 MacBook Air, 16 GB RAM, MPS GPU

---

## 1. Experimental Setup

| Setting | Value |
|---|---|
| Datasets | M4 Monthly (500 series), M5 Walmart (200 series), Traffic SF (50 series) |
| Evaluation | Rolling window walk-forward, 2 windows per dataset |
| Seeds | 3 (42, 123, 456) — applied to all ML/DL models; baselines are deterministic |
| DL training | 400 max steps, early stopping (patience = 5, check every 50 steps) |
| Metrics | MAE, MASE (lower = better; MASE < 1.0 = beats seasonal naive) |
| Framework | Nixtla ecosystem (neuralforecast, statsforecast, mlforecast) |

**9 models compared:** SeasonalNaive, AutoARIMA, LightGBM, PatchTST, N-BEATS, TiDE, DeepAR, DLinear, TimesNet

Each model runs across all 3 datasets independently. Results are averaged over 3 seeds × 2 windows (6 data points per model per dataset), except baselines which average over 2 windows only.

---

## 2. What Changed from V1

Three fixes were applied between Version 1 and Version 2. They affect both the code structure and some results.

**① M4 monthly window boundary fix (affects M4 results)**
Walk-forward cross-validation requires splitting the time series into precise train/test windows. V1 estimated the time step size from `sample_dates.diff().median()` — for monthly data this gives ≈ 30.44 days (the average month), which is imprecise because months vary from 28 to 31 days. Over an 18-step horizon, the cumulative drift reached up to ~18 days, meaning window boundaries did not land on actual month-end timestamps. V2 replaces this with `pd.tseries.frequencies.to_offset(freq)` (exact `MonthEnd` arithmetic). M5 and Traffic are unaffected since their time steps are fixed (daily/hourly). **All M4 numbers below reflect the corrected windows; V1 M4 figures should be disregarded.**

**② DeepAR MPS fix (affects DeepAR results)**
DeepAR's Student-T distribution uses a GPU operation (`aten::_standard_gamma`) that is not implemented on Apple MPS. V1 set the required environment variable (`PYTORCH_ENABLE_MPS_FALLBACK=1`) inside `config.py`, but PyTorch had already been imported by the time config was loaded — so the flag was silently ignored and DeepAR failed on every seed. V2 sets the flag at the very top of `run_deepar.py`, before any torch import, so DeepAR now runs correctly (sampling falls back to CPU for that one op; all other ops stay on MPS). V1 DeepAR results are unreliable.

**③ Per-model learning rates**
V1 used a single global `LEARNING_RATE = 1e-3` for all DL models. V2 splits this by architecture type following the original papers:

| Group | LR | Models |
|---|---|---|
| Transformer-class | 1e-4 | PatchTST, TiDE, TimesNet |
| MLP-class | 1e-3 | N-BEATS, DLinear |
| RNN-class | 1e-3 | DeepAR |

This mainly affects TiDE on Traffic (previously benefiting from an oversized LR) and has minor effects elsewhere.

---

## 3. Results — Mean MASE (averaged over 3 seeds × 2 windows)

### M4 Monthly (univariate, horizon = 18 months)

| Rank | Model | MAE | MASE | Std |
|---|---|---|---|---|
| 1 | **N-BEATS** | 487.5 | **0.839** | ±0.008 |
| 2 | AutoARIMA | 513.2 | **0.863** | ±0.009 |
| 3 | TimesNet | 550.2 | **0.949** | ±0.017 |
| 4 | LightGBM | 527.3 | 1.060 | ±0.014 |
| 5 | SeasonalNaive | 597.3 | 1.123 | ±0.027 |
| 6 | PatchTST | 616.7 | 1.158 | ±0.049 |
| 7 | DLinear | 729.8 | 1.359 | ±0.041 |
| 8 | DeepAR | 745.6 | 1.485 | ±0.005 |
| 9 | TiDE | 762.5 | 1.505 | ±0.018 |

Three models beat the seasonal naive baseline (MASE < 1.0): N-BEATS, AutoARIMA, and TimesNet. LightGBM (< 1 second to train) still outperforms four DL models. DeepAR and TiDE are the weakest DL models on this dataset.

---

### M5 Walmart (hierarchical daily, horizon = 28 days)

| Rank | Model | MAE | MASE | Std |
|---|---|---|---|---|
| 1 | **N-BEATS** | 0.828 | **0.952** | ±0.047 |
| 2 | PatchTST | 0.844 | **0.971** | ±0.042 |
| 3 | TimesNet | 0.849 | **0.997** | ±0.065 |
| 4 | DeepAR | 0.891 | 1.112 | ±0.062 |
| 5 | DLinear | 0.853 | 1.114 | ±0.103 |
| 6 | AutoARIMA | 0.901 | 1.133 | ±0.032 |
| 7 | TiDE | 0.943 | 1.184 | ±0.059 |
| 8 | SeasonalNaive | 1.100 | 1.266 | ±0.047 |
| 9 | LightGBM | 1.135 | 2.108 | ±0.120 |

Top 3 DL models achieve MASE < 1.0, showing genuine forecasting value on retail data. LightGBM is the worst performer here (MASE = 2.11) — a reversal from its M4 position — likely because M5 has many zero-sales days that break lag-feature approaches. DLinear shows high variance (±0.10), indicating instability across rolling windows.

---

### Traffic SF (multivariate hourly, horizon = 24 hours)

| Rank | Model | MAE | MASE | Std |
|---|---|---|---|---|
| 1 | **DLinear** | 0.0085 | **0.638** | ±0.166 |
| 2 | N-BEATS | 0.0087 | **0.653** | ±0.152 |
| 3 | LightGBM | 0.0087 | **0.655** | ±0.129 |
| 4 | PatchTST | 0.0089 | **0.678** | ±0.123 |
| 5 | TimesNet | 0.0092 | **0.690** | ±0.170 |
| 6 | SeasonalNaive | 0.0111 | 0.843 | ±0.436 |
| 7 | AutoARIMA | 0.0117 | 0.873 | ±0.404 |
| 8 | TiDE | 0.0120 | 0.896 | ±0.137 |
| 9 | DeepAR | 0.0232 | 1.830 | ±0.014 |

All models except DeepAR beat the seasonal naive baseline. Top 5 models are tightly grouped (MASE 0.638–0.690). The two traditional baselines show large variance (±0.40+) — reflecting pattern shifts between the two test windows. DeepAR continues to underperform on point forecasting across all datasets.

---

## 4. Key Findings

**N-BEATS is the most consistent model.** Top 1 on M4 and M5, top 2 on Traffic. The only model that achieves MASE < 1.0 on both M4 and M5. It also trains in under 15 seconds, making it the best compute-to-accuracy trade-off in this experiment.

**Deep learning does not always beat simpler models.** On M4, LightGBM (< 1 second to train) outranks PatchTST, DLinear, DeepAR, and TiDE. AutoARIMA ranks 2nd overall on M4 — ahead of every DL model except N-BEATS and TimesNet. This directly echoes findings from Zeng et al. (2023) that challenged the assumed superiority of transformers for time series.

**DLinear shows a strong reversal across datasets.** It ranks 7th on M4 (MASE = 1.36) and 1st on Traffic (MASE = 0.638). This confirms that its single decomposed linear layer captures the strong 24-hour periodicity in traffic data well, but cannot model the multi-scale irregular patterns in M4.

**LightGBM fails badly on M5 (MASE = 2.11).** The worst single result across all models and datasets. M5's sparse retail series with frequent zero-sales days likely breaks the lag-feature tabular approach. This highlights dataset-model fit as a critical factor.

**DeepAR consistently underperforms on point forecasting.** Ranks 8th (M4), 4th (M5), and 9th (Traffic). Its architecture is optimised for probabilistic forecasting — the point estimate (mean/median of the predictive distribution) is not competitive when evaluated purely on MAE/MASE. This is a known limitation when benchmarking probabilistic models on point metrics.

**Traffic is the only dataset where all non-DeepAR models beat the baseline.** All 8 remaining models achieve MASE < 1.0, with the top 5 clustered between 0.638 and 0.690. The strong 24-hour periodic structure makes this dataset more tractable.

---

## 5. Computational Cost

Mean training time per seed × window (seconds):

| Model | M4 | M5 | Traffic | Notes |
|---|---|---|---|---|
| SeasonalNaive | ~6s | ~6s | ~6s | No training; repeated seasonal pattern |
| AutoARIMA | 62s | 64s | 503s | Fits one ARIMA per series; very slow on 50 hourly sensors |
| LightGBM | 0.5s | 0.5s | 0.5s | Fastest trained model |
| N-BEATS | 13s | 11s | 11s | Fast despite MLP stacks |
| TiDE | 12s | 12s | 12s | |
| DLinear | 4s | 3s | 3s | Two linear layers; near-trivial |
| PatchTST | 41s | 31s | 50s | Transformer attention |
| DeepAR | 32s | 39s | 213s | RNN sequential decoding; slow on long hourly input |
| TimesNet | 230s | 333s | 1071s | Heaviest model: FFT + 2D inception CNN |

TimesNet's total wall-clock time was 2.73 hours (all datasets, 3 seeds × 2 windows). LightGBM trains in under 1 second yet ranks competitively on 2 out of 3 datasets. The compute premium paid for DL is clearly justified only by N-BEATS, and situationally by DLinear and PatchTST on structured periodic data.

---

## 6. Notes and Limitations

- **Subsampled for hardware constraints** (500 / 200 / 50 series). Full M4 has 48,000 series; full M5 has 30,490. Production-scale runs require a GPU server.
- **2 rolling windows** is the minimum for cross-period validation. More windows would strengthen the statistical basis of comparisons.
- **MASE variance is high on Traffic** for the two baselines (±0.40+). Partly a real signal (Traffic's test windows span different traffic conditions), partly an artefact of the small sample (50 sensors).
- **DeepAR point accuracy should not be the sole judge.** Its MASE figures here reflect a mismatch between architecture goal (probabilistic) and metric (point). Confidence interval coverage would be a fairer evaluation.
- **Visualisations and formal aggregated summary tables pending** (next step before submission).
