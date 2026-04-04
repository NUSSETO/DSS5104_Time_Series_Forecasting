"""
Walk-forward (sliding-window) evaluation driver.

Implements the core evaluation loop:
  For each sliding window:
    For each seed:
      For each model group:
        Fit → Predict → Compute metrics → Record timing

This is the main engine that ties data, models, and evaluation together.
"""

import pandas as pd
import numpy as np
import traceback
from pathlib import Path

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from evaluation.metrics import compute_metrics_per_series
from evaluation.timing import Timer


def _sliding_window_splits(
    df: pd.DataFrame,
    horizon: int,
    input_size: int,
    n_windows: int,
    max_train_size: int | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate sliding-window train/test splits.

    Each window holds out `horizon` time steps for testing. Windows move
    forward so that test sets do not overlap.

    Parameters
    ----------
    df : DataFrame
        Full dataset in long format ['unique_id', 'ds', 'y'].
    horizon : int
        Number of time steps to forecast.
    input_size : int
        Minimum number of time steps needed for model input.
    n_windows : int
        Number of sliding windows.

    Returns
    -------
    List of (df_train, df_test) tuples.
    """
    splits = []

    # Determine the global max date across all series
    max_date = df["ds"].max()

    # Determine frequency from the data
    sample_uid = df["unique_id"].iloc[0]
    sample_dates = df[df["unique_id"] == sample_uid]["ds"].sort_values()
    if len(sample_dates) >= 2:
        freq_delta = sample_dates.diff().median()
    else:
        freq_delta = pd.Timedelta(days=1)

    for w in range(n_windows):
        # Test window: ends at max_date - w*horizon*freq_delta
        test_end = max_date - w * horizon * freq_delta
        test_start = test_end - (horizon - 1) * freq_delta

        # Train: everything before test_start
        train_cutoff = test_start - freq_delta

        if max_train_size is not None:
            train_start = train_cutoff - (max_train_size * freq_delta)
            df_train_w = df[(df["ds"] > train_start) & (df["ds"] <= train_cutoff)].copy()
        else:
            df_train_w = df[df["ds"] <= train_cutoff].copy()

        df_test_w = df[(df["ds"] >= test_start) & (df["ds"] <= test_end)].copy()

        # Only keep series that have enough history
        series_lens = df_train_w.groupby("unique_id").size()
        valid_series = series_lens[series_lens >= input_size].index
        df_train_w = df_train_w[df_train_w["unique_id"].isin(valid_series)]
        df_test_w = df_test_w[df_test_w["unique_id"].isin(valid_series)]

        if len(df_test_w) == 0 or len(df_train_w) == 0:
            print(f"  [Walk-forward] Window {w+1}: skipped (insufficient data)")
            continue

        n_series = df_train_w["unique_id"].nunique()
        print(f"  [Walk-forward] Window {w+1}: {n_series} series, "
              f"train up to {train_cutoff}, test {test_start} → {test_end}")
        splits.append((df_train_w, df_test_w))

    # Reverse so earliest window comes first
    splits.reverse()
    return splits


def run_walk_forward(
    df_full: pd.DataFrame,
    dataset_name: str,
    horizon: int,
    input_size: int,
    freq: str,
    season_length: int,
    n_windows: int,
    seeds: list[int],
    results_dir: Path,
    get_baseline_fn,
    get_lgbm_fn,
    get_dl_fn,
    max_steps: int | None = None,
    max_train_size: int | None = None,
) -> pd.DataFrame:
    """Run the full walk-forward evaluation for one dataset.

    Parameters
    ----------
    df_full : DataFrame
        Complete dataset in long format.
    dataset_name : str
        Name for logging/saving (e.g., 'M4', 'M5', 'Traffic').
    horizon, input_size : int
        Forecast horizon and lookback window.
    freq : str
        Pandas frequency string.
    season_length : int
        Seasonal period.
    n_windows : int
        Number of sliding windows.
    seeds : list[int]
        Random seeds for DL models.
    results_dir : Path
        Directory to save results.
    get_baseline_fn, get_lgbm_fn, get_dl_fn : callable
        Factory functions returning model objects. get_dl_fn must return a
        NeuralForecast configured with all 6 DL models (PatchTST, N-BEATS,
        TiDE, DeepAR, DLinear, TimesNet).
    max_steps : int or None
        Override training steps (for smoke tests).

    Returns
    -------
    DataFrame with all per-model, per-seed, per-window results.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    # Generate sliding-window splits
    print(f"\n{'='*60}")
    print(f"[{dataset_name}] Generating {n_windows} sliding-window splits...")
    print(f"{'='*60}")
    splits = _sliding_window_splits(df_full, horizon, input_size, n_windows, max_train_size)

    if not splits:
        print(f"[{dataset_name}] WARNING: No valid splits generated!")
        return pd.DataFrame()

    for w_idx, (df_train, df_test) in enumerate(splits):
        window_id = w_idx + 1
        print(f"\n{'─'*50}")
        print(f"[{dataset_name}] Window {window_id}/{len(splits)}")
        print(f"{'─'*50}")

        # ── Baselines (Seasonal Naive + AutoARIMA) — no seed dependency ──
        # Both models are fitted together in one StatsForecast call (statsforecast
        # batches them internally). The combined elapsed time is divided equally
        # across the two models as an approximation for per-model cost reporting.
        print(f"\n  ▸ Fitting baselines (SeasonalNaive, AutoARIMA)...")
        try:
            sf = get_baseline_fn()
            with Timer() as t_base:
                sf.fit(df_train)
                preds_base = sf.predict(h=horizon)

            preds_base = preds_base.reset_index()
            baseline_cols = [m for m in ["SeasonalNaive", "AutoARIMA"]
                             if m in preds_base.columns]
            n_baselines = len(baseline_cols)  # usually 2
            for model_col in baseline_cols:
                metrics_df = compute_metrics_per_series(
                    df_test, preds_base, df_train, season_length, model_col
                )
                avg_mae = metrics_df["mae"].mean()
                avg_mase = metrics_df["mase"].replace([np.inf], np.nan).mean()
                all_results.append({
                    "dataset": dataset_name,
                    "model": model_col,
                    "seed": "N/A",
                    "window": window_id,
                    "mae_mean": avg_mae,
                    "mase_mean": avg_mase,
                    "train_time_sec": t_base.elapsed / n_baselines,  # approx per model
                    "peak_gpu_mb": t_base.peak_gpu_mb,
                })
                print(f"    {model_col}: MAE={avg_mae:.4f}, MASE={avg_mase:.4f}, "
                      f"Time≈{t_base.elapsed / n_baselines:.1f}s")
        except Exception as e:
            print(f"    ✗ Baselines failed: {e}")
            traceback.print_exc()

        # ── Seed-dependent models (Machine Learning & Deep Learning) ──
        for seed in seeds:
            print(f"\n  ▸ Machine & Deep Learning models (seed={seed})...")
            
            # --- LightGBM ---
            try:
                mlf = get_lgbm_fn(seed=seed)
                with Timer() as t_lgbm:
                    mlf.fit(df_train)
                    preds_lgbm = mlf.predict(h=horizon)

                preds_lgbm = preds_lgbm.reset_index()
                model_col = "LightGBM"
                metrics_df = compute_metrics_per_series(
                    df_test, preds_lgbm, df_train, season_length, model_col
                )
                avg_mae = metrics_df["mae"].mean()
                avg_mase = metrics_df["mase"].replace([np.inf], np.nan).mean()
                all_results.append({
                    "dataset": dataset_name,
                    "model": "LightGBM",
                    "seed": seed,
                    "window": window_id,
                    "mae_mean": avg_mae,
                    "mase_mean": avg_mase,
                    "train_time_sec": t_lgbm.elapsed,
                    "peak_gpu_mb": t_lgbm.peak_gpu_mb,
                })
                print(f"    LightGBM: MAE={avg_mae:.4f}, MASE={avg_mase:.4f}, "
                      f"Time={t_lgbm.elapsed:.1f}s")
            except Exception as e:
                print(f"    ✗ LightGBM failed: {e}")
                traceback.print_exc()

            # Core DL: PatchTST, N-BEATS, TiDE, DeepAR, DLinear, TimesNet
            # All 6 models share one NeuralForecast fit call — data preprocessing
            # (scaling, windowing, DataLoader setup) runs only once.
            try:
                nf = get_dl_fn(seed=seed, max_steps=max_steps)
                with Timer() as t_dl:
                    nf.fit(df=df_train, val_size=horizon)
                    preds_dl = nf.predict()

                preds_dl = preds_dl.reset_index()
                # Drop spurious 'index' column from reset_index()
                if "index" in preds_dl.columns:
                    preds_dl = preds_dl.drop(columns=["index"])
                n_dl_models = sum(1 for c in preds_dl.columns if c not in ("unique_id", "ds"))
                for model_col in preds_dl.columns:
                    if model_col in ("unique_id", "ds"):
                        continue
                    metrics_df = compute_metrics_per_series(
                        df_test, preds_dl, df_train, season_length, model_col
                    )
                    avg_mae = metrics_df["mae"].mean()
                    avg_mase = metrics_df["mase"].replace([np.inf], np.nan).mean()
                    all_results.append({
                        "dataset": dataset_name,
                        "model": model_col,
                        "seed": seed,
                        "window": window_id,
                        "mae_mean": avg_mae,
                        "mase_mean": avg_mase,
                        "train_time_sec": t_dl.elapsed / n_dl_models,  # approx per model
                        "peak_gpu_mb": t_dl.peak_gpu_mb,
                    })
                    print(f"    {model_col}: MAE={avg_mae:.4f}, MASE={avg_mase:.4f}")
                print(f"    [{n_dl_models} DL models total: {t_dl.elapsed:.1f}s]")
            except Exception as e:
                print(f"    ✗ DL models failed (seed={seed}): {e}")
                traceback.print_exc()

    # ── Save results ──
    results_df = pd.DataFrame(all_results)
    out_path = results_dir / f"{dataset_name.lower()}_raw_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n[{dataset_name}] Results saved to {out_path}")
    print(f"[{dataset_name}] Total rows: {len(results_df)}")

    return results_df
