"""
End-to-end runner for M5 dataset.

Usage:
    python runners/run_m5.py                  # Full run
    python runners/run_m5.py --smoke-test     # Quick validation
"""

import argparse
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import M5_CONFIG, SEEDS, RESULTS_DIR
from data_prep.m5_prep import load_m5
from models.baselines import get_baseline_forecaster
from models.lgbm_model import get_lgbm_forecaster
from models.dl_models import get_dl_models
from evaluation.walk_forward import run_walk_forward


def main(smoke_test: bool = False):
    cfg = M5_CONFIG
    seeds = SEEDS if not smoke_test else [42]
    n_series = 30 if smoke_test else cfg["n_series_sample"]
    max_steps = 10 if smoke_test else None
    n_windows = 1 if smoke_test else cfg["walk_forward_windows"]

    print(f"{'='*60}")
    print(f"M5 — {'SMOKE TEST' if smoke_test else 'FULL RUN'}")
    print(f"  Series: {n_series}, Seeds: {len(seeds)}, Windows: {n_windows}")
    print(f"{'='*60}")

    # 1. Load data
    df_train, df_test = load_m5(n_series=n_series)
    df_full = pd.concat([df_train, df_test], ignore_index=True)

    # 2. Define model factory functions
    def get_baselines():
        return get_baseline_forecaster(
            season_length=cfg["season_length"], freq=cfg["freq"]
        )

    def get_lgbm(seed):
        return get_lgbm_forecaster(
            freq=cfg["freq"], season_length=cfg["season_length"], seed=seed
        )

    def get_dl(seed, max_steps=max_steps):
        return get_dl_models(
            horizon=cfg["horizon"], input_size=cfg["input_size"],
            freq=cfg["freq"], seed=seed, max_steps=max_steps,
        )

    # 3. Run walk-forward evaluation
    results_dir = RESULTS_DIR / "m5"
    results = run_walk_forward(
        df_full=df_full,
        dataset_name="M5",
        horizon=cfg["horizon"],
        input_size=cfg["input_size"],
        freq=cfg["freq"],
        season_length=cfg["season_length"],
        n_windows=n_windows,
        seeds=seeds,
        results_dir=results_dir,
        get_baseline_fn=get_baselines,
        get_lgbm_fn=get_lgbm,
        get_dl_fn=get_dl,
        max_steps=max_steps,
        max_train_size=cfg.get("max_train_size")
    )

    print(f"\n✓ M5 complete. Results: {results_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run with minimal settings for quick validation")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
