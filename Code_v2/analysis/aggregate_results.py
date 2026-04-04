"""
Aggregate raw results from all model pipelines into summary tables.

Reads all per-model CSVs from results/ (e.g. PatchTST_M4.csv),
computes mean ± std across seeds and windows, and outputs:
  - summary_table.csv          (main comparison table)
  - computational_costs.csv    (training time & hardware)
  - pivot_mae.csv              (Model × Dataset pivot)
  - pivot_mase.csv             (Model × Dataset pivot)

Run after one or more model pipelines have completed:
    python analysis/aggregate_results.py
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR

# Output files produced by this script (excluded from CSV discovery)
_SUMMARY_FILES = {"summary_table", "computational_costs", "pivot_mae", "pivot_mase"}


def aggregate():
    """Load all per-model CSVs and produce summary tables."""
    results_dir = RESULTS_DIR
    all_dfs = []

    # Discover all per-model result CSVs, excluding summary files
    csv_files = sorted(results_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if f.stem not in _SUMMARY_FILES]

    if not csv_files:
        print(f"No result CSVs found in {results_dir}")
        print("Run one or more model pipelines first, e.g.:")
        print("    python pipelines/run_seasonal_naive.py")
        return

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        all_dfs.append(df)
        print(f"Loaded {csv_path.name} ({len(df)} rows)")

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows: {len(df_all)}")

    # ── Summary table: mean ± std of MAE and MASE per (dataset, model) ──
    summary = (
        df_all
        .groupby(["dataset", "model"])
        .agg(
            mae_mean=("mae_mean", "mean"),
            mae_std=("mae_mean", "std"),
            mase_mean=("mase_mean", "mean"),
            mase_std=("mase_mean", "std"),
            n_runs=("mae_mean", "count"),
        )
        .reset_index()
    )
    # Fill NaN std (single-run models like baselines)
    summary["mae_std"] = summary["mae_std"].fillna(0)
    summary["mase_std"] = summary["mase_std"].fillna(0)

    # Format display columns
    summary["MAE"] = summary.apply(
        lambda r: f"{r['mae_mean']:.4f} ± {r['mae_std']:.4f}", axis=1
    )
    summary["MASE"] = summary.apply(
        lambda r: f"{r['mase_mean']:.4f} ± {r['mase_std']:.4f}", axis=1
    )

    summary_path = results_dir / "summary_table.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary table saved to {summary_path}")
    print(summary[["dataset", "model", "MAE", "MASE", "n_runs"]].to_string(index=False))

    # ── Computational costs: mean training time per model per dataset ──
    costs = (
        df_all
        .groupby(["dataset", "model"])
        .agg(
            avg_train_time_sec=("train_time_sec", "mean"),
            total_train_time_sec=("train_time_sec", "sum"),
            peak_gpu_mb=("peak_gpu_mb", "max"),
        )
        .reset_index()
    )

    costs_path = results_dir / "computational_costs.csv"
    costs.to_csv(costs_path, index=False)
    print(f"\nComputational costs saved to {costs_path}")
    print(costs.to_string(index=False))

    # ── Pivot tables for report (Model × Dataset) ──
    pivot_mae = summary.pivot(index="model", columns="dataset", values="MAE")
    pivot_mase = summary.pivot(index="model", columns="dataset", values="MASE")

    pivot_path = results_dir / "pivot_mae.csv"
    pivot_mae.to_csv(pivot_path)
    print(f"\nPivot MAE table saved to {pivot_path}")

    pivot_path = results_dir / "pivot_mase.csv"
    pivot_mase.to_csv(pivot_path)
    print(f"Pivot MASE table saved to {pivot_path}")

    return summary, costs


if __name__ == "__main__":
    aggregate()
