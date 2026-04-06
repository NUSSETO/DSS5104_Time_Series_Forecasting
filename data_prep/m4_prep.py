"""
M4 Monthly dataset preparation.

Loads the M4 Monthly train/test CSVs (wide format), melts to Nixtla long
format (unique_id, ds, y), samples a subset of series for tractability,
and assigns synthetic monthly timestamps.
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import M4_CONFIG


def _wide_to_long(df_wide: pd.DataFrame, freq: str, end_date: str = "2020-01-01") -> pd.DataFrame:
    """Convert M4 wide-format CSV to Nixtla long format.

    M4 wide format: first column is Series ID, remaining columns are time
    steps with values (NaN-padded at the end for shorter series).

    Uses pd.melt() to pivot in one vectorised call, drops NaN-padding rows,
    then assigns dates per series via a single groupby (one date_range per
    series instead of one dict per data point).
    """
    id_col = df_wide.columns[0]
    value_cols = df_wide.columns[1:]
    end_ts = pd.Timestamp(end_date)

    # Preserve original column order: column names like "V1","V2",... sort
    # lexicographically wrong ("V10" < "V2"), so map each name to its index.
    col_order = {col: i for i, col in enumerate(value_cols)}

    # ── Melt wide → long in one call ──────────────────────────────────────
    df_long = df_wide.melt(
        id_vars=[id_col],
        value_vars=value_cols,
        var_name="_step",
        value_name="y",
    )
    df_long.rename(columns={id_col: "unique_id"}, inplace=True)

    # Drop NaN-padded trailing entries (shorter series)
    df_long = df_long.dropna(subset=["y"])
    df_long["y"] = df_long["y"].astype(float).fillna(0.0)

    # Sort by original column order to restore temporal sequence
    df_long["_col_idx"] = df_long["_step"].map(col_order)
    df_long.sort_values(["unique_id", "_col_idx"], inplace=True)
    df_long.drop(columns=["_step", "_col_idx"], inplace=True)
    df_long.reset_index(drop=True, inplace=True)

    # ── Assign dates: each series ends at end_date ─────────────────────────
    # One date_range call per series (replaces one dict-append per value).
    def _assign_dates(group):
        group = group.copy()
        group["ds"] = pd.date_range(end=end_ts, periods=len(group), freq=freq)
        return group

    df_long = df_long.groupby("unique_id", group_keys=False).apply(_assign_dates)
    df_long["ds"] = pd.to_datetime(df_long["ds"])
    return df_long[["unique_id", "ds", "y"]]


def load_m4_monthly(
    n_series: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load M4 Monthly data and return (df_train, df_test) in long format.

    Parameters
    ----------
    n_series : int or None
        Number of series to sample. None = use all.
    random_state : int
        Seed for reproducible sampling.

    Returns
    -------
    df_train : pd.DataFrame  — columns ['unique_id', 'ds', 'y']
    df_test  : pd.DataFrame  — columns ['unique_id', 'ds', 'y']
    """
    cfg = M4_CONFIG

    # --- Read wide-format CSVs ---
    print(f"[M4] Reading train CSV: {cfg['train_csv']}")
    train_wide = pd.read_csv(cfg["train_csv"])
    print(f"[M4] Reading test CSV: {cfg['test_csv']}")
    test_wide = pd.read_csv(cfg["test_csv"])

    # --- Determine which series to use ---
    all_ids = train_wide.iloc[:, 0].values
    if n_series is not None and n_series < len(all_ids):
        rng = np.random.RandomState(random_state)
        selected_ids = rng.choice(all_ids, size=n_series, replace=False)
        train_wide = train_wide[train_wide.iloc[:, 0].isin(selected_ids)].reset_index(drop=True)
        test_wide = test_wide[test_wide.iloc[:, 0].isin(selected_ids)].reset_index(drop=True)
        print(f"[M4] Sampled {n_series} series out of {len(all_ids)}")
    else:
        print(f"[M4] Using all {len(all_ids)} series")

    # --- Convert train to long format ---
    print("[M4] Converting train to long format...")
    df_train = _wide_to_long(train_wide, freq=cfg["freq"])

    # --- Convert test to long format ---
    # Test timestamps continue from each series' last train date + 1 month.
    # Uses melt + groupby (one date_range per series) instead of iterrows().
    print("[M4] Converting test to long format...")
    id_col = test_wide.columns[0]
    value_cols = test_wide.columns[1:]
    last_dates = df_train.groupby("unique_id")["ds"].max()
    col_order = {col: i for i, col in enumerate(value_cols)}

    df_test = test_wide.melt(
        id_vars=[id_col],
        value_vars=value_cols,
        var_name="_step",
        value_name="y",
    )
    df_test.rename(columns={id_col: "unique_id"}, inplace=True)
    df_test = df_test.dropna(subset=["y"])
    df_test["y"] = df_test["y"].astype(float).fillna(0.0)
    df_test["_col_idx"] = df_test["_step"].map(col_order)
    df_test.sort_values(["unique_id", "_col_idx"], inplace=True)
    df_test.drop(columns=["_step", "_col_idx"], inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    def _assign_test_dates(group):
        uid = group["unique_id"].iloc[0]
        start = last_dates[uid] + pd.DateOffset(months=1)
        group = group.copy()
        group["ds"] = pd.date_range(start=start, periods=len(group), freq=cfg["freq"])
        return group

    df_test = df_test.groupby("unique_id", group_keys=False).apply(_assign_test_dates)
    df_test["ds"] = pd.to_datetime(df_test["ds"])
    df_test = df_test[["unique_id", "ds", "y"]]

    print(f"[M4] Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    print(f"[M4] Unique series: {df_train['unique_id'].nunique()}")

    return df_train, df_test


if __name__ == "__main__":
    df_train, df_test = load_m4_monthly(n_series=M4_CONFIG["n_series_sample"])
    print("\n--- Train sample ---")
    print(df_train.head(10))
    print(f"\n--- Test sample ---")
    print(df_test.head(10))
    print(f"\nTrain date range: {df_train['ds'].min()} — {df_train['ds'].max()}")
    print(f"Test date range:  {df_test['ds'].min()} — {df_test['ds'].max()}")
