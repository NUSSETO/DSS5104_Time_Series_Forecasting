"""
M5 dataset preparation.

Loads Walmart sales data (sales_train_evaluation.csv + calendar.csv),
converts to Nixtla long format. Samples item-level series for tractability.
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import M5_CONFIG


def load_m5(
    n_series: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load M5 data and return (df_train, df_test) in long format.

    The last `horizon` days are held out as the test set.

    Parameters
    ----------
    n_series : int or None
        Number of item-level series to sample. None = use all (~30k).
    random_state : int
        Seed for reproducible sampling.

    Returns
    -------
    df_train : pd.DataFrame  — columns ['unique_id', 'ds', 'y']
    df_test  : pd.DataFrame  — columns ['unique_id', 'ds', 'y']
    """
    cfg = M5_CONFIG
    horizon = cfg["horizon"]

    # --- Load calendar for date mapping ---
    print(f"[M5] Reading calendar: {cfg['calendar_csv']}")
    calendar = pd.read_csv(cfg["calendar_csv"])
    # Create d_col -> date mapping
    day_to_date = dict(zip(calendar["d"], pd.to_datetime(calendar["date"])))

    # --- Load sales data ---
    print(f"[M5] Reading sales CSV: {cfg['sales_csv']}")
    sales = pd.read_csv(cfg["sales_csv"])

    # Build unique_id from id column (item_id + store_id already encoded)
    id_col = "id"
    d_cols = [c for c in sales.columns if c.startswith("d_")]

    # --- Sample series ---
    all_ids = sales[id_col].values
    if n_series is not None and n_series < len(all_ids):
        rng = np.random.RandomState(random_state)
        selected_ids = rng.choice(all_ids, size=n_series, replace=False)
        sales = sales[sales[id_col].isin(selected_ids)].reset_index(drop=True)
        print(f"[M5] Sampled {n_series} series out of {len(all_ids)}")
    else:
        print(f"[M5] Using all {len(all_ids)} series")

    # --- Melt to long format ---
    print("[M5] Melting to long format...")
    df_long = sales.melt(
        id_vars=[id_col],
        value_vars=d_cols,
        var_name="d",
        value_name="y",
    )
    df_long.rename(columns={id_col: "unique_id"}, inplace=True)
    
    # Explicit missing value handling for protocol documentation
    df_long["y"] = df_long["y"].fillna(0.0)

    # Map d_col to date
    df_long["ds"] = df_long["d"].map(day_to_date)
    df_long.drop(columns=["d"], inplace=True)
    df_long.dropna(subset=["ds"], inplace=True)
    df_long["y"] = df_long["y"].astype(float)

    # Sort
    df_long.sort_values(["unique_id", "ds"], inplace=True)
    df_long.reset_index(drop=True, inplace=True)

    # --- Split: last `horizon` days per series as test ---
    # Compute per-series max date via groupby, then merge — avoids a
    # Python lambda in transform() which iterates group-by-group.
    print(f"[M5] Splitting: last {horizon} days as test...")
    max_dates = df_long.groupby("unique_id")["ds"].max().reset_index()
    max_dates["cutoff"] = max_dates["ds"] - pd.Timedelta(days=horizon - 1)
    df_long = df_long.merge(max_dates[["unique_id", "cutoff"]], on="unique_id")
    df_train = df_long[df_long["ds"] < df_long["cutoff"]][["unique_id", "ds", "y"]].reset_index(drop=True)
    df_test = df_long[df_long["ds"] >= df_long["cutoff"]][["unique_id", "ds", "y"]].reset_index(drop=True)

    print(f"[M5] Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    print(f"[M5] Unique series: {df_train['unique_id'].nunique()}")

    return df_train, df_test


if __name__ == "__main__":
    df_train, df_test = load_m5(n_series=M5_CONFIG["n_series_sample"])
    print("\n--- Train sample ---")
    print(df_train.head(10))
    print(f"\n--- Test sample ---")
    print(df_test.head(10))
    print(f"\nTrain date range: {df_train['ds'].min()} — {df_train['ds'].max()}")
    print(f"Test date range:  {df_test['ds'].min()} — {df_test['ds'].max()}")
