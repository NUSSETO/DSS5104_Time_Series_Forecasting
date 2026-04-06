"""
Traffic dataset preparation.

Loads the Traffic.tsf file (San Francisco road occupancy rates),
parses it into Nixtla long format, and samples a subset of sensors.
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import TRAFFIC_CONFIG


def _parse_tsf(file_path: str) -> pd.DataFrame:
    """Parse a .tsf (Time Series Format) file into a long-format DataFrame.

    The TSF format has a header section with @-prefixed metadata lines,
    followed by data lines in the format:
        series_name:start_timestamp:value1,value2,...

    Three optimisations over the original:
    1. freq_map and pd_freq are resolved once from the @frequency header,
       not re-created inside the per-series loop.
    2. Dead code (an overwritten timestamp_str variable) is removed.
    3. One DataFrame is built per series and concatenated at the end,
       replacing one dict-append per data point.
    """
    # ── Hoist freq_map outside the loop — it never changes ────────────────
    freq_map = {
        "hourly": "h",
        "daily": "D",
        "weekly": "W",
        "monthly": "ME",
        "yearly": "YE",
        "minutely": "min",
    }
    pd_freq = "h"  # default; overwritten when @frequency header is parsed

    series_dfs = []
    in_data = False

    print(f"[Traffic] Parsing TSF file: {file_path}")
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse header
            if line.startswith("@"):
                lower = line.lower()
                if lower.startswith("@frequency"):
                    frequency = line.split()[-1].strip()
                    pd_freq = freq_map.get(frequency.lower(), "h")
                    print(f"[Traffic] Frequency: {frequency}")
                if lower == "@data":
                    in_data = True
                    print(f"[Traffic] Data section starts at line {line_num}")
                continue

            if not in_data:
                continue

            # Parse data lines: name:timestamp:val1,val2,...
            # maxsplit=2 handles colons that may appear inside value fields
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue

            series_name = parts[0].strip()
            value_str = parts[2].strip()

            # Parse values with np.fromstring — faster than list comprehension
            values = np.fromstring(value_str, sep=",")
            n = len(values)
            if n == 0:
                continue

            # Parse start timestamp: "2015-01-01 00-00-01" → "2015-01-01 00:00:01"
            ts_raw = parts[1].strip()
            date_parts = ts_raw.split(" ")
            if len(date_parts) == 2:
                ts_clean = f"{date_parts[0]} {date_parts[1].replace('-', ':')}"
            else:
                ts_clean = ts_raw
            try:
                start = pd.to_datetime(ts_clean)
            except Exception:
                start = pd.Timestamp("2015-01-01")

            dates = pd.date_range(start=start, periods=n, freq=pd_freq)

            # Build one DataFrame per series instead of one dict per data point
            series_dfs.append(pd.DataFrame({
                "unique_id": series_name,
                "ds": dates,
                "y": values,
            }))

    df = pd.concat(series_dfs, ignore_index=True)
    df["ds"] = pd.to_datetime(df["ds"])
    # Explicit missing value handling for protocol documentation
    df["y"] = df["y"].fillna(0.0)
    print(f"[Traffic] Parsed {df['unique_id'].nunique()} series, {len(df)} total rows")
    return df


def load_traffic(
    n_series: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Traffic data and return (df_train, df_test) in long format.

    The last `horizon` time steps per series are held out as the test set.

    Parameters
    ----------
    n_series : int or None
        Number of sensor series to sample. None = use all.
    random_state : int
        Seed for reproducible sampling.

    Returns
    -------
    df_train : pd.DataFrame  — columns ['unique_id', 'ds', 'y']
    df_test  : pd.DataFrame  — columns ['unique_id', 'ds', 'y']
    """
    cfg = TRAFFIC_CONFIG
    horizon = cfg["horizon"]

    # --- Parse TSF file ---
    df = _parse_tsf(str(cfg["data_file"]))

    # --- Sample series ---
    all_ids = df["unique_id"].unique()
    if n_series is not None and n_series < len(all_ids):
        rng = np.random.RandomState(random_state)
        selected_ids = rng.choice(all_ids, size=n_series, replace=False)
        df = df[df["unique_id"].isin(selected_ids)].reset_index(drop=True)
        print(f"[Traffic] Sampled {n_series} sensors out of {len(all_ids)}")
    else:
        print(f"[Traffic] Using all {len(all_ids)} sensors")

    # Sort
    df.sort_values(["unique_id", "ds"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Split: last `horizon` time steps per series as test ---
    # Compute per-series max date via groupby+merge — avoids a Python
    # loop over groups. Data is already sorted by (unique_id, ds) and
    # has regular 1-hour spacing, so time-based and positional splits
    # are equivalent.
    print(f"[Traffic] Splitting: last {horizon} time steps as test...")
    max_dates = df.groupby("unique_id")["ds"].max().reset_index()
    max_dates["cutoff"] = max_dates["ds"] - pd.Timedelta(hours=horizon - 1)
    df = df.merge(max_dates[["unique_id", "cutoff"]], on="unique_id")
    df_train = df[df["ds"] < df["cutoff"]][["unique_id", "ds", "y"]].reset_index(drop=True)
    df_test = df[df["ds"] >= df["cutoff"]][["unique_id", "ds", "y"]].reset_index(drop=True)

    print(f"[Traffic] Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    print(f"[Traffic] Unique series: {df_train['unique_id'].nunique()}")

    return df_train, df_test


if __name__ == "__main__":
    df_train, df_test = load_traffic(n_series=TRAFFIC_CONFIG["n_series_sample"])
    print("\n--- Train sample ---")
    print(df_train.head(10))
    print(f"\n--- Test sample ---")
    print(df_test.head(10))
    print(f"\nTrain date range: {df_train['ds'].min()} — {df_train['ds'].max()}")
    print(f"Test date range:  {df_test['ds'].min()} — {df_test['ds'].max()}")
