"""
Orchestrator: runs all three dataset pipelines sequentially.

Usage:
    python runners/run_all.py                  # Full run
    python runners/run_all.py --smoke-test     # Quick validation of all 3
"""

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runners.run_m4 import main as run_m4
from runners.run_m5 import main as run_m5
from runners.run_traffic import main as run_traffic


def main(smoke_test: bool = False):
    mode = "SMOKE TEST" if smoke_test else "FULL RUN"
    print(f"\n{'#'*60}")
    print(f"# DSS5104 CA2 — Run All Datasets ({mode})")
    print(f"{'#'*60}")

    total_start = time.time()

    # ── M4 ──
    print(f"\n{'='*60}")
    print("STARTING: M4 Monthly")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        run_m4(smoke_test=smoke_test)
        print(f"M4 completed in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"M4 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # ── M5 ──
    print(f"\n{'='*60}")
    print("STARTING: M5")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        run_m5(smoke_test=smoke_test)
        print(f"M5 completed in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"M5 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # ── Traffic ──
    print(f"\n{'='*60}")
    print("STARTING: Traffic")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        run_traffic(smoke_test=smoke_test)
        print(f"Traffic completed in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"Traffic FAILED: {e}")
        import traceback
        traceback.print_exc()

    total_elapsed = time.time() - total_start
    print(f"\n{'#'*60}")
    print(f"# ALL DONE — Total time: {total_elapsed:.1f}s ({total_elapsed/3600:.2f}h)")
    print(f"{'#'*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run all datasets with minimal settings")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
