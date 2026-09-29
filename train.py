"""Train the models and save the artifacts used by the Streamlit app.

Usage:
    python train.py            # full grid search (about 1-3 minutes)
    python train.py --quick    # small grids, for a fast test run
"""
import argparse
import sys

from src.data import DatasetError
from src.training import run_training


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the diabetes risk model.")
    parser.add_argument("--quick", action="store_true", help="use small hyper-parameter grids")
    parser.add_argument("--no-plots", action="store_true", help="skip saving report images")
    args = parser.parse_args()
    try:
        run_training(quick=args.quick, make_plots=not args.no_plots)
    except DatasetError as exc:
        print(f"\nDATASET ERROR:\n{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # the guard matters on Windows (multiprocessing)
    sys.exit(main())
