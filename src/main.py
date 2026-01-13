from __future__ import annotations

import argparse

from preprocessing import run_preprocessing
from classification import run_classification
from merge_and_clustering import run_merge_and_clustering


def main():
    parser = argparse.ArgumentParser(description="Run full Big Data Analytics pipeline")
    parser.add_argument(
        "--step",
        choices=["all", "preprocess", "classify", "cluster"],
        default="all",
        help="Which step to run"
    )
    args = parser.parse_args()

    if args.step in ["all", "preprocess"]:
        run_preprocessing()

    if args.step in ["all", "classify"]:
        run_classification()

    if args.step in ["all", "cluster"]:
        run_merge_and_clustering()


if __name__ == "__main__":
    main()
