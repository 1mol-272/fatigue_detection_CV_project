"""
scripts/run_models.py

One python entrypoint for all model-related things:

- train: train & compare multiple models on ONE processed_dir
- sweep: run parameter sweep (window_size / thresholds / min_frames), rebuild data artifacts, and train models per setting

Data vs results policy (strict):
- exported_data/ : data artifacts ONLY (e.g., frame/window CSVs)
- processed_data/: data artifacts ONLY (e.g., parquet splits)
- results/       : ALL model/training related outputs (metrics, models, reports, confusion matrices, summaries)

This script enforces that by:
- train -> default results_dir = <repo>/results/train
- sweep -> default results_root = <repo>/results/experiments (and passed into experiments_lib.run_sweep)
"""

from __future__ import annotations

import os
import argparse
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from train_models_lib import train_and_compare
from experiments_lib import run_sweep


def _parse_csv_list(s: str, cast):
    return [cast(x.strip()) for x in (s or "").split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Model-related runner (train or sweep).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # =========================
    # Train once
    # =========================
    t = sub.add_parser("train", help="Train & compare models on an existing processed dataset.")
    t.add_argument("--processed_dir", default="processed_data", help="Folder with train/test(/val).parquet")
    t.add_argument(
        "--results_dir",
        default=os.path.join(REPO_ROOT, "results", "train"),
        help="ALL results go here. Default: results/train",
    )
    t.add_argument("--random_state", type=int, default=42)
    t.add_argument(
        "--choose_by",
        default="f1_macro",
        choices=["f1_macro", "accuracy", "precision_macro", "recall_macro"],
        help="Metric used to select best model on val (if exists) else test.",
    )

    # =========================
    # Sweep experiments
    # =========================
    s = sub.add_parser("sweep", help="Sweep window/threshold/min_frames; rebuild data artifacts; train models per setting.")
    s.add_argument("--frame_csv", default="exported_data/features_frame_level.csv")

    # Data artifacts (keep your team workflow)
    s.add_argument("--export_root", default="exported_data/experiments", help="DATA artifacts: window CSVs per exp")
    s.add_argument("--processed_root", default="processed_data/experiments", help="DATA artifacts: parquet splits per exp")

    # Results artifacts (all training/model outputs must go here)
    s.add_argument(
        "--results_root",
        default=os.path.join(REPO_ROOT, "results", "experiments"),
        help="RESULTS root for all experiments. Default: results/experiments",
    )
    s.add_argument(
        "--summary_csv",
        default=None,
        help="Optional path for sweep summary CSV. If not set, saved under results_root/summary_<timestamp>.csv",
    )

    # Sweep grids
    s.add_argument("--window_sizes", default="30,60,90")
    s.add_argument("--blink_ear_threshs", default="0.20,0.21,0.22")
    s.add_argument("--yawn_mar_threshs", default="0.55,0.60,0.65")
    s.add_argument("--min_frames", default="15")

    s.add_argument("--random_state", type=int, default=42)
    s.add_argument(
        "--choose_by",
        default="f1_macro",
        choices=["f1_macro", "accuracy", "precision_macro", "recall_macro"],
        help="Metric used to rank experiments (based on best-on-test).",
    )

    return p


def main():
    args = build_parser().parse_args()

    if args.cmd == "train":
        os.makedirs(args.results_dir, exist_ok=True)

        metrics_df, best_name, best_test_metrics = train_and_compare(
            processed_dir=args.processed_dir,
            results_dir=args.results_dir,
            random_state=args.random_state,
            choose_by=args.choose_by,
        )

        print("\nSelection metrics (top 10):")
        print(metrics_df.head(10))
        print("\nBest model:", best_name)
        print("Best-on-test:", best_test_metrics)
        print("\nSaved training results to:", args.results_dir)
        return

    if args.cmd == "sweep":
        window_sizes = _parse_csv_list(args.window_sizes, int)
        blink = _parse_csv_list(args.blink_ear_threshs, float)
        yawn = _parse_csv_list(args.yawn_mar_threshs, float)
        min_frames_list = _parse_csv_list(args.min_frames, int)

        os.makedirs(args.results_root, exist_ok=True)

        summary = run_sweep(
            frame_csv=args.frame_csv,

            # DATA artifacts (unchanged workflow)
            export_root=args.export_root,
            processed_root=args.processed_root,

            # RESULTS artifacts (forced to results/)
            results_root=args.results_root,

            window_sizes=window_sizes,
            blink_ear_threshs=blink,
            yawn_mar_threshs=yawn,
            min_frames_list=min_frames_list,
            random_state=args.random_state,
            choose_by=args.choose_by,
            summary_csv=args.summary_csv,
        )

        print("\nSweep summary (top 10):")
        print(summary.head(10))
        print("\nSaved experiment results under:", args.results_root)
        return

    raise RuntimeError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
