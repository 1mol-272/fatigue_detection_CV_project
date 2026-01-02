"""
Run the full pipeline end-to-end (Scheme C).

Usage:
  python scripts/run_all.py \
    --video_root videos \
    --landmarks_root data \
    --export_dir exported_data \
    --processed_dir processed_data \
    --process_every 1 \
    --window_size 60 \
    --blink_ear_thresh 0.21 \
    --yawn_mar_thresh 0.60 \
    --min_frames 15
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from landmarks import run_extract_landmarks
from frame_features import run_extract_frame_features
from window_features import run_build_window_features
from dataset_prep import run_prepare_dataset


def build_parser():
    p = argparse.ArgumentParser(description="Run the full pipeline.")
    p.add_argument("--video_root", default="videos")
    p.add_argument("--landmarks_root", default="data")
    p.add_argument("--export_dir", default="exported_data")
    p.add_argument("--processed_dir", default="processed_data")
    p.add_argument("--process_every", type=int, default=1)
    p.add_argument("--window_size", type=int, default=60)
    p.add_argument("--blink_ear_thresh", type=float, default=0.21)
    p.add_argument("--yawn_mar_thresh", type=float, default=0.60)
    p.add_argument("--min_frames", type=int, default=15)
    p.add_argument("--random_state", type=int, default=42)
    return p


def main():
    args = build_parser().parse_args()

    frame_csv = os.path.join(args.export_dir, "features_frame_level.csv")
    window_csv = os.path.join(args.export_dir, "features_window_level.csv")

    run_extract_landmarks(
        input_root=args.video_root,
        output_root=args.landmarks_root,
        process_every_n_frames=args.process_every,
        skip_existing=True,
        label_rules=None,
    )

    run_extract_frame_features(
        landmarks_root=args.landmarks_root,
        out_csv=frame_csv,
    )

    run_build_window_features(
        frame_csv=frame_csv,
        out_csv=window_csv,
        window_size=args.window_size,
        blink_ear_thresh=args.blink_ear_thresh,
        yawn_mar_thresh=args.yawn_mar_thresh,
    )

    run_prepare_dataset(
        input_csv=window_csv,
        output_dir=args.processed_dir,
        min_frames=args.min_frames,
        random_state=args.random_state,
    )

    print("All done.")


if __name__ == "__main__":
    main()
