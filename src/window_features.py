"""
Window-level feature aggregation (non-overlapping windows by default).

Based on `build_window_features.ipynb`:
- Load exported_data/features_frame_level.csv
- Sort by (subject_id, video_id, frame_idx)
- frame_order = cumcount per (subject_id, video_id)
- window_id = frame_order // window_size
- Aggregate stats + blink/yawn ratios
- Label inferred from video_id prefix: "10" -> drowsy, "0" -> alert
"""

import os
import numpy as np
import pandas as pd


def agg_window(g: pd.DataFrame, blink_ear_thresh: float, yawn_mar_thresh: float) -> pd.Series:
    g = g.sort_values("frame_idx")
    ear = g["ear_mean"].values
    mar = g["mar"].values

    ear_mean = float(ear.mean()) if len(ear) else np.nan
    ear_std  = float(ear.std()) if len(ear) else np.nan
    mar_mean = float(mar.mean()) if len(mar) else np.nan
    mar_std  = float(mar.std()) if len(mar) else np.nan

    blink_ratio = float((ear < blink_ear_thresh).mean()) if len(ear) else np.nan
    yawn_ratio  = float((mar > yawn_mar_thresh).mean()) if len(mar) else np.nan

    ear_diff_mean = float(np.abs(np.diff(ear)).mean()) if len(ear) > 1 else 0.0

    vid = str(g["video_id"].iloc[0])
    if vid.startswith("10"):
        label = "drowsy"
    elif vid.startswith("0"):
        label = "alert"
    else:
        label = None

    return pd.Series({
        "ear_mean_mean": ear_mean,
        "ear_mean_std": ear_std,
        "mar_mean": mar_mean,
        "mar_std": mar_std,
        "blink_ratio": blink_ratio,
        "yawn_ratio": yawn_ratio,
        "ear_diff_mean": ear_diff_mean,
        "num_frames": int(len(g)),
        "label": label,
    })


def run_build_window_features(
    frame_csv: str = "exported_data/features_frame_level.csv",
    out_csv: str = "exported_data/features_window_level.csv",
    window_size: int = 60,
    blink_ear_thresh: float = 0.21,
    yawn_mar_thresh: float = 0.60,
):
    df = pd.read_csv(frame_csv)

    expected_cols = {"subject_id", "video_id", "frame_idx", "ear_mean", "mar"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in frame CSV: {sorted(missing)}")

    df = df.sort_values(["subject_id", "video_id", "frame_idx"]).reset_index(drop=True)
    df["frame_order"] = df.groupby(["subject_id", "video_id"]).cumcount()
    df["window_id"] = (df["frame_order"] // window_size).astype(int)

    group_cols = ["subject_id", "video_id", "window_id"]
    window_df = (
        df.groupby(group_cols, as_index=False)
          .apply(lambda g: agg_window(g, blink_ear_thresh, yawn_mar_thresh))
          .reset_index()
    )
    # drop helper columns produced by groupby/apply
    for col in ["level_0", "index"]:
        if col in window_df.columns:
            window_df = window_df.drop(columns=[col])

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    window_df.to_csv(out_csv, index=False)
    print("Saved window-level features to:", out_csv, "(n=", len(window_df), ")")
    return window_df
