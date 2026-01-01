"""
Final dataset preparation.

Based on `final_dataset_preparation.ipynb`:
- Read window-level CSV
- Drop duplicates
- Filter by min_frames and sanity checks
- Map labels -> label_id
- Split by subject_id into train/val/test
- Save parquet outputs + label_map.json
"""

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_MAP = {"alert": 0, "drowsy": 1}

REQUIRED_COLUMNS = [
    "subject_id", "video_id", "window_id",
    "ear_mean_mean", "ear_mean_std",
    "mar_mean", "mar_std",
    "blink_ratio", "yawn_ratio",
    "ear_diff_mean", "num_frames", "label"
]


def run_prepare_dataset(
    input_csv: str = "exported_data/features_window_level.csv",
    output_dir: str = "processed_data",
    min_frames: int = 15,
    random_state: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in window CSV: {sorted(missing)}")

    df = df.drop_duplicates(subset=["subject_id", "video_id", "window_id"])
    df = df[df["num_frames"] >= min_frames]

    # sanity checks (same idea as notebook)
    df = df[df["ear_mean_mean"] > 0]
    df = df[df["ear_mean_std"] >= 0]
    df = df[df["mar_mean"] > 0]
    df = df[(df["blink_ratio"] >= 0) & (df["blink_ratio"] <= 1)]
    df = df[(df["yawn_ratio"] >= 0) & (df["yawn_ratio"] <= 1)]

    df["label_id"] = df["label"].map(LABEL_MAP)

    with open(os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, indent=2)

    subjects = df["subject_id"].unique()
    if len(subjects) < 3:
        raise ValueError(f"Need >=3 subjects for split, got {len(subjects)}")

    train_subj, temp_subj = train_test_split(subjects, test_size=0.2, random_state=random_state)
    val_subj, test_subj = train_test_split(temp_subj, test_size=0.5, random_state=random_state)

    train_df = df[df["subject_id"].isin(train_subj)]
    val_df = df[df["subject_id"].isin(val_subj)]
    test_df = df[df["subject_id"].isin(test_subj)]

    train_df.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(output_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)
    df.to_parquet(os.path.join(output_dir, "final_dataset_windows.parquet"), index=False)

    print("Saved processed datasets to:", output_dir)
    print("Train/Val/Test:", train_df.shape, val_df.shape, test_df.shape)
    return train_df, val_df, test_df
