"""
Frame-level feature extraction (EAR/MAR).

Based on `extract_frame_features.ipynb`:
- Read each *_landmarks.npz + *_meta.json
- Compute EAR (left/right/mean) and MAR per frame
- Save concatenated results to exported_data/features_frame_level.csv
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 385, 387, 362, 380, 373]
MOUTH_IDX     = [13, 14, 78, 308, 82, 312]


def dist2d(a, b):
    return np.linalg.norm(a[:2] - b[:2])


def eye_aspect_ratio(pts, idx):
    p1, p2, p3, p4, p5, p6 = [pts[i] for i in idx]
    vertical = dist2d(p2, p6) + dist2d(p3, p5)
    horizontal = 2.0 * dist2d(p1, p4) + 1e-6
    return vertical / horizontal


def mouth_aspect_ratio(pts, idx):
    p_up1, p_down1, p_left, p_right, p_up2, p_down2 = [pts[i] for i in idx]
    vertical = dist2d(p_up1, p_down1) + dist2d(p_up2, p_down2)
    horizontal = 2.0 * dist2d(p_left, p_right) + 1e-6
    return vertical / horizontal


def compute_frame_features(pts):
    if np.isnan(pts).all():
        return None

    ear_left = eye_aspect_ratio(pts, LEFT_EYE_IDX)
    ear_right = eye_aspect_ratio(pts, RIGHT_EYE_IDX)
    mar = mouth_aspect_ratio(pts, MOUTH_IDX)

    return {
        "ear_left": float(ear_left),
        "ear_right": float(ear_right),
        "ear_mean": float((ear_left + ear_right) / 2.0),
        "mar": float(mar),
    }


def compute_features_for_video(npz_path: Path, meta_path: Path) -> pd.DataFrame:
    data = np.load(npz_path)
    landmarks = data["landmarks"]
    frame_indices = data["frame_indices"]

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    subject_id = npz_path.parent.name
    video_id = npz_path.stem  # e.g., "0_landmarks" / "10_landmarks"

    rows = []
    for i in range(landmarks.shape[0]):
        feats = compute_frame_features(landmarks[i])
        if feats is None:
            continue
        rows.append({
            "subject_id": subject_id,
            "video_id": video_id,
            "frame_idx": int(frame_indices[i]),
            "label": meta.get("label", None),
            **feats,
        })

    return pd.DataFrame(rows)


def run_extract_frame_features(landmarks_root: str = "data", out_csv: str = "exported_data/features_frame_level.csv"):
    root = Path(landmarks_root)
    all_dfs = []

    for npz_path in root.rglob("*_landmarks.npz"):
        meta_path = npz_path.with_name(npz_path.name.replace("_landmarks.npz", "_meta.json"))
        print("Processing:", npz_path)
        all_dfs.append(compute_features_for_video(npz_path, meta_path))

    if not all_dfs:
        raise RuntimeError(f"No *_landmarks.npz found under: {landmarks_root}")

    df_all = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_all.to_csv(out_csv, index=False)
    print("Saved frame-level features to:", out_csv, "(n=", len(df_all), ")")
    return df_all
