"""
experiments_lib.py

Reusable sweep utilities.

Data vs results policy:
- exported_data/: data artifacts ONLY (e.g., window-level CSVs)
- processed_data/: data artifacts ONLY (e.g., parquet splits)
- results/: model/results artifacts ONLY (metrics, models, reports, confusion matrices, summaries)

This sweep repeatedly:
  1) build window-level features from a fixed frame-level CSV -> exported_data/...
  2) prepare dataset splits -> processed_data/...
  3) train & compare models -> results/...
"""

from __future__ import annotations

import os
import itertools
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from window_features import run_build_window_features
from dataset_prep import run_prepare_dataset
from train_models_lib import train_and_compare


def run_sweep(
    *,
    frame_csv: str,
    export_root: str,
    processed_root: str,
    results_root: str,
    window_sizes: List[int],
    blink_ear_threshs: List[float],
    yawn_mar_threshs: List[float],
    min_frames_list: List[int],
    random_state: int = 42,
    choose_by: str = "f1_macro",
    summary_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns a summary DataFrame sorted by `choose_by` (based on best-on-test metrics).

    Per experiment, we write:
      Data:
        <export_root>/<exp_name>/features_window_level.csv
        <processed_root>/<exp_name>/{train,val,test}.parquet (+ label_map.json)
      Results:
        <results_root>/<exp_name>/*
    """
    os.makedirs(export_root, exist_ok=True)
    os.makedirs(processed_root, exist_ok=True)
    os.makedirs(results_root, exist_ok=True)

    rows: List[Dict] = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for ws, be, ym, mf in itertools.product(window_sizes, blink_ear_threshs, yawn_mar_threshs, min_frames_list):
        exp_name = f"ws{ws}_be{be:.2f}_ym{ym:.2f}_mf{mf}"

        exp_export_dir = os.path.join(export_root, exp_name)           # DATA
        exp_processed_dir = os.path.join(processed_root, exp_name)     # DATA
        exp_results_dir = os.path.join(results_root, exp_name)         # RESULTS

        os.makedirs(exp_export_dir, exist_ok=True)
        os.makedirs(exp_processed_dir, exist_ok=True)
        os.makedirs(exp_results_dir, exist_ok=True)

        window_csv = os.path.join(exp_export_dir, "features_window_level.csv")

        print(f"\n==== EXP: {exp_name} ====")

        # 1) Window features -> exported_data (DATA)
        run_build_window_features(
            frame_csv=frame_csv,
            out_csv=window_csv,
            window_size=ws,
            blink_ear_thresh=be,
            yawn_mar_thresh=ym,
        )

        # 2) Dataset prep -> processed_data (DATA)
        run_prepare_dataset(
            input_csv=window_csv,
            output_dir=exp_processed_dir,
            min_frames=mf,
            random_state=random_state,
        )

        # 3) Train & compare -> results (RESULTS)
        _, best_name, best_test_metrics = train_and_compare(
            processed_dir=exp_processed_dir,
            results_dir=exp_results_dir,
            random_state=random_state,
            choose_by=choose_by,
        )

        rows.append({
            "exp": exp_name,
            "window_size": ws,
            "blink_ear_thresh": be,
            "yawn_mar_thresh": ym,
            "min_frames": mf,
            "best_model": best_name,
            **best_test_metrics,
        })

    summary = pd.DataFrame(rows).sort_values(by=choose_by, ascending=False)

    # Summary is a RESULTS artifact -> put it under results_root by default
    out_csv = summary_csv or os.path.join(results_root, f"summary_{ts}.csv")
    summary.to_csv(out_csv, index=False)
    print("\nSaved sweep summary:", out_csv)

    return summary
