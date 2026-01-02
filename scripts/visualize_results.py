# scripts/visualize_results.py
from __future__ import annotations

import os
import json
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _require_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")


def _load_features_from_metadata(best_model_name_json: str) -> Optional[List[str]]:
    if not os.path.exists(best_model_name_json):
        return None
    with open(best_model_name_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feats = meta.get("features")
    if isinstance(feats, list) and all(isinstance(x, str) for x in feats):
        return feats
    return None


def _save_sorted_metrics_tables(metrics_test_all_csv: str, out_dir: str) -> None:
    _require_file(metrics_test_all_csv)
    df = pd.read_csv(metrics_test_all_csv)

    # Prefer f1_macro, fallback if missing
    sort_col = "f1_macro" if "f1_macro" in df.columns else df.columns[-1]
    df_sorted = df.sort_values(by=sort_col, ascending=False)

    out_csv = os.path.join(out_dir, "model_test_metrics_sorted.csv")
    df_sorted.to_csv(out_csv, index=False)

    # Save a Markdown table for easy copy into report
    out_md = os.path.join(out_dir, "model_test_metrics_sorted.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(df_sorted.to_markdown(index=False))

    print(f"[OK] Saved sorted metrics tables:\n  - {out_csv}\n  - {out_md}")


def _plot_confusion_matrix(confusion_csv: str, out_png: str) -> None:
    _require_file(confusion_csv)
    cm = pd.read_csv(confusion_csv, header=None).values

    fig = plt.figure()
    ax = fig.add_subplot(111)

    im = ax.imshow(cm, interpolation="nearest")
    plt.colorbar(im, ax=ax)

    ax.set_title("Confusion Matrix (Test)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Annotate cells
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"[OK] Saved confusion matrix figure:\n  - {out_png}")


def _plot_permutation_importance(
    model_path: str,
    processed_dir: str,
    features: List[str],
    out_png: str,
    *,
    label_col: str = "label_id",
    split: str = "test",
    n_repeats: int = 15,
    random_state: int = 42,
) -> None:
    _require_file(model_path)

    split_path = os.path.join(processed_dir, f"{split}.parquet")
    _require_file(split_path)

    df = pd.read_parquet(split_path).dropna(subset=[label_col])
    X = df[features]
    y = df[label_col].astype(int)

    model = joblib.load(model_path)

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importances_mean = result.importances_mean
    importances_std = result.importances_std

    order = np.argsort(importances_mean)[::-1]
    feats_sorted = [features[i] for i in order]
    mean_sorted = importances_mean[order]
    std_sorted = importances_std[order]

    fig = plt.figure(figsize=(10, max(4, 0.35 * len(features))))
    ax = fig.add_subplot(111)

    y_pos = np.arange(len(features))
    ax.barh(y_pos, mean_sorted, xerr=std_sorted)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats_sorted)
    ax.invert_yaxis()
    ax.set_title(f"Permutation Importance ({split} split)")
    ax.set_xlabel("Importance (decrease in accuracy)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"[OK] Saved permutation importance figure:\n  - {out_png}")


def _plot_feature_distributions(
    processed_dir: str,
    features: List[str],
    out_png: str,
    *,
    label_col: str = "label_id",
    split: str = "test",
    max_features: int = 12,
) -> None:
    split_path = os.path.join(processed_dir, f"{split}.parquet")
    _require_file(split_path)

    df = pd.read_parquet(split_path).dropna(subset=[label_col])
    # keep only a manageable number of features for a single figure
    feats = features[:max_features]

    # Create a grid layout
    n = len(feats)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(12, max(4, 3 * nrows)))

    for idx, feat in enumerate(feats, start=1):
        ax = fig.add_subplot(nrows, ncols, idx)

        g0 = df[df[label_col].astype(int) == 0][feat].dropna().values
        g1 = df[df[label_col].astype(int) == 10][feat].dropna().values

        # boxplot 
        ax.boxplot([g0, g1], labels=["Label 0 (Alert)", "Label 10 (Drowsy)"])
        ax.set_title(feat)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

    fig.suptitle(f"Feature Distributions by Class ({split} split)", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved feature distribution figure:\n  - {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate figures/tables from training results.")
    ap.add_argument("--processed_dir", default="processed_data", help="Where train/test parquet files live.")
    ap.add_argument("--results_dir", default=os.path.join("results", "train"), help="Where best_model + metrics live.")
    ap.add_argument("--label_col", default="label_id", help="Label column name in parquet files.")
    ap.add_argument("--split_for_plots", default="test", choices=["train", "val", "test"], help="Which split to plot on.")
    ap.add_argument("--n_repeats", type=int, default=15, help="Permutation importance repeats.")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--max_features", type=int, default=12, help="Max features to show in distribution figure.")
    args = ap.parse_args()

    # Inputs
    best_model_path = os.path.join(args.results_dir, "best_model.joblib")
    best_model_meta = os.path.join(args.results_dir, "best_model_name.json")
    metrics_test_all_csv = os.path.join(args.results_dir, "metrics_test_all.csv")
    confusion_csv = os.path.join(args.results_dir, "confusion_matrix_test.csv")

    _require_file(best_model_path)
    _require_file(metrics_test_all_csv)
    _require_file(confusion_csv)

    # Output dirs
    figures_dir = os.path.join(args.results_dir, "figures")
    tables_dir = os.path.join(args.results_dir, "tables")
    _safe_makedirs(figures_dir)
    _safe_makedirs(tables_dir)

    # Features
    features = _load_features_from_metadata(best_model_meta)
    if not features:
        raise RuntimeError(
            f"Could not load feature list from {best_model_meta}. "
            "Make sure train_models_lib writes `features` into best_model_name.json."
        )

    # (1) Table: model test metrics 
    _save_sorted_metrics_tables(metrics_test_all_csv, tables_dir)

    # (2) Figure: confusion matrix
    _plot_confusion_matrix(
        confusion_csv=confusion_csv,
        out_png=os.path.join(figures_dir, "confusion_matrix_test.png"),
    )

    # (3) Figure: permutation importance 
    _plot_permutation_importance(
        model_path=best_model_path,
        processed_dir=args.processed_dir,
        features=features,
        out_png=os.path.join(figures_dir, f"permutation_importance_{args.split_for_plots}.png"),
        label_col=args.label_col,
        split=args.split_for_plots,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )

    # (4) Figure: feature distributions by label
    _plot_feature_distributions(
        processed_dir=args.processed_dir,
        features=features,
        out_png=os.path.join(figures_dir, f"feature_distributions_{args.split_for_plots}.png"),
        label_col=args.label_col,
        split=args.split_for_plots,
        max_features=args.max_features,
    )

    print("\n[DONE] All artifacts saved under:")
    print(f"  - {tables_dir}")
    print(f"  - {figures_dir}")


if __name__ == "__main__":
    main()
