"""
train_models_lib.py

Reusable training/evaluation utilities

Expected inputs (produced by dataset_prep.py):
  <processed_dir>/train.parquet
  <processed_dir>/val.parquet   
  <processed_dir>/test.parquet

Outputs:
  <results_dir>/metrics_select.csv
  <results_dir>/metrics_test_best.csv
  <results_dir>/best_model.joblib
  <results_dir>/classification_report_test.txt
  <results_dir>/confusion_matrix_test.csv
  <results_dir>/best_model_name.json
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

import joblib


# Keep the same features as your baseline idea (EAR/MAR + ratios + dynamics)
DEFAULT_FEATURES: List[str] = [
    "ear_mean_mean", "ear_mean_std",
    "mar_mean", "mar_std",
    "blink_ratio", "yawn_ratio",
    "ear_diff_mean",
]


@dataclass
class SplitData:
    train: pd.DataFrame
    val: Optional[pd.DataFrame]
    test: pd.DataFrame


def load_splits(processed_dir: str) -> SplitData:
    train_path = os.path.join(processed_dir, "train.parquet")
    test_path = os.path.join(processed_dir, "test.parquet")
    val_path = os.path.join(processed_dir, "val.parquet")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing: {test_path}")

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    val = pd.read_parquet(val_path) if os.path.exists(val_path) else None

    return SplitData(train=train, val=val, test=test)


def make_models(random_state: int) -> Dict[str, object]:
    """
    Multi-model comparison set.
    The first one (RF_baseline) is your previous baseline idea.
    """
    models: Dict[str, object] = {}

    models["RF"] = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )

    models["LogReg"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
    ])

    # LinearSVC doesn't output probabilities; calibrate for a more stable decision boundary
    models["LinearSVM"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            estimator=LinearSVC(random_state=random_state),
            method="sigmoid",
            cv=3
        )),
    ])

    models["HGB"] = HistGradientBoostingClassifier(random_state=random_state)

    return models


def _compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def fit_and_eval(model, X_train, y_train, X_eval, y_eval) -> Dict[str, float]:
    model.fit(X_train, y_train)
    pred = model.predict(X_eval)
    return _compute_metrics(y_eval, pred)


def train_and_compare(
    processed_dir: str,
    results_dir: str,
    *,
    features: Optional[List[str]] = None,
    random_state: int = 42,
    choose_by: str = "f1_macro",
) -> Tuple[pd.DataFrame, str, Dict[str, float]]:
    """
    Train multiple models on train split, select best on val if available (else test),
    then evaluate best on test and write artifacts.

    Returns:
      metrics_select_df, best_model_name, best_test_metrics
    """
    os.makedirs(results_dir, exist_ok=True)

    splits = load_splits(processed_dir)
    features = features or DEFAULT_FEATURES

    # Prefer val for selection when present
    eval_df = splits.val if splits.val is not None else splits.test
    eval_name = "val" if splits.val is not None else "test"

    # Drop missing labels
    train = splits.train.dropna(subset=["label_id"])
    eval_df = eval_df.dropna(subset=["label_id"])
    test = splits.test.dropna(subset=["label_id"])

    X_train, y_train = train[features], train["label_id"].astype(int)
    X_eval, y_eval = eval_df[features], eval_df["label_id"].astype(int)
    X_test, y_test = test[features], test["label_id"].astype(int)

    models = make_models(random_state)

    rows_select: List[dict] = []
    rows_test_all: List[dict] = []

    best_name = None
    best_score = float("-inf")
    best_model = None

    for name, model in models.items():
        # fit once
        model.fit(X_train, y_train)

        # 1) selection metrics (val preferred else test) for choosing best model
        pred_eval = model.predict(X_eval)
        m_eval = _compute_metrics(y_eval, pred_eval)
        score = float(m_eval.get(choose_by, m_eval["f1_macro"]))
        rows_select.append({"model": name, "select_split": eval_name, **m_eval})

        # 2) test metrics (always test) for robustness aggregation later
        pred_test = model.predict(X_test)
        m_test = _compute_metrics(y_test, pred_test)
        rows_test_all.append({"model": name, **m_test})

        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    assert best_model is not None and best_name is not None

    metrics_select_df = pd.DataFrame(rows_select).sort_values(by=choose_by, ascending=False)
    metrics_select_df.to_csv(os.path.join(results_dir, "metrics_select.csv"), index=False)
    
    metrics_test_all_df = pd.DataFrame(rows_test_all).sort_values(by="f1_macro", ascending=False)
    metrics_test_all_df.to_csv(os.path.join(results_dir, "metrics_test_all.csv"), index=False)

    # Refit best on train, evaluate on test
    best_model.fit(X_train, y_train)
    test_pred = best_model.predict(X_test)
    best_test_metrics = _compute_metrics(y_test, test_pred)

    pd.DataFrame([{"model": best_name, **best_test_metrics}]).to_csv(
        os.path.join(results_dir, "metrics_test_best.csv"), index=False
    )

    # Save model + reports
    joblib.dump(best_model, os.path.join(results_dir, "best_model.joblib"))

    with open(os.path.join(results_dir, "best_model_name.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"best_model": best_name, "selected_on": eval_name, "choose_by": choose_by, "features": features},
            f,
            indent=2,
        )

    with open(os.path.join(results_dir, "classification_report_test.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, test_pred, digits=4))

    cm = confusion_matrix(y_test, test_pred)
    pd.DataFrame(cm).to_csv(os.path.join(results_dir, "confusion_matrix_test.csv"), index=False)

    return metrics_select_df, best_name, best_test_metrics
