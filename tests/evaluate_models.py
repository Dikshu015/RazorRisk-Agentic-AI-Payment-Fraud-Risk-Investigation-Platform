"""Reproducible evaluation runner for RazorRisk.

Usage
-----
Synthetic/project artifacts (no retraining):
    python tests/evaluate_models.py --dataset synthetic

Synthetic, retrain all models first:
    python tests/evaluate_models.py --dataset synthetic --retrain

Public ULB/Kaggle credit-card dataset:
    1. Put creditcard.csv in data/ (or pass --csv PATH).
    2. Run:
       python tests/evaluate_models.py --dataset kaggle --csv data/creditcard.csv

The Kaggle benchmark is intentionally tabular-only. The public dataset contains
Time, V1..V28, Amount and Class, but no stable user/device/IP identities. It is
therefore not honest to manufacture a real graph benchmark from it and call the
result observed fraud-network performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.common import classification_report_dict


def _print_metrics(title: str, metrics: dict):
    print(f"\n## {title}\n")
    print("| Metric | Value |")
    print("|---|---:|")
    for key in ("roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "tp", "fp", "fn", "tn"):
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                print(f"| {key} | {value:.6f} |")
            else:
                print(f"| {key} | {value} |")


def evaluate_synthetic(retrain: bool = False):
    if retrain:
        from ml.risk_aggregator import train_stacker
        metrics = train_stacker()
    else:
        model_dir = ROOT / "ml" / "models"
        with open(model_dir / "aggregator_eval.json") as f:
            metrics = json.load(f)

    print("# RazorRisk Synthetic Evaluation")
    print("\nSource: project synthetic dataset; held-out split produced by the current training pipeline.\n")
    print("| Model | ROC-AUC | PR-AUC | Accuracy | Balanced Acc. | Precision | Recall | F1 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, label in (("tabular_only", "Tabular / XGBoost"), ("gnn_only", "GraphSAGE / GNN"), ("stacked", "Learned stacker")):
        m = metrics[name]
        print(f"| {label} | {m['roc_auc']:.6f} | {m['pr_auc']:.6f} | {m['accuracy']:.6f} | {m['balanced_accuracy']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} |")
    print("\nConfusion matrices:")
    for name, label in (("tabular_only", "Tabular / XGBoost"), ("gnn_only", "GraphSAGE / GNN"), ("stacked", "Learned stacker")):
        _print_metrics(label, metrics[name])


def evaluate_kaggle(csv_path: str):
    import pandas as pd
    import numpy as np
    from xgboost import XGBClassifier

    csv = Path(csv_path)
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found. Download the ULB/Kaggle 'Credit Card Fraud Detection' "
            "creditcard.csv and place it at data/creditcard.csv."
        )
    df = pd.read_csv(csv)
    required = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    # Time-ordered split: later transactions are the held-out set.
    df = df.sort_values("Time").reset_index(drop=True)
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    features = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    X_train, y_train = train[features], train["Class"].astype(int)
    X_test, y_test = test[features], test["Class"].astype(int)
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        objective="binary:logistic", eval_metric="aucpr",
        scale_pos_weight=neg / max(pos, 1), random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train)
    score = model.predict_proba(X_test)[:, 1]
    metrics = classification_report_dict(y_test.to_numpy(), score)

    print("# ULB/Kaggle External Tabular Benchmark")
    print("\n**Important:** this is a separate external benchmark, not the RazorRisk GNN benchmark. The public dataset has no user/device/IP identities, so graph performance is not evaluated here. The split is chronological (first 80% train, last 20% test).\n")
    _print_metrics("XGBoost on Kaggle/ULB V1–V28 + Time + Amount", metrics)
    print(f"\nRows: {len(df):,} | Train: {len(train):,} ({int(y_train.sum()):,} fraud) | Test: {len(test):,} ({int(y_test.sum()):,} fraud)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "kaggle"], required=True)
    ap.add_argument("--retrain", action="store_true", help="Retrain the full RazorRisk synthetic pipeline before evaluating.")
    ap.add_argument("--csv", default="data/creditcard.csv", help="Path to Kaggle/ULB creditcard.csv")
    args = ap.parse_args()
    if args.dataset == "synthetic":
        evaluate_synthetic(args.retrain)
    else:
        evaluate_kaggle(args.csv)


if __name__ == "__main__":
    main()
