"""
Shared utilities across ml/. Split logic lives here (rather than duplicated
in train_gnn.py and train_tabular_model.py) so both models are evaluated on
the exact same held-out users — required for the risk_aggregator's stacker
to combine their scores on a fair, leak-free basis, and ported directly from
the more rigorous train/test discipline in the merged reference project.
"""
import numpy as np

RNG_SEED = 42
TRAIN_FRAC = 0.7


def user_level_split(user_ids, y, seed=RNG_SEED, train_frac=TRAIN_FRAC):
    """
    Stratified split by USER (not by transaction) — a user's transactions
    must all land on the same side of the split, otherwise the tabular
    model could see a fraud-ring member's behavior in training and be
    evaluated on the same person's transactions in test, which isn't a
    real held-out evaluation.

    Returns (train_mask, test_mask) boolean arrays aligned to user_ids/y.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_pos_train = max(1, int(len(pos_idx) * train_frac)) if len(pos_idx) else 0
    n_neg_train = int(len(neg_idx) * train_frac)
    train_idx = np.concatenate([pos_idx[:n_pos_train], neg_idx[:n_neg_train]])

    train_mask = np.zeros(len(y), dtype=bool)
    train_mask[train_idx] = True
    test_mask = ~train_mask
    return train_mask, test_mask


def classification_report_dict(y_true, y_score, threshold=0.5):
    """Held-out evaluation metrics, returned as a dict so callers can both
    print and log/persist them (rather than only printing, which the
    deployed API can't capture)."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix,
    )
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_score) if len(set(y_true.tolist())) > 1 else float("nan")
    ap = average_precision_score(y_true, y_score)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(auc) if auc == auc else None,  # NaN-safe
        "pr_auc": float(ap),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
