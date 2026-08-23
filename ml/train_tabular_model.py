"""
RazorRisk — tabular fraud model.

Operates at TRANSACTION level (the GNN operates at USER level — combining
the two is risk_aggregator.py's job).

Deliberate scope boundary: this model does NOT use graph features (degree,
community_size). Those are the GNN's job. This model only uses signals
cheap to compute at transaction-scoring time: amount pattern, velocity,
merchant history. Keeping the two models' feature sets non-overlapping is
what makes the aggregator meaningfully additive rather than two models
learning the same thing twice.

Label: a transaction is labeled fraud via is_fraud_ground_truth (synthetic
data) or the real Kaggle dataset's Class column, propagated at ingestion.
A real system would ideally have this per-transaction rather than relying
on synthetic ground truth, but the leakage discipline below is the same
either way.

Leakage discipline:
  - velocity_1h and prior_avg_amount/prior_std_amount are computed using
    ONLY transactions at or before the current one (see FEATURE_SQL) — no
    future information.
  - merchant_fraud_rate is target-encoded using ONLY the training split.
  - train/test split is by USER (ml.common.user_level_split), matching the
    GNN's split exactly, so a fraud-ring member's transactions never appear
    on both sides.
"""
import os
import json

import numpy as np
import pandas as pd

from ml.common import user_level_split, classification_report_dict
from db.database import get_raw_sqlite_connection
from utils.logger import get_logger

logger = get_logger("tabular_training")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "tabular_model.json")
SKLEARN_MODEL_PATH = os.path.join(MODEL_DIR, "tabular_model.joblib")
MERCHANT_RATES_PATH = os.path.join(MODEL_DIR, "merchant_fraud_rates.json")
TABULAR_EVAL_PATH = os.path.join(MODEL_DIR, "tabular_eval.json")

MERCHANT_SMOOTHING = 10  # additive smoothing strength for target encoding

FEATURES = [
    "amount_log", "hour_of_day", "day_of_week",
    "velocity_1h", "amount_zscore_prior", "merchant_fraud_rate",
]

# Leak-free feature SQL: velocity_1h is a correlated subquery counting only
# transactions at-or-before the current one within the trailing hour;
# prior_avg/prior_std use a ROWS frame ending one row BEFORE the current
# transaction. SQLite has no built-in STDDEV, so it's computed by hand from
# the window SUM/SUM-of-squares/COUNT over the same frame.
FEATURE_SQL = """
SELECT
    t.transaction_id,
    t.user_id,
    t.merchant_id,
    t.amount,
    t.timestamp,
    t.is_fraud_ground_truth AS is_fraud,
    CAST(strftime('%H', t.timestamp) AS INTEGER) AS hour_of_day,
    CAST(strftime('%w', t.timestamp) AS INTEGER) AS day_of_week,
    (
        SELECT COUNT(*) FROM transactions t2
        WHERE t2.user_id = t.user_id
          AND t2.timestamp <= t.timestamp
          AND t2.timestamp > datetime(t.timestamp, '-1 hours')
    ) AS velocity_1h,
    SUM(t.amount) OVER w AS prior_sum,
    SUM(t.amount * t.amount) OVER w AS prior_sumsq,
    COUNT(t.amount) OVER w AS prior_count
FROM transactions t
WINDOW w AS (
    PARTITION BY t.user_id ORDER BY t.timestamp
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
)
ORDER BY t.transaction_id
"""


def load_transactions(conn):
    df = pd.read_sql_query(FEATURE_SQL, conn)
    df["amount"] = df["amount"].astype(float)
    df["is_fraud"] = df["is_fraud"].astype(int)

    prior_avg = df["prior_sum"] / df["prior_count"].replace(0, np.nan)
    prior_var = (df["prior_sumsq"] / df["prior_count"].replace(0, np.nan)) - prior_avg ** 2
    prior_std = np.sqrt(prior_var.clip(lower=0))

    std_safe = prior_std.replace(0, np.nan)
    df["amount_zscore_prior"] = ((df["amount"] - prior_avg) / std_safe).fillna(0.0)
    df["amount_log"] = np.log1p(df["amount"])
    return df


def add_merchant_target_encoding(df, train_mask):
    global_rate = df.loc[train_mask, "is_fraud"].mean() if train_mask.any() else df["is_fraud"].mean()
    stats = df.loc[train_mask].groupby("merchant_id")["is_fraud"].agg(["sum", "count"])
    smoothed = (stats["sum"] + MERCHANT_SMOOTHING * global_rate) / (stats["count"] + MERCHANT_SMOOTHING)
    df["merchant_fraud_rate"] = df["merchant_id"].map(smoothed).fillna(global_rate)
    return df, float(global_rate), smoothed.to_dict()


class TabularModel:
    """Wraps XGBoost when available, falling back to scikit-learn's
    HistGradientBoostingClassifier when it isn't (e.g. a constrained
    install environment where the xgboost wheel can't be installed). Both
    branches share the exact same feature contract (FEATURES, in order),
    so risk_aggregator.py's live-inference code never needs to know which
    backend trained the model on disk — it only needs predict_proba()."""

    def __init__(self):
        self.backend = None
        self.model = None

    def fit(self, X_train, y_train, scale_pos_weight):
        try:
            import xgboost as xgb
            self.backend = "xgboost"
            self.model = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
                random_state=42,
            )
            logger.info(f"Training XGBoost (scale_pos_weight={scale_pos_weight:.1f})...")
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.backend = "sklearn_hgb"
            sample_weight_ratio = min(scale_pos_weight, 50)  # HGB has no scale_pos_weight; approximate via class_weight-like sampling below
            self.model = HistGradientBoostingClassifier(
                max_depth=4, learning_rate=0.05, max_iter=200, random_state=42,
            )
            logger.warning(
                "xgboost not importable in this environment — falling back to "
                "sklearn HistGradientBoostingClassifier. Functionally equivalent "
                "gradient-boosted trees; install xgboost for the intended backend."
            )
        if self.backend == "sklearn_hgb":
            sample_weight = np.where(y_train == 1, scale_pos_weight, 1.0)
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def feature_importances(self):
        imp = getattr(self.model, "feature_importances_", None)
        if imp is None:
            return {}
        return dict(zip(FEATURES, [float(x) for x in imp]))

    def save(self):
        if self.backend == "xgboost":
            self.model.save_model(XGB_MODEL_PATH)
        else:
            import joblib
            joblib.dump(self.model, SKLEARN_MODEL_PATH)
        with open(os.path.join(MODEL_DIR, "tabular_backend.json"), "w") as f:
            json.dump({"backend": self.backend}, f)

    @classmethod
    def load(cls):
        backend_path = os.path.join(MODEL_DIR, "tabular_backend.json")
        with open(backend_path) as f:
            backend = json.load(f)["backend"]
        inst = cls()
        inst.backend = backend
        if backend == "xgboost":
            import xgboost as xgb
            inst.model = xgb.XGBClassifier()
            inst.model.load_model(XGB_MODEL_PATH)
        else:
            import joblib
            inst.model = joblib.load(SKLEARN_MODEL_PATH)
        return inst


def train_tabular_model():
    conn = get_raw_sqlite_connection()
    df = load_transactions(conn)

    cur = conn.cursor()
    cur.execute("SELECT user_id FROM users ORDER BY user_id")
    user_ids = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT DISTINCT user_id FROM transactions WHERE is_fraud_ground_truth = 1")
    fraud_users = {r[0] for r in cur.fetchall()}
    y_user = np.array([1.0 if uid in fraud_users else 0.0 for uid in user_ids])

    # Same user-level split as the GNN — see ml/common.py
    train_user_mask, test_user_mask = user_level_split(user_ids, y_user)
    train_users = {uid for uid, m in zip(user_ids, train_user_mask) if m}
    test_users = {uid for uid, m in zip(user_ids, test_user_mask) if m}

    df["split"] = np.where(
        df["user_id"].isin(train_users), "train",
        np.where(df["user_id"].isin(test_users), "test", "unknown"),
    )
    train_mask = (df["split"] == "train").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()

    df, global_rate, merchant_rates = add_merchant_target_encoding(df, train_mask)

    logger.info(f"Transactions: {len(df)} | train: {int(train_mask.sum())} "
                f"({int(df.loc[train_mask, 'is_fraud'].sum())} fraud) | "
                f"test: {int(test_mask.sum())} ({int(df.loc[test_mask, 'is_fraud'].sum())} fraud)")

    X_train, y_train = df.loc[train_mask, FEATURES], df.loc[train_mask, "is_fraud"]
    X_test, y_test = df.loc[test_mask, FEATURES], df.loc[test_mask, "is_fraud"]

    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = TabularModel().fit(X_train, y_train, scale_pos_weight)

    eval_metrics = {"note": "insufficient held-out fraud examples to evaluate"}
    if len(y_test) and y_test.sum() > 0:
        test_scores = model.predict_proba(X_test)
        eval_metrics = classification_report_dict(y_test.to_numpy(), test_scores)
        logger.info(f"Tabular model held-out test metrics ({model.backend}): {eval_metrics}")
    else:
        logger.warning("Skipping tabular held-out evaluation — not enough fraud examples in the test split.")

    logger.info(f"Feature importances: {model.feature_importances()}")

    model.save()
    with open(MERCHANT_RATES_PATH, "w") as f:
        json.dump({"global_rate": global_rate, "rates": {str(k): v for k, v in merchant_rates.items()}}, f)
    with open(TABULAR_EVAL_PATH, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info(f"Tabular model ({model.backend}) + merchant rates saved to {MODEL_DIR}/")

    # Per-transaction scores for the risk_aggregator's stacker
    df["tabular_score"] = np.nan
    df.loc[train_mask, "tabular_score"] = model.predict_proba(X_train)
    df.loc[test_mask, "tabular_score"] = model.predict_proba(X_test) if len(X_test) else []
    return df[["transaction_id", "user_id", "is_fraud", "split", "tabular_score"]]


def predict_tabular_fraud_prob(feature_row: dict) -> float:
    """Live single-transaction inference — used by risk_aggregator.py.
    feature_row must contain every key in FEATURES."""
    model = TabularModel.load()
    X = pd.DataFrame([feature_row])[FEATURES]
    return float(model.predict_proba(X)[0])


if __name__ == "__main__":
    train_tabular_model()
