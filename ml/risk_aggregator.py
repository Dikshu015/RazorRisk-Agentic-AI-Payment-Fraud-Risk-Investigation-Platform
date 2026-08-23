"""
RazorRisk — risk aggregator: stacker training + live composite scoring.

Two responsibilities, kept in one file because they share the same
contract (tabular_score, gnn_score -> combined probability):

1. train_stacker(): offline. Combines the two independently-trained models'
   held-out scores via a small logistic regression stacker — LEARNED
   combination weights rather than an arbitrary hand-picked average (an
   earlier version of this module used a fixed 0.35/0.45/0.20 weighted sum;
   that number was never validated against anything). Evaluated against
   tabular-only and GNN-only on the SAME test transactions, to check the
   combination is actually additive and not just theatre.

2. calculate_composite_risk_score(): online. The live per-transaction
   scoring path the API calls. Loads the trained stacker's coefficients
   (2 inputs, 3 parameters — small enough to fully explain) and combines a
   fresh tabular prediction with a fresh inductive GNN prediction. The
   velocity/VPN-proxy multiplier is applied AFTER the calibrated
   probability as an explicit, separately-labeled rule-based overlay, not
   folded into the "learned" score — VPN/proxy flags aren't in either
   model's feature set, so multiplying them in is a deliberate rule, and
   pretending it was learned would misrepresent what actually happened.
"""
import os
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.common import classification_report_dict
from ml.train_tabular_model import train_tabular_model, predict_tabular_fraud_prob, MERCHANT_RATES_PATH
from ml.train_gnn import train_gnn, GraphSAGEInference, GNN_WEIGHTS_PATH
from ml.risk_graph import build_user_graph, detect_communities, fetch_node_features, build_adjacency
from db.database import get_raw_sqlite_connection
from utils.logger import get_logger

logger = get_logger("risk_aggregator")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
AGGREGATOR_PATH = os.path.join(MODEL_DIR, "aggregator.npz")
AGGREGATOR_EVAL_PATH = os.path.join(MODEL_DIR, "aggregator_eval.json")

HIGH_RISK_THRESHOLD = 70.0
CRITICAL_THRESHOLD = 90.0
MEDIUM_THRESHOLD = 40.0


def train_stacker():
    """Runs the full training sequence (tabular -> GNN -> stacker) and
    persists all three artifacts. This is the one function the admin
    pipeline endpoints call after a reseed/ingest."""
    tabular_df = train_tabular_model()
    user_ids, gnn_scores, y_user, train_user_mask, test_user_mask = train_gnn()

    gnn_by_user = pd.DataFrame({"user_id": user_ids, "gnn_score": gnn_scores})
    df = tabular_df.merge(gnn_by_user, on="user_id", how="left")
    if df["gnn_score"].isna().any():
        fallback = float(np.mean(gnn_scores))
        logger.warning(f"{df['gnn_score'].isna().sum()} transactions had no matching GNN score — filling with population mean.")
        df["gnn_score"] = df["gnn_score"].fillna(fallback)

    train_mask = (df["split"] == "train").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()

    X_train = df.loc[train_mask, ["tabular_score", "gnn_score"]]
    y_train = df.loc[train_mask, "is_fraud"]
    X_test = df.loc[test_mask, ["tabular_score", "gnn_score"]]
    y_test = df.loc[test_mask, "is_fraud"]

    if y_train.nunique() < 2:
        logger.warning("Training split has only one class — stacker cannot be fit; using equal-weight fallback coefficients.")
        coef = np.array([2.0, 2.0])
        intercept = np.array([-2.0])
    else:
        stacker = LogisticRegression(class_weight="balanced")
        stacker.fit(X_train, y_train)
        coef = stacker.coef_[0]
        intercept = stacker.intercept_
        logger.info(f"Stacker learned weights: tabular_coef={coef[0]:.4f} gnn_coef={coef[1]:.4f} intercept={intercept[0]:.4f}")

    eval_metrics = {"note": "insufficient held-out fraud examples to evaluate"}
    if len(y_test) and y_test.sum() > 0:
        z = coef[0] * X_test["tabular_score"] + coef[1] * X_test["gnn_score"] + intercept[0]
        combined_scores = 1 / (1 + np.exp(-z))
        eval_metrics = {
            "tabular_only": classification_report_dict(y_test, X_test["tabular_score"]),
            "gnn_only": classification_report_dict(y_test, X_test["gnn_score"]),
            "stacked": classification_report_dict(y_test, combined_scores),
        }
        logger.info(f"Aggregator comparison (tabular-only vs GNN-only vs stacked): {eval_metrics}")
    else:
        logger.warning("Skipping stacker held-out evaluation — not enough fraud examples in the test split.")

    np.savez(AGGREGATOR_PATH, coef=coef, intercept=intercept)
    with open(AGGREGATOR_EVAL_PATH, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info(f"Aggregator weights saved to {AGGREGATOR_PATH}")
    return eval_metrics


class _LiveModels:
    """Loaded lazily on first score request and cached for the process
    lifetime. Call reset() after a pipeline retrain so the next request
    picks up newly trained weights instead of stale in-memory ones."""
    gnn = None
    coef = None
    intercept = None
    merchant_rates = None
    merchant_global_rate = None

    @classmethod
    def reset(cls):
        cls.gnn = None
        cls.coef = None
        cls.intercept = None
        cls.merchant_rates = None
        cls.merchant_global_rate = None

    @classmethod
    def ensure_loaded(cls):
        if cls.gnn is None:
            if not os.path.exists(GNN_WEIGHTS_PATH):
                train_gnn()
            cls.gnn = GraphSAGEInference()
        if cls.coef is None:
            if not os.path.exists(AGGREGATOR_PATH):
                train_stacker()
            data = np.load(AGGREGATOR_PATH)
            cls.coef, cls.intercept = data["coef"], float(data["intercept"][0])
        if cls.merchant_rates is None:
            if not os.path.exists(MERCHANT_RATES_PATH):
                train_tabular_model()
            with open(MERCHANT_RATES_PATH) as f:
                data = json.load(f)
            cls.merchant_rates = data["rates"]
            cls.merchant_global_rate = data["global_rate"]


def _merchant_fraud_rate(merchant_id: str) -> float:
    _LiveModels.ensure_loaded()
    return _LiveModels.merchant_rates.get(str(merchant_id), _LiveModels.merchant_global_rate)


def live_gnn_score_and_evidence(user_id: str):
    """Rebuilds a fresh graph snapshot and runs one inductive forward pass.
    Cheap at this dataset's scale (sub-second for a few thousand nodes) —
    the piece a production system would move to a scheduled/cached job
    instead of recomputing synchronously per request, flagged here rather
    than silently accepted."""
    conn = get_raw_sqlite_connection()
    try:
        G = build_user_graph(conn)
        communities, community_size = detect_communities(G)
        user_ids, X_raw = fetch_node_features(conn, G, community_size)
        A_mean = build_adjacency(G, user_ids)

        _LiveModels.ensure_loaded()
        scores = _LiveModels.gnn.score_all(X_raw, A_mean)

        if user_id not in user_ids:
            return 0.0, {"graph_degree": 0, "community_size": 1, "shared_device_accounts": 1, "shared_ip_accounts": 1}

        idx = user_ids.index(user_id)
        gnn_score = float(scores[idx])
        degree = G.degree(user_id) if user_id in G else 0
        comm_size = community_size.get(user_id, 1)

        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(DISTINCT user_id) FROM transactions
            WHERE device_id = (SELECT device_id FROM transactions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1)
        """, (user_id,))
        shared_device = cur.fetchone()[0] or 1
        cur.execute("""
            SELECT COUNT(DISTINCT user_id) FROM transactions
            WHERE ip_address = (SELECT ip_address FROM transactions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1)
        """, (user_id,))
        shared_ip = cur.fetchone()[0] or 1

        return gnn_score, {
            "graph_degree": degree, "community_size": comm_size,
            "shared_device_accounts": shared_device, "shared_ip_accounts": shared_ip,
        }
    finally:
        conn.close()


def live_tabular_score(txn_payload: dict) -> float:
    conn = get_raw_sqlite_connection()
    try:
        user_id = txn_payload["user_id"]
        amount = float(txn_payload.get("amount", 0.0))
        cur = conn.cursor()

        cur.execute("""
            SELECT COUNT(*) FROM transactions
            WHERE user_id = ? AND timestamp > datetime('now', '-1 hours')
        """, (user_id,))
        velocity_1h = cur.fetchone()[0] + 1

        cur.execute("SELECT AVG(amount), COUNT(amount) FROM transactions WHERE user_id = ?", (user_id,))
        prior_avg, prior_count = cur.fetchone()
        cur.execute("SELECT amount FROM transactions WHERE user_id = ?", (user_id,))
        prior_amounts = [r[0] for r in cur.fetchall()]
        prior_std = float(np.std(prior_amounts, ddof=1)) if prior_count and prior_count > 1 else 0.0
        amount_zscore_prior = 0.0 if not prior_std else (amount - float(prior_avg)) / prior_std

        import datetime as _dt
        feature_row = {
            "amount_log": float(np.log1p(amount)),
            "hour_of_day": _dt.datetime.now().hour,
            "day_of_week": _dt.datetime.now().weekday(),
            "velocity_1h": txn_payload.get("velocity_1h", velocity_1h),
            "amount_zscore_prior": amount_zscore_prior,
            "merchant_fraud_rate": _merchant_fraud_rate(txn_payload.get("merchant_id", "")),
        }
        return predict_tabular_fraud_prob(feature_row)
    finally:
        conn.close()


def calculate_composite_risk_score(txn_payload: dict) -> dict:
    """Live per-transaction scoring — the API's hot path."""
    user_id = txn_payload.get("user_id", "")
    amount = float(txn_payload.get("amount", 0.0))

    tabular_prob = live_tabular_score(txn_payload)
    gnn_prob, graph_evidence = live_gnn_score_and_evidence(user_id)

    _LiveModels.ensure_loaded()
    z = float(_LiveModels.coef[0] * tabular_prob + _LiveModels.coef[1] * gnn_prob + _LiveModels.intercept)
    calibrated_prob = 1 / (1 + np.exp(-z))

    velocity_1h = int(txn_payload.get("velocity_1h", 1))
    velocity_mult = 1.0
    if velocity_1h >= 10:
        velocity_mult = 1.5
    elif velocity_1h >= 5:
        velocity_mult = 1.25
    if txn_payload.get("is_vpn_proxy", False) or txn_payload.get("is_suspicious_proxy", False):
        velocity_mult *= 1.15

    raw_score = calibrated_prob * velocity_mult * 100.0
    final_risk_score = round(float(min(max(raw_score, 0.0), 100.0)), 1)

    if final_risk_score >= CRITICAL_THRESHOLD:
        tier, decision = "CRITICAL", "BLOCK_AND_INVESTIGATE"
    elif final_risk_score >= HIGH_RISK_THRESHOLD:
        tier, decision = "HIGH", "HOLD_FOR_INVESTIGATION"
    elif final_risk_score >= MEDIUM_THRESHOLD:
        tier, decision = "MEDIUM", "MONITOR"
    else:
        tier, decision = "LOW", "APPROVE"

    result = {
        "risk_score": final_risk_score,
        "tabular_score": round(tabular_prob * 100.0, 1),
        "gnn_score": round(gnn_prob * 100.0, 1),
        "stacker_calibrated_score": round(float(calibrated_prob) * 100.0, 1),
        "velocity_multiplier": round(velocity_mult, 2),
        "risk_tier": tier,
        "decision": decision,
        "graph_evidence": {
            "shared_device_accounts": graph_evidence["shared_device_accounts"],
            "shared_ip_accounts": graph_evidence["shared_ip_accounts"],
            "community_size": graph_evidence["community_size"],
            "graph_degree": graph_evidence["graph_degree"],
        },
    }
    logger.info(f"Transaction User:{user_id} Amt:{amount} -> RiskScore:{final_risk_score}/100 [{tier}] Action:{decision}")
    return result


if __name__ == "__main__":
    train_stacker()
    sample_txn = {
        "user_id": "USER_RING1_1", "device_id": "DEV_FRAUD_RING1",
        "ip_address": "185.220.101.44", "merchant_id": "MCH_042",
        "amount": 88000, "velocity_1h": 12, "is_vpn_proxy": True,
    }
    print("Risk Score Output:", calculate_composite_risk_score(sample_txn))
