"""
Temporary CI diagnostic for the full synthetic fraud scenario matrix.

This is intentionally separate from the normal pytest suite: it regenerates the
synthetic dataset, retrains tabular + GNN + stacker from the repository's
committed hyperparameters, then runs every PASS/partial assertion in
tests/test_edge_case_matrix.py plus the three frontend fraud-demo presets.

The script is for audit/debugging only and is not part of the product runtime.
"""
from __future__ import annotations

import json
import os
import unittest

# CI passes DATABASE_URL explicitly. Keep this assertion here so the script
# cannot accidentally train against a developer/production database.
if not os.getenv("DATABASE_URL", "").startswith("sqlite:///"):
    raise RuntimeError("full_scenario_audit.py requires an explicit SQLite DATABASE_URL")

from data.generate_synthetic_data import generate_dataset
from db.database import init_db, get_raw_sqlite_connection
from ml.risk_aggregator import calculate_composite_risk_score, train_stacker, _LiveModels
from ml.decision_policy import apply_decision_policy
import tests.test_edge_case_matrix as matrix


def latest_context(user_id: str):
    conn = get_raw_sqlite_connection()
    try:
        row = conn.execute(
            "SELECT device_id, ip_address, timestamp FROM transactions "
            "WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        if not row:
            raise AssertionError(f"No transaction history for {user_id}")
        return row
    finally:
        conn.close()


def score_demo(user_id, merchant_id, amount, velocity, is_vpn=True):
    device_id, ip_address, timestamp = latest_context(user_id)
    txn = {
        "transaction_id": f"AUDIT_{user_id}",
        "user_id": user_id,
        "device_id": device_id,
        "ip_address": ip_address,
        "merchant_id": merchant_id,
        "amount": amount,
        "is_vpn_proxy": is_vpn,
        "is_suspicious_proxy": is_vpn,
        "velocity_enabled": True,
        "velocity_1h": velocity,
        "timestamp": timestamp,
    }
    risk = calculate_composite_risk_score(txn)
    policy = apply_decision_policy(txn, risk)
    return risk, policy


def run_matrix():
    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    suite.addTests(loader.loadTestsFromTestCase(matrix.TestGoldenMatrixBenign))
    suite.addTests(loader.loadTestsFromTestCase(matrix.TestGoldenMatrixFraud))
    suite.addTests(loader.loadTestsFromTestCase(matrix.TestGoldenMatrixDocumentedGaps))

    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(
        f"\nGOLDEN MATRIX: tests={result.testsRun} "
        f"failures={len(result.failures)} errors={len(result.errors)}"
    )
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    db = os.getenv("DATABASE_URL")
    try:
        os.remove(db[len("sqlite:///"):])
    except FileNotFoundError:
        pass

    init_db()
    generated = generate_dataset()
    print(f"\nSYNTHETIC DATASET: {generated} transactions")

    _LiveModels.reset()
    eval_metrics = train_stacker()
    print("\nSTACKER EVALUATION:")
    print(json.dumps(eval_metrics, indent=2))

    # Force the newly trained artifacts into the live scoring process.
    _LiveModels.reset()

    run_matrix()

    demos = [
        ("Ring 1", "USER_RING1_1", "MCH_042", 88000, 25),
        ("Ring 2", "USER_RING2_1", "MCH_042", 95000, 30),
        ("Carding", "USER_CARDER_X", "MCH_042", 49, 15),
    ]

    print("\nFRONTEND FRAUD DEMOS:")
    for name, user, merchant, amount, velocity in demos:
        risk, policy = score_demo(user, merchant, amount, velocity)
        print(
            f"{name:8} | GNN={risk['gnn_score']:5.1f}% "
            f"| Tabular={risk['tabular_score']:5.1f}% "
            f"| Stacker={risk['stacker_calibrated_score']:5.1f}% "
            f"| Risk={risk['risk_score']:5.1f} "
            f"| Tier={risk['risk_tier']:8} "
            f"| Decision={policy['decision']:<14} "
            f"| Reasons={','.join(policy['review_reasons']) or '-'}"
        )

    print("\nFULL SCENARIO AUDIT PASSED.")


if __name__ == "__main__":
    main()
