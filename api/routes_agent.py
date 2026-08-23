import json
from fastapi import APIRouter, HTTPException
from agent.graph_agent import investigation_agent
from ml.risk_aggregator import calculate_composite_risk_score
from db.database import get_raw_sqlite_connection
from utils.logger import get_logger

logger = get_logger("api_agent")

router = APIRouter(prefix="/api/v1/investigations", tags=["Agentic Investigations"])

@router.post("/run/{transaction_id}")
def run_investigation(transaction_id: str):
    """Triggers LangGraph Investigation Agent on a specific transaction ID."""
    conn = get_raw_sqlite_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT t.user_id, t.device_id, t.ip_address, t.merchant_id, t.amount, t.velocity_1h,
               d.is_vpn_proxy, ip.is_suspicious_proxy
        FROM transactions t
        LEFT JOIN devices d ON t.device_id = d.device_id
        LEFT JOIN ip_addresses ip ON t.ip_address = ip.ip_address
        WHERE t.transaction_id = ?
    """, (transaction_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found.")

    txn_payload = {
        "transaction_id": transaction_id,
        "user_id": row[0],
        "device_id": row[1],
        "ip_address": row[2],
        "merchant_id": row[3],
        "amount": row[4],
        "velocity_1h": row[5],
        "is_vpn_proxy": bool(row[6]),
        "is_suspicious_proxy": bool(row[7])
    }

    risk_res = calculate_composite_risk_score(txn_payload)
    investigation_res = investigation_agent.investigate(txn_payload, risk_res)

    # Save to database
    cursor.execute("""
        INSERT OR REPLACE INTO investigation_reports
        (investigation_id, transaction_id, risk_score, evidence_json, fraud_hypothesis, recommended_action, summary_report)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        investigation_res["investigation_id"], transaction_id, risk_res["risk_score"],
        json.dumps(investigation_res["evidence"]), investigation_res["fraud_hypothesis"],
        investigation_res["recommended_action"], investigation_res["summary_report"]
    ))

    conn.commit()
    conn.close()

    return investigation_res

@router.get("/{transaction_id}")
def get_investigation_report(transaction_id: str):
    """Retrieves an existing investigation report for a transaction."""
    conn = get_raw_sqlite_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT investigation_id, transaction_id, risk_score, evidence_json, fraud_hypothesis, recommended_action, summary_report, created_at
        FROM investigation_reports
        WHERE transaction_id = ?
        ORDER BY created_at DESC LIMIT 1
    """, (transaction_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"No investigation report found for transaction {transaction_id}.")

    return {
        "investigation_id": row[0],
        "transaction_id": row[1],
        "risk_score": row[2],
        "evidence": json.loads(row[3]),
        "fraud_hypothesis": row[4],
        "recommended_action": row[5],
        "summary_report": row[6],
        "created_at": row[7]
    }
