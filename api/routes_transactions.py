import json
import uuid
import datetime
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
from ml.risk_aggregator import calculate_composite_risk_score
from ml.graph_builder import graph_builder
from db.database import get_raw_sqlite_connection
from utils.logger import get_logger, bind_correlation_id, clear_correlation_id

logger = get_logger("api_transactions")

router = APIRouter(prefix="/api/v1/transactions", tags=["Transactions & Risk Scoring"])

class TransactionPayload(BaseModel):
    transaction_id: Optional[str] = None
    user_id: str = Field(..., example="USER_0012")
    device_id: str = Field(..., example="DEV_0045")
    ip_address: str = Field(..., example="192.168.1.100")
    merchant_id: str = Field(default="MCH_001", example="MCH_001")
    amount: float = Field(..., example=45000.0)
    currency: str = Field(default="INR")
    velocity_1h: int = Field(default=1, example=3)
    amount_zscore_prior: float = Field(default=0.0)
    is_vpn_proxy: bool = Field(default=False)
    is_suspicious_proxy: bool = Field(default=False)

@router.post("/score")
def score_transaction(payload: TransactionPayload):
    """
    Evaluates an incoming payment transaction in real-time.
    Computes Tabular ML fraud probability, GNN node embedding score, and graph topology risk.
    Triggers LangGraph Investigation Agent if Risk Score >= 70.0.
    """
    txn_dict = payload.model_dump()
    txn_id = txn_dict.get("transaction_id") or f"TXN_{uuid.uuid4().hex[:8].upper()}"
    txn_dict["transaction_id"] = txn_id

    # Every log line emitted while scoring this transaction — across the
    # tabular model, GNN, aggregator, and (for high-risk ones) the agent —
    # carries this same ID, so `grep <corr_id> logs/*.log` reconstructs the
    # full cross-subsystem trace of one transaction's decision.
    corr_id = bind_correlation_id()
    logger.info(f"Received live transaction scoring request: TxnID:{txn_id}, User:{payload.user_id}, Amt:₹{payload.amount}")

    try:
        # 1. Compute Composite Risk Score
        risk_res = calculate_composite_risk_score(txn_dict)

        # 2. High-risk transactions (>= 70) need an investigation, but that
        # can involve a real LLM call and take several seconds — it's run
        # as a separate follow-up request (POST /api/v1/investigations/run/{id})
        # from the frontend so the risk score itself renders instantly instead
        # of the whole scoring UI blocking on the slower agent step.
        needs_investigation = risk_res["risk_score"] >= 70.0

        # 3. Persist to Database
        conn = get_raw_sqlite_connection()
        cursor = conn.cursor()

        # Ensure Entities exist
        cursor.execute("INSERT OR IGNORE INTO users (user_id, name, email) VALUES (?, ?, ?)",
                       (payload.user_id, f"User {payload.user_id}", f"{payload.user_id}@example.com"))
        cursor.execute("INSERT OR IGNORE INTO devices (device_id, device_type, os, is_vpn_proxy) VALUES (?, ?, ?, ?)",
                       (payload.device_id, "Mobile-Android", "Android 14", payload.is_vpn_proxy))
        cursor.execute("INSERT OR IGNORE INTO ip_addresses (ip_address, country, city, isp, is_suspicious_proxy) VALUES (?, ?, ?, ?, ?)",
                       (payload.ip_address, "IN", "Mumbai", "Airtel", payload.is_suspicious_proxy))

        # Insert Transaction
        cursor.execute("""
            INSERT OR REPLACE INTO transactions
            (transaction_id, user_id, device_id, ip_address, merchant_id, amount, currency, timestamp, status, velocity_1h, amount_zscore_prior)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            txn_id, payload.user_id, payload.device_id, payload.ip_address, payload.merchant_id,
            payload.amount, payload.currency, datetime.datetime.now().isoformat(), "COMPLETED",
            payload.velocity_1h, payload.amount_zscore_prior
        ))

        # Insert Risk Score
        scoring_id = f"SCORE_{uuid.uuid4().hex[:8].upper()}"
        cursor.execute("""
            INSERT INTO risk_scores
            (scoring_id, transaction_id, risk_score, tabular_score, gnn_score, velocity_multiplier, risk_tier, decision)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scoring_id, txn_id, risk_res["risk_score"], risk_res["tabular_score"],
            risk_res["gnn_score"], risk_res["velocity_multiplier"], risk_res["risk_tier"], risk_res["decision"]
        ))

        conn.commit()
        conn.close()

        # 4. Fold this transaction into the live in-memory graph immediately
        # (O(1) incremental update) so the Graph Topology tab reflects it
        # without waiting for a full pipeline rebuild.
        graph_builder.add_transaction(
            user_id=payload.user_id, device_id=payload.device_id, ip_address=payload.ip_address,
            merchant_id=payload.merchant_id, amount=payload.amount,
            is_fraud=risk_res["risk_score"] >= 70.0
        )

        return {
            "transaction_id": txn_id,
            "user_id": payload.user_id,
            "amount": payload.amount,
            "risk_evaluation": risk_res,
            "needs_investigation": needs_investigation,
            "correlation_id": corr_id
        }
    except Exception as e:
        # Full detail goes to the audit log for debugging; the API response
        # stays generic on purpose — internal exception text (module paths,
        # dict keys, stack detail) should never reach the dashboard directly.
        logger.error(f"Scoring failed for TxnID:{txn_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transaction scoring failed (correlation_id={corr_id}). Check logs/risk_engine.log for details.")
    finally:
        clear_correlation_id()

@router.get("/recent")
def get_recent_transactions(limit: int = 20, tier: Optional[str] = None):
    """Retrieves recent transactions with optional filtering by risk tier."""
    conn = get_raw_sqlite_connection()
    cursor = conn.cursor()

    query = """
        SELECT t.transaction_id, t.user_id, t.device_id, t.ip_address, t.merchant_id, t.amount, t.timestamp,
               r.risk_score, r.risk_tier, r.decision
        FROM transactions t
        LEFT JOIN risk_scores r ON t.transaction_id = r.transaction_id
    """
    params = []
    if tier:
        query += " WHERE r.risk_tier = ?"
        params.append(tier.upper())
    
    query += " ORDER BY t.timestamp DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    result = []
    for r in rows:
        result.append({
            "transaction_id": r[0],
            "user_id": r[1],
            "device_id": r[2],
            "ip_address": r[3],
            "merchant_id": r[4],
            "amount": r[5],
            "timestamp": r[6],
            "risk_score": r[7] if r[7] is not None else 0.0,
            "risk_tier": r[8] or "UNSCORED",
            "decision": r[9] or "NONE"
        })

    return {"count": len(result), "transactions": result}
