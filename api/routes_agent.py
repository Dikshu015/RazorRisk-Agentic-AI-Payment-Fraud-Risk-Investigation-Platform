import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agent.graph_agent import investigation_agent
from agent import llm_investigator, mode_state
from ml.risk_aggregator import calculate_composite_risk_score
from db.database import get_raw_sqlite_connection
from utils.logger import get_logger

logger = get_logger("api_agent")

router = APIRouter(prefix="/api/v1/investigations", tags=["Agentic Investigations"])

_PROVIDER_LABELS = {"anthropic": "Anthropic", "groq": "Groq", "openai": "OpenAI"}


class AgentModeRequest(BaseModel):
    mode: str  # "auto" | "anthropic" | "groq" | "openai" | "deterministic"


@router.get("/agent-status")
def get_agent_status():
    """Tells the dashboard which providers actually have an API key
    configured, which one is currently active (for the given mode
    selection), and what the mode selector should offer/disable."""
    configured = llm_investigator.configured_providers()
    override = mode_state.get_mode()

    if override == "deterministic":
        active_provider, active_label = None, "Deterministic rule-based (manually forced)"
    elif override == "auto":
        active_provider = configured[0] if configured else None
        active_label = (f"{_PROVIDER_LABELS[active_provider]} (auto)" if active_provider
                         else "Deterministic rule-based fallback (no LLM API key configured)")
    else:
        if override in configured:
            active_provider, active_label = override, f"{_PROVIDER_LABELS[override]} (forced)"
        else:
            active_provider = None
            active_label = f"Deterministic rule-based fallback ({_PROVIDER_LABELS.get(override, override)} selected but its API key isn't configured)"

    modes = [{"value": "auto", "label": "Auto (priority order)", "available": True},
              {"value": "deterministic", "label": "Deterministic only", "available": True}]
    for p in ("anthropic", "groq", "openai"):
        modes.insert(-1, {"value": p, "label": _PROVIDER_LABELS[p], "available": p in configured})

    return {
        "configured_providers": configured,
        "current_mode": override,
        "active_provider": active_provider,
        "active_label": active_label,
        "modes": modes,
    }


@router.post("/agent-mode")
def set_agent_status(req: AgentModeRequest):
    """Lets the dashboard force which investigation path the agent takes
    on the next run(s). This is a process-local, in-memory toggle — see
    agent/mode_state.py — it does not persist across a server restart."""
    try:
        new_mode = mode_state.set_mode(req.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    logger.info(f"Agent mode override set to '{new_mode}'.")
    return get_agent_status()

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
