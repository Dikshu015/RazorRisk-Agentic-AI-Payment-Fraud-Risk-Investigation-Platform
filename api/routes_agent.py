import json
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from agent.graph_agent import investigation_agent
from agent import llm_investigator, mode_state
from ml.risk_aggregator import calculate_composite_risk_score, HIGH_RISK_THRESHOLD
from ml.decision_policy import apply_decision_policy
from db.database import get_raw_sqlite_connection
from utils.logger import get_logger
from infra.rate_limit import enforce_rate_limit
from infra.jobs import enqueue_investigation, get_job
from infra.redis_client import REDIS_REQUIRED
from infra.worker import execute_job_sync

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
async def run_investigation(request: Request, transaction_id: str, force: bool = False):
    """Triggers the investigation agent (deterministic and/or LLM) on a
    specific transaction ID.

    Server-side necessity guard: this used to run unconditionally for
    ANY transaction_id passed in, trusting the dashboard frontend to only
    call this endpoint for transactions that actually crossed the
    HIGH_RISK_THRESHOLD / hitl_required bar (routes_transactions.py computes
    that same condition as `needs_investigation` and only then shows the
    dashboard's "Investigate" button). Nothing enforced that server-side —
    a direct API call (or a bug in a future frontend change) could invoke a
    full investigation, including a real LLM call, for every low-risk
    transaction, which is both needless cost/latency and exactly the
    "checking every transaction this heavily" the LLM step is meant to
    avoid. Recomputes the same risk_score/hitl_required check here and
    refuses (with the numbers that justify the refusal) unless the caller
    explicitly passes ?force=true — an analyst manually pulling up a report
    for a low-risk transaction out of curiosity is still one query param
    away, but it's now an explicit choice rather than the unenforced
    default."""
    await enforce_rate_limit(request, scope="investigation-sync", limit=30)

    conn = get_raw_sqlite_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT t.user_id, t.device_id, t.ip_address, t.merchant_id, t.amount,
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
        "is_vpn_proxy": bool(row[5]),
        "is_suspicious_proxy": bool(row[6])
    }

    risk_res = calculate_composite_risk_score(txn_payload)
    policy_res = apply_decision_policy(txn_payload, risk_res)
    needs_investigation = risk_res["risk_score"] >= HIGH_RISK_THRESHOLD or policy_res.get("hitl_required", False)

    if not needs_investigation and not force:
        conn.close()
        logger.info(
            f"Investigation skipped for {transaction_id}: risk_score={risk_res['risk_score']} "
            f"< {HIGH_RISK_THRESHOLD} and no HITL trigger — not necessary. Pass ?force=true to run anyway."
        )
        return {
            "transaction_id": transaction_id,
            "investigation_skipped": True,
            "reason": (
                f"risk_score {risk_res['risk_score']} is below the {HIGH_RISK_THRESHOLD} investigation "
                f"threshold and no HITL review reason was triggered — an LLM/agent investigation isn't "
                f"warranted for this transaction. Call again with ?force=true to run one anyway."
            ),
            "risk_evaluation": risk_res,
            "policy_evaluation": policy_res,
        }

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


@router.post("/enqueue/{transaction_id}", status_code=202)
async def enqueue_investigation_job(request: Request, transaction_id: str, force: bool = False):
    """Queue an investigation in the shared Redis queue.

    Returns immediately with a durable job ID. Any API replica may accept the
    request; any worker replica may execute it.
    """
    await enforce_rate_limit(request, scope="investigation-enqueue", limit=60)
    # Validate existence before enqueueing so clients get a deterministic 404.
    conn = get_raw_sqlite_connection()
    try:
        exists = conn.execute("SELECT 1 FROM transactions WHERE transaction_id = ?", (transaction_id,)).fetchone()
    finally:
        conn.close()
    if not exists:
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found.")
    try:
        job = await enqueue_investigation(transaction_id, force=force)
    except Exception as exc:
        if REDIS_REQUIRED:
            logger.exception("Failed to enqueue investigation for %s", transaction_id)
            raise HTTPException(status_code=503, detail="Investigation queue unavailable") from exc
        # REDIS_REQUIRED=false means this deployment never promised a durable
        # distributed queue (local dev, a zero-infra demo). Rather than
        # leaving the investigation feature entirely broken when Redis isn't
        # running, execute the exact same job logic the worker would have
        # run (infra.worker.execute_job_sync) synchronously, inline in this
        # request. This is explicitly a degraded, non-distributed path --
        # "degraded_mode" in the response says so rather than silently
        # pretending the Redis queue handled it -- and REDIS_REQUIRED=true
        # in production still fails closed above, unchanged.
        logger.warning(
            "Redis unavailable and REDIS_REQUIRED=false; running investigation "
            "for %s synchronously in-process instead of queuing it.", transaction_id
        )
        try:
            result = execute_job_sync({"transaction_id": transaction_id, "force": force})
            return {
                "job_id": None, "transaction_id": transaction_id,
                "status": "completed", "result": result,
                "degraded_mode": "synchronous_no_redis",
            }
        except Exception as exec_exc:
            logger.exception("Synchronous fallback investigation failed for %s", transaction_id)
            return {
                "job_id": None, "transaction_id": transaction_id,
                "status": "failed", "error": str(exec_exc),
                "degraded_mode": "synchronous_no_redis",
            }
    return {
        "job_id": job["job_id"],
        "transaction_id": transaction_id,
        "status": job["status"],
        "sla_deadline": job["sla_deadline"],
        "queue": "redis",
    }


@router.get("/jobs/{job_id}")
async def investigation_job_status(job_id: str):
    """Return durable state/result for a queued investigation job."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Investigation job {job_id} not found or expired.")
    return job
