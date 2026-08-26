
import json
import uuid
import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from db.database import get_raw_sqlite_connection
from utils.logger import get_logger

router = APIRouter(prefix="/api/v1/hitl", tags=["Human-in-the-Loop"])
logger = get_logger("hitl")

class ReviewDecision(BaseModel):
    decision: str = Field(..., pattern="^(APPROVE|HOLD|BLOCK)$")
    reviewer: str = Field(default="human-reviewer", min_length=1, max_length=80)
    rationale: str = Field(..., min_length=3, max_length=2000)

def enqueue_review(transaction_id: str, risk: dict):
    if not risk.get("hitl_required"):
        return None
    conn = get_raw_sqlite_connection()

    # Idempotent queueing: rescoring a transaction must not create duplicate
    # human-review work items.
    existing = conn.execute(
        "SELECT review_id FROM human_reviews WHERE transaction_id = ? AND status = 'PENDING' "
        "ORDER BY created_at DESC LIMIT 1",
        (transaction_id,),
    ).fetchone()
    if existing:
        conn.close()
        return existing[0]

    review_id = f"REV_{uuid.uuid4().hex[:10].upper()}"
    conn.execute(
        """INSERT OR REPLACE INTO human_reviews
        (review_id, transaction_id, status, risk_score, reasons_json, evidence_json, created_at)
        VALUES (?, ?, 'PENDING', ?, ?, ?, ?)""",
        (
            review_id,
            transaction_id,
            risk["risk_score"],
            json.dumps(risk.get("review_reasons", [])),
            json.dumps(risk.get("external_evidence", {})),
            datetime.datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    logger.info("HITL review queued: %s txn=%s", review_id, transaction_id)
    return review_id

@router.get("/queue")
def get_review_queue(limit: int = 50):
    limit = max(1, min(limit, 100))
    conn = get_raw_sqlite_connection()
    rows = conn.execute(
        """SELECT review_id, transaction_id, status, risk_score, reasons_json,
                  evidence_json, created_at, reviewer, reviewer_decision,
                  reviewer_rationale, reviewed_at
           FROM human_reviews
           WHERE status = 'PENDING'
           ORDER BY created_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return {
        "count": len(rows),
        "reviews": [
            {
                "review_id": r[0], "transaction_id": r[1], "status": r[2],
                "risk_score": r[3], "reasons": json.loads(r[4] or "[]"),
                "evidence": json.loads(r[5] or "{}"), "created_at": r[6],
                "reviewer": r[7], "decision": r[8], "rationale": r[9],
                "reviewed_at": r[10],
            } for r in rows
        ],
    }

@router.post("/review/{review_id}")
def review_transaction(review_id: str, body: ReviewDecision):
    conn = get_raw_sqlite_connection()
    row = conn.execute(
        "SELECT transaction_id, status FROM human_reviews WHERE review_id = ?",
        (review_id,),
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Review not found")
    if row[1] != "PENDING":
        conn.close()
        raise HTTPException(status_code=409, detail="Review is already resolved")

    now = datetime.datetime.now().isoformat()
    conn.execute(
        """UPDATE human_reviews
           SET status='RESOLVED', reviewer=?, reviewer_decision=?,
               reviewer_rationale=?, reviewed_at=?
           WHERE review_id=?""",
        (body.reviewer, body.decision, body.rationale, now, review_id),
    )
    conn.execute(
        "UPDATE risk_scores SET decision=? WHERE transaction_id=?",
        (body.decision, row[0]),
    )
    conn.commit()
    conn.close()
    logger.info("HITL resolved: review=%s txn=%s decision=%s", review_id, row[0], body.decision)
    return {"review_id": review_id, "transaction_id": row[0], "status": "RESOLVED", "decision": body.decision}


@router.get("/transaction/{transaction_id}")
def get_review_for_transaction(transaction_id: str):
    """Return the latest human-review state for a transaction."""
    conn = get_raw_sqlite_connection()
    row = conn.execute(
        """SELECT review_id, transaction_id, status, risk_score, reasons_json,
                  evidence_json, created_at, reviewer, reviewer_decision,
                  reviewer_rationale, reviewed_at
           FROM human_reviews
           WHERE transaction_id = ?
           ORDER BY created_at DESC LIMIT 1""",
        (transaction_id,),
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="No human review exists for this transaction")
    return {
        "review_id": row[0], "transaction_id": row[1], "status": row[2],
        "risk_score": row[3], "reasons": json.loads(row[4] or "[]"),
        "evidence": json.loads(row[5] or "{}"), "created_at": row[6],
        "reviewer": row[7], "decision": row[8], "rationale": row[9],
        "reviewed_at": row[10],
    }
