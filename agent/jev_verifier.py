"""
RazorRisk — Jev (TypeSafe System One) verification pass.

Runs only when TYPESAFE_API_KEY is set AND the dashboard's Jev toggle is on
(agent/mode_state.py — off by default, same "opt-in, never block the report"
posture as the LLM providers in agent/llm_investigator.py). This is a
verification layer, not a decision-maker: it never writes the fraud
hypothesis or chooses the recommended_action for the report. It runs after
agent/llm_investigator.py (or agent/deterministic_agent.py) has already
produced one, and cross-checks it against the same deterministic evidence.

Two independent checks against the evidence dict from agent/graph_agent.py,
never fed the investigator's own answer as an input:

1. independent_action (Choice): Jev picks a recommended_action from the same
   five-option set the investigator uses, from evidence alone. If this
   disagrees with what the investigator (LLM or deterministic) actually
   picked, that's a second opinion the investigator never saw, at a few
   hundredths of a cent and well under a second — cheap enough to run on
   every investigation, unlike a second LLM call, which would double
   cost/latency for a check that's supposed to be the safety net under the
   LLM's own answer.

2. hypothesis_grounded (Noul): given the investigator's fraud_hypothesis
   TEXT plus the evidence, is every specific claim in it traceable to the
   evidence? agent/llm_investigator.py's system prompt already instructs the
   LLM not to invent numbers/IDs, but nothing upstream verifies that
   instruction was followed — this is that check. (On the deterministic path this should normally score high since agent/deterministic_agent.py
   only ever renders evidence fields verbatim; a low score there points at
   a template bug, not a hallucination.)

Disagreement or a low grounding score sets verification_flag to
"REVIEW_RECOMMENDED" instead of silently trusting either path's answer —
mirrors the MODEL_DISAGREEMENT HITL trigger already used for GNN vs. XGBoost
(ml/risk_aggregator.py), applied one layer up.

api/routes_agent.py uses a CONSISTENT verdict, above
JEV_AUTO_RESOLVE_MIN_CONFIDENCE, to auto-resolve an already-queued HITL
review instead of leaving it pending — but only for reviews that weren't
already flagged for one of ml.decision_policy.MANDATORY_HUMAN_REASONS
(HIGH_IMPACT, MODEL_DISAGREEMENT, EVIDENCE_CONFLICT, MODEL_UNCERTAINTY).
Those are compliance/ambiguity triggers, not confidence triggers, so no
amount of Jev+LLM agreement is allowed to bypass them — same principle
ml/decision_policy.py already applies to its own AUTO_BLOCK_THRESHOLD
override. This module only ever computes the verdict; it never touches the
HITL queue itself.

Raises on any failure (bad key, network, non-200, malformed response) —
caller (agent/graph_agent.py) catches and logs, and the investigation report
is still returned in full without a verification section. This is a bonus
signal, never a blocker.

NOTE: Jev is in early access as of Sep 2026 and TypeSafe's own docs mark the
raw API's request/response shape as not yet stable — expect to revisit this
file if a field name here changes upstream.
"""
import requests

from config import (
    TYPESAFE_API_KEY, TYPESAFE_MODEL, TYPESAFE_API_BASE, TYPESAFE_TIMEOUT_SECONDS,
    JEV_AUTO_RESOLVE_MIN_CONFIDENCE,
)
from utils.logger import get_logger

logger = get_logger("jev_verifier")

ACTION_CRITERIA = {
    "BLOCK_ACCOUNT_AND_HOLD_FUNDS": "Evidence shows an active, high-confidence fraud ring or carding pattern; freeze the account and hold funds immediately.",
    "HOLD_FOR_MANUAL_REVIEW": "Evidence is suspicious but not conclusive enough to act on unilaterally; a human analyst should look at it.",
    "TEMPORARY_VELOCITY_FREEZE": "The main signal is an abnormal transaction velocity spike rather than a confirmed fraud ring; freeze new transactions temporarily.",
    "REQUIRE_TWO_FACTOR_AUTHENTICATION": "Some risk signals are present, but a stronger identity check could resolve them without blocking the user outright.",
    "APPROVE_WITH_VERIFICATION": "Evidence overall looks like a false positive or low residual risk; approve after a light verification step.",
}

GROUNDING_THRESHOLD = 0.5


def is_available() -> bool:
    return bool(TYPESAFE_API_KEY)


def _call_systemone(state: str, questions: dict) -> dict:
    resp = requests.post(
        f"{TYPESAFE_API_BASE}/v1/systemone",
        headers={"Authorization": f"Bearer {TYPESAFE_API_KEY}", "Content-Type": "application/json"},
        json={"model": TYPESAFE_MODEL, "state": state, "questions": questions},
        timeout=TYPESAFE_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()


def verify_investigation(txn_payload: dict, risk_summary: dict, evidence: dict,
                          fraud_hypothesis: str, recommended_action: str) -> dict:
    """Returns a verification dict; raises on any failure (see module docstring)."""
    if not TYPESAFE_API_KEY:
        raise RuntimeError("TYPESAFE_API_KEY is not configured.")

    evidence_state = {
        "risk_scores": risk_summary,
        "graph_evidence": evidence["graph_evidence"],
        "history_evidence": evidence["history_evidence"],
        "device_evidence": evidence["device_evidence"],
        "model_evidence": evidence["model_evidence"],
    }

    action_resp = _call_systemone(
        state=str(evidence_state),
        questions={
            "recommended_action": {
                "type": "choice",
                "instructions": "Given this fraud-risk evidence for one transaction, which action best fits?",
                "criteria": ACTION_CRITERIA,
            }
        },
    )
    action_answer = action_resp["answers"]["recommended_action"]

    grounding_resp = _call_systemone(
        state=f"Evidence:
{evidence_state}

Investigator's fraud hypothesis:
{fraud_hypothesis}",
        questions={
            "hypothesis_grounded": {
                "type": "noul",
                "instructions": (
                    "Is every specific claim in the investigator's fraud hypothesis "
                    "(numbers, counts, device/IP/account details) traceable to the "
                    "evidence provided, with nothing invented?"
                ),
            }
        },
    )
    grounded_probability = grounding_resp["answers"]["hypothesis_grounded"]["noul"]

    actions_agree = action_answer["choice"] == recommended_action
    is_grounded = grounded_probability >= GROUNDING_THRESHOLD
    verification_flag = "CONSISTENT" if (actions_agree and is_grounded) else "REVIEW_RECOMMENDED"
    confidence = action_answer.get("confidence") or 0.0
    eligible_for_auto_resolve = verification_flag == "CONSISTENT" and confidence >= JEV_AUTO_RESOLVE_MIN_CONFIDENCE

    result = {
        "jev_model": action_resp.get("model", TYPESAFE_MODEL),
        "independent_action": action_answer["choice"],
        "independent_action_confidence": confidence,
        "independent_action_probabilities": action_answer.get("probabilities"),
        "actions_agree": actions_agree,
        "hypothesis_grounded_probability": grounded_probability,
        "hypothesis_grounded": is_grounded,
        "verification_flag": verification_flag,
        "eligible_for_auto_resolve": eligible_for_auto_resolve,
    }

    if verification_flag == "REVIEW_RECOMMENDED":
        logger.warning(
            f"[Jev] Flagged for review — investigator picked '{recommended_action}', "
            f"Jev independently picked '{action_answer['choice']}' (agree={actions_agree}); "
            f"hypothesis grounding={grounded_probability:.2f}."
        )
    else:
        logger.info(
            f"[Jev] Verification consistent (confidence={confidence:.2f}, "
            f"auto-resolve eligible={eligible_for_auto_resolve})."
        )

    return result
