
"""Post-ML decision policy.

ML produces risk evidence. This module decides whether the transaction can be
handled automatically, needs additional evidence, or must enter human review.
It deliberately does not turn graph connectivity into a fraud verdict.
"""
from __future__ import annotations

from typing import Any
from security.evidence_api import call_guardrailed_evidence
from utils.logger import get_logger

logger = get_logger("decision_policy")

HITL_REASONS = {
    "MODEL_UNCERTAINTY",
    "MODEL_DISAGREEMENT",
    "HIGH_IMPACT",
    "EVIDENCE_CONFLICT",
    "NOVEL_BEHAVIOR",
}


def apply_decision_policy(txn: dict[str, Any], risk: dict[str, Any]) -> dict[str, Any]:
    tab = float(risk["tabular_score"]) / 100.0
    gnn = float(risk["gnn_score"]) / 100.0
    stack = float(risk["stacker_calibrated_score"]) / 100.0
    amount = float(txn.get("amount", 0))
    # Velocity is always part of the scoring/policy path. The frontend toggle
    # only selects its source: CLIENT (simulation/trust mode) or BACKEND
    # (calculated from transaction history).
    velocity = int(risk.get("effective_velocity_1h", risk.get("velocity_1h", 1)))
    graph = risk.get("graph_evidence", {})

    reasons: list[str] = []
    evidence: dict[str, Any] = {}

    # These are policy triggers, not fraud labels.
    if 0.35 <= stack <= 0.65:
        reasons.append("MODEL_UNCERTAINTY")
    if abs(tab - gnn) >= 0.45:
        reasons.append("MODEL_DISAGREEMENT")
    if amount >= 50000:
        reasons.append("HIGH_IMPACT")
    if (
        float(graph.get("shared_ip_accounts", 1)) >= 5
        and float(graph.get("shared_device_accounts", 1)) <= 2
        and stack < 0.70
    ):
        # Strong shared-IP evidence with otherwise modest model risk is
        # deliberately treated as a conflict, not as fraud.
        reasons.append("EVIDENCE_CONFLICT")
    if velocity >= 10 and stack < 0.55:
        reasons.append("NOVEL_BEHAVIOR")

    # Call external-style evidence only for transactions where it can resolve
    # ambiguity or materially improve an investigation. Every call passes
    # through the allowlist + input validation + rate limiter.
    should_enrich = (
        risk["risk_tier"] in {"MEDIUM", "HIGH", "CRITICAL"}
        or bool(reasons)
        or bool(txn.get("is_vpn_proxy") or txn.get("is_suspicious_proxy"))
    )

    if should_enrich:
        evidence["device"] = call_guardrailed_evidence(
            "device_intelligence",
            transaction_id=txn["transaction_id"],
            user_id=txn["user_id"],
            device_id=txn.get("device_id"),
        )
        evidence["network"] = call_guardrailed_evidence(
            "network_intelligence",
            transaction_id=txn["transaction_id"],
            user_id=txn["user_id"],
            ip_address=txn.get("ip_address"),
        )
        evidence["merchant"] = call_guardrailed_evidence(
            "merchant_intelligence",
            transaction_id=txn["transaction_id"],
            user_id=txn["user_id"],
            merchant_id=txn.get("merchant_id"),
        )

        # A provider saying "shared by many accounts" is not itself a HITL
        # trigger. It becomes useful when it conflicts with ML evidence.
        if (
            evidence["device"].get("device_account_count", 0) >= 3
            and stack < 0.45
        ):
            if "EVIDENCE_CONFLICT" not in reasons:
                reasons.append("EVIDENCE_CONFLICT")

    hitl_required = bool(reasons) and (
        risk["risk_tier"] in {"MEDIUM", "HIGH", "CRITICAL"} or
        "HIGH_IMPACT" in reasons or
        "MODEL_DISAGREEMENT" in reasons
    )

    if hitl_required:
        final_decision = "HUMAN_REVIEW"
    elif risk["risk_tier"] == "CRITICAL":
        final_decision = "BLOCK_PENDING_REVIEW"
    elif risk["risk_tier"] == "HIGH":
        final_decision = "HOLD_FOR_INVESTIGATION"
    elif risk["risk_tier"] == "MEDIUM":
        final_decision = "MONITOR"
    else:
        final_decision = "APPROVE"

    return {
        "decision": final_decision,
        "hitl_required": hitl_required,
        "review_reasons": reasons,
        "external_evidence": evidence,
        "policy_version": "v2.0-confluence-hitl",
    }
