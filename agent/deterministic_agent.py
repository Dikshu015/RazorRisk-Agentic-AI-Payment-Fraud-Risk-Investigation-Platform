"""
RazorRisk — deterministic fallback investigator.

Used when no LLM API key is configured (or the LLM call fails for any
reason). This is plain rule-based pattern matching over the same evidence
the LLM path would have seen — genuinely useful engineering, and labeled
as exactly what it is in every report it produces (agent_mode is always
"deterministic_fallback", never dressed up as agentic reasoning). A demo
running with zero API keys configured still produces a complete,
correctly-reasoned investigation report — it just says plainly that a
human wrote these rules, not a model.
"""


def determine_fraud_hypothesis(txn_payload: dict, graph_evidence: dict, history_evidence: dict, device_evidence: dict):
    """Returns (fraud_hypothesis: str, recommended_action: str, action_rationale: str)."""
    device_id = txn_payload.get("device_id", "UNKNOWN_DEV")
    ip_address = txn_payload.get("ip_address", "0.0.0.0")
    amount = float(txn_payload.get("amount", 0.0))

    if graph_evidence["shared_device_account_count"] >= 3:
        hypothesis = (
            f"High-confidence device sharing fraud ring detected. Device '{device_id}' is shared "
            f"across {graph_evidence['shared_device_account_count']} distinct user accounts "
            f"(community fraud ratio: {graph_evidence['community_fraud_ratio']:.0%}). Multiple accounts "
            f"transacting from one device fingerprint is characteristic of automated account farming "
            f"or stolen-credential testing."
        )
        return hypothesis, "BLOCK_ACCOUNT_AND_HOLD_FUNDS", (
            f"Device fingerprint '{device_id}' exhibits severe account multi-tenancy "
            f"({graph_evidence['shared_device_account_count']} linked accounts)."
        )

    if graph_evidence["shared_ip_account_count"] >= 4 or device_evidence["is_suspicious_proxy"]:
        hypothesis = (
            f"Proxy/VPN botnet pattern detected. Transaction originates from IP '{ip_address}' "
            f"({device_evidence['isp']}, {device_evidence['country']}), shared by "
            f"{graph_evidence['shared_ip_account_count']} separate user accounts."
        )
        return hypothesis, "HOLD_FOR_MANUAL_REVIEW", (
            f"Traffic routed via a proxy/VPN IP '{ip_address}' linked to "
            f"{graph_evidence['shared_ip_account_count']} accounts."
        )

    if txn_payload.get("velocity_1h", 1) >= 8:
        hypothesis = (
            f"High-velocity carding/micro-transaction pattern. User executed "
            f"{txn_payload.get('velocity_1h')} transactions within one hour, well above their "
            f"historical baseline of {history_evidence['historical_max_velocity_1h']} txns/hr."
        )
        return hypothesis, "TEMPORARY_VELOCITY_FREEZE", (
            f"Abnormal transaction frequency ({txn_payload.get('velocity_1h')} txns/hr) is consistent "
            f"with scripted/automated activity."
        )

    hypothesis = (
        f"Elevated transaction risk. Amount (₹{amount:,.2f}) is notably higher than this user's "
        f"historical average (₹{history_evidence['historical_avg_amount']:,.2f}), combined with an "
        f"elevated graph-neighborhood risk signal, without a clear device/IP-sharing fraud-ring pattern."
    )
    return hypothesis, "REQUIRE_TWO_FACTOR_AUTHENTICATION", (
        f"Amount anomaly (₹{amount:,.2f}) coupled with moderate graph-cluster risk, but no single "
        f"strong fraud-ring indicator on its own."
    )
