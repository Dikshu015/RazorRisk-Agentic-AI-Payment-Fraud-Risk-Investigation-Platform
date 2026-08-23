"""
RazorRisk — investigation orchestrator.

Evidence gathering (the 4 deterministic tools) always runs the same way
regardless of mode — this is the "zero hallucination" guarantee: neither
path can produce a number that didn't come from GraphTool,
TransactionHistoryTool, DeviceRiskTool, or FraudModelTool.

What differs is who writes the fraud_hypothesis/recommended_action from
that evidence:
  - agent.llm_investigator, if an LLM API key is configured and the call
    succeeds — genuine LLM reasoning over the gathered evidence.
  - agent.deterministic_agent otherwise — rule-based pattern matching.

Every investigation_result carries an explicit "agent_mode" field
("llm:<provider>" or "deterministic_fallback") and the report itself
states which one ran, rather than letting a README claim about "LangGraph
agentic reasoning" imply more than the code that ran for THIS report
actually did.
"""
import uuid
import datetime

from agent.tools import GraphTool, TransactionHistoryTool, DeviceRiskTool, FraudModelTool
from agent.deterministic_agent import determine_fraud_hypothesis
from agent import llm_investigator, mode_state
from agent.prompts import REPORT_TEMPLATE
from utils.logger import get_logger

logger = get_logger("graph_agent")


class RiskInvestigationAgent:
    def investigate(self, txn_payload: dict, risk_summary: dict) -> dict:
        txn_id = txn_payload.get("transaction_id", f"TXN_{uuid.uuid4().hex[:8]}")
        user_id = txn_payload.get("user_id", "UNKNOWN_USER")
        device_id = txn_payload.get("device_id", "UNKNOWN_DEV")
        ip_address = txn_payload.get("ip_address", "0.0.0.0")
        amount = float(txn_payload.get("amount", 0.0))

        logger.info(f"========== Starting investigation for Txn: {txn_id} (User: {user_id}) ==========")

        # Step 1: deterministic evidence gathering — identical for both modes
        graph_evidence = GraphTool.run(user_id)
        history_evidence = TransactionHistoryTool.run(user_id)
        device_evidence = DeviceRiskTool.run(device_id, ip_address)
        model_evidence = FraudModelTool.run(txn_payload)
        evidence = {
            "graph_evidence": graph_evidence, "history_evidence": history_evidence,
            "device_evidence": device_evidence, "model_evidence": model_evidence,
        }
        logger.info(f"[Evidence] Shared devices: {graph_evidence['shared_device_account_count']}, "
                    f"shared IPs: {graph_evidence['shared_ip_account_count']}, "
                    f"proxy: {device_evidence['is_suspicious_proxy']}")

        # Step 2: hypothesis — try LLM if configured, fall back on any failure
        # `override` is the dashboard's "Agent mode" selector (agent/mode_state.py):
        # "auto" preserves the original priority-order behavior, a specific
        # provider name forces that provider (report generation still falls
        # back to deterministic if that provider's call fails), and
        # "deterministic" skips the LLM path entirely even if a key is set.
        override = mode_state.get_mode()
        agent_mode = "deterministic_fallback"
        agent_mode_label = "Deterministic rule-based fallback (no LLM API key configured)"
        hypothesis = rec_action = rationale = None

        if override == "deterministic":
            agent_mode_label = "Deterministic rule-based (agent mode manually forced)"
        elif llm_investigator.is_available(None if override == "auto" else override):
            try:
                provider, hypothesis, rec_action, rationale = llm_investigator.investigate_with_llm(
                    txn_payload, risk_summary, evidence, forced_provider=override
                )
                if(provider): agent_mode = f"llm:{provider.lower()}"
                agent_mode_label = f"LLM-generated investigation (via {provider})"
            except Exception as e:
                logger.warning(f"LLM investigation failed ({e}) — falling back to deterministic rules.")
        elif override != "auto":
            agent_mode_label = f"Deterministic rule-based fallback ({override} selected but its API key isn't configured)"

        if hypothesis is None:
            hypothesis, rec_action, rationale = determine_fraud_hypothesis(
                txn_payload, graph_evidence, history_evidence, device_evidence
            )

        # Step 3: build the report — same template regardless of mode
        graph_summary = (
            f"{graph_evidence['shared_device_account_count']} accounts linked to same device, "
            f"{graph_evidence['shared_ip_account_count']} accounts linked to same IP. "
            f"Community size: {graph_evidence['community_size']} users."
        )
        device_summary = (
            f"Device OS: {device_evidence['os']} ({device_evidence['device_type']}). "
            f"Proxy/VPN: {device_evidence['is_suspicious_proxy']}. "
            f"Location: {device_evidence['city']}, {device_evidence['country']} ({device_evidence['isp']})."
        )
        behavior_summary = (
            f"Current velocity: {txn_payload.get('velocity_1h', 1)} txns/hr "
            f"(historical: {history_evidence['total_historical_txns']} total txns, "
            f"avg amt: ₹{history_evidence['historical_avg_amount']:,.2f})."
        )

        summary_report = REPORT_TEMPLATE.format(
            agent_mode_label=agent_mode_label,
            transaction_id=txn_id, user_id=user_id, amount=amount,
            risk_score=risk_summary.get("risk_score", 85.0),
            risk_tier=risk_summary.get("risk_tier", "HIGH"),
            decision=risk_summary.get("decision", "HOLD_FOR_INVESTIGATION"),
            graph_evidence_summary=graph_summary,
            device_risk_summary=device_summary,
            behavior_summary=behavior_summary,
            tabular_score=risk_summary.get("tabular_score", 0.0),
            gnn_score=risk_summary.get("gnn_score", 0.0),
            fraud_hypothesis=hypothesis,
            recommended_action=rec_action,
            action_rationale=rationale,
        )

        investigation_result = {
            "investigation_id": f"INV_{uuid.uuid4().hex[:8]}",
            "transaction_id": txn_id,
            "user_id": user_id,
            "risk_score": risk_summary.get("risk_score", 85.0),
            "agent_mode": agent_mode,
            "agent_mode_label": agent_mode_label,
            "evidence": evidence,
            "fraud_hypothesis": hypothesis,
            "recommended_action": rec_action,
            "summary_report": summary_report,
            "created_at": datetime.datetime.now().isoformat(),
        }
        logger.info(f"========== Investigation complete for Txn: {txn_id} | Mode: {agent_mode} | Action: {rec_action} ==========")
        return investigation_result


investigation_agent = RiskInvestigationAgent()

if __name__ == "__main__":
    sample_payload = {
        "transaction_id": "TXN_TEST_999", "user_id": "USER_RING1_1",
        "device_id": "DEV_FRAUD_RING1", "ip_address": "185.220.101.44",
        "amount": 92000, "velocity_1h": 14,
    }
    sample_risk = {
        "risk_score": 94.5, "risk_tier": "CRITICAL", "decision": "BLOCK_AND_INVESTIGATE",
        "tabular_score": 88.0, "gnn_score": 98.0,
    }
    rep = investigation_agent.investigate(sample_payload, sample_risk)
    print(rep["summary_report"])
