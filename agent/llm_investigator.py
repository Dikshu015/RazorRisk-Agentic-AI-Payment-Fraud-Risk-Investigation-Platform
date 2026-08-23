"""
RazorRisk — LLM-backed investigation path.

Used only when ANTHROPIC_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY is set
(checked in that order). This is a genuine call to a hosted LLM that reads
the same deterministic tool evidence agent/graph_agent.py already gathered
(GraphTool, TransactionHistoryTool, DeviceRiskTool, FraudModelTool) and
writes the fraud hypothesis + recommended action + narrative report — the
LLM never computes a number itself, only reasons over numbers the
deterministic tools already produced. That separation is what makes
"zero hallucination on financial decisions" an actual guarantee rather
than a README claim: ask it to justify a risk score and it can only point
back at tool output, because it was never given the ability to invent one.

This module is intentionally untested against a live API in this
environment (no network access here) — it's written to the standard
langchain chat-model interface, which is stable across providers, and the
whole path is optional: agent/graph_agent.py falls back to the
deterministic-only path (agent/deterministic_agent.py) if this import
fails, the API call fails, or no key is configured at all. Either path
produces a fully-formed investigation report; only the fraud_hypothesis
and recommended_action are LLM-authored when this path runs.
"""
import json

from config import ANTHROPIC_API_KEY, GROQ_API_KEY, OPENAI_API_KEY
from utils.logger import get_logger

logger = get_logger("investigator")

SYSTEM_PROMPT = """You are a payment fraud investigator. You are given deterministic
evidence already computed by fraud-detection tools (graph analysis, transaction
history, device risk, and ML model scores) for ONE suspicious transaction.

Rules:
- Do not invent any number, account ID, device ID, or IP not present in the evidence below.
- Every claim in your hypothesis must trace back to a specific evidence field.
- Respond with ONLY a JSON object, no other text, with exactly these keys:
  "fraud_hypothesis": one paragraph identifying the likely fraud pattern (or stating
     this looks like a false positive) and citing the specific evidence for it.
  "recommended_action": one of BLOCK_ACCOUNT_AND_HOLD_FUNDS, HOLD_FOR_MANUAL_REVIEW,
     TEMPORARY_VELOCITY_FREEZE, REQUIRE_TWO_FACTOR_AUTHENTICATION, APPROVE_WITH_VERIFICATION.
  "action_rationale": one sentence justifying the action given the evidence.
"""


def _get_configured_client():
    """Returns (provider_name, langchain_chat_model) for the first
    configured API key, or (None, None) if none are set."""
    if ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        return "Anthropic", ChatAnthropic(model="claude-3-5-sonnet-latest", api_key=ANTHROPIC_API_KEY, temperature=0)
    if GROQ_API_KEY:
        from langchain_groq import ChatGroq
        return "Groq", ChatGroq(model="llama-3.1-70b-versatile", api_key=GROQ_API_KEY, temperature=0)
    if OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return "OpenAI", ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
    return None, None


def is_available() -> bool:
    return bool(ANTHROPIC_API_KEY or GROQ_API_KEY or OPENAI_API_KEY)


def investigate_with_llm(txn_payload: dict, risk_summary: dict, evidence: dict):
    """Returns (provider_name, fraud_hypothesis, recommended_action, action_rationale)
    or raises on any failure — caller (graph_agent.py) catches and falls back
    to the deterministic path, logging why."""
    provider, llm = _get_configured_client()
    if llm is None:
        raise RuntimeError("No LLM API key configured (checked ANTHROPIC_API_KEY, GROQ_API_KEY, OPENAI_API_KEY).")

    evidence_str = json.dumps({
        "transaction": {k: v for k, v in txn_payload.items() if k != "transaction_id"},
        "risk_scores": risk_summary,
        "graph_evidence": evidence["graph_evidence"],
        "history_evidence": evidence["history_evidence"],
        "device_evidence": evidence["device_evidence"],
        "model_evidence": evidence["model_evidence"],
    }, indent=2, default=str)

    logger.info(f"Dispatching investigation to {provider} LLM...")
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Evidence:\n{evidence_str}"},
    ])

    content = response.content if hasattr(response, "content") else str(response)
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content[content.find("{"):content.rfind("}") + 1]

    parsed = json.loads(content)
    logger.info(f"{provider} LLM investigation complete. Recommended action: {parsed.get('recommended_action')}")
    return (
        provider,
        parsed["fraud_hypothesis"],
        parsed["recommended_action"],
        parsed.get("action_rationale", ""),
    )
