"""
RazorRisk — runtime agent-mode override.

Holds a single in-memory value that lets the dashboard force which
investigation path runs next, without restarting the process or editing
.env: "auto" (default — provider priority order, same as before this
existed), a specific provider name to force that provider's LLM path,
or "deterministic" to force the rule-based fallback even if a key is
configured.

Deliberately process-local, in-memory, not persisted — this is an
operator toggle for demoing/debugging which path runs, not application
config. Restarting the server resets it to "auto".
"""

VALID_MODES = ("auto", "anthropic", "groq", "openai", "deterministic")

_current_mode = "auto"


def get_mode() -> str:
    return _current_mode


def set_mode(mode: str) -> str:
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid agent mode '{mode}'. Must be one of {VALID_MODES}.")
    global _current_mode
    _current_mode = mode
    return _current_mode
