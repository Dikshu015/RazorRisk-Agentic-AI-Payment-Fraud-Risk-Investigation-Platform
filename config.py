import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Vercel's Python runtime and Hugging Face Spaces' Docker runtime both
# deploy the project onto a read-only filesystem outside /tmp, and neither
# persists /tmp across a redeploy/restart. Vercel sets VERCEL=1 and HF
# Spaces sets SPACE_ID automatically in every container's environment, so
# either one flips this flag — redirecting the SQLite DB and log files to
# /tmp instead of crashing on the first write. Render (and local dev) use a
# real writable, persistent process, so BASE_DIR is fine there.
IS_RESTRICTED_FS = bool(os.getenv("VERCEL") or os.getenv("SPACE_ID"))
IS_SERVERLESS = IS_RESTRICTED_FS  # kept as an alias — existing call sites (api/main.py) use this name
RUNTIME_DIR = Path("/tmp") if IS_RESTRICTED_FS else BASE_DIR

# Settings
SQLITE_DB_PATH = RUNTIME_DIR / "razor_risk.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{SQLITE_DB_PATH}")
LOG_DIR = RUNTIME_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

APP_LOG_PATH = LOG_DIR / "app.log"
RISK_ENGINE_LOG_PATH = LOG_DIR / "risk_engine.log"
AGENT_LOG_PATH = LOG_DIR / "agent_investigations.log"
ML_TRAINING_LOG_PATH = LOG_DIR / "ml_training.log"
GRAPH_LOG_PATH = LOG_DIR / "graph.log"
DATABASE_LOG_PATH = LOG_DIR / "database.log"
PIPELINE_LOG_PATH = LOG_DIR / "pipeline.log"
FRONTEND_LOG_PATH = LOG_DIR / "frontend_client.log"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # reserved, not currently used
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Per-provider model overrides (agent/llm_investigator.py). Defaults are
# current, non-deprecated production models as of Aug 2026 — Groq retired
# llama-3.1/3.3-versatile from its free/developer tier in June 2026, so
# openai/gpt-oss-120b (Groq's recommended migration target) is the default
# rather than the older Llama name.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

# Risk threshold triggering agent investigation
HIGH_RISK_THRESHOLD = 70.0
