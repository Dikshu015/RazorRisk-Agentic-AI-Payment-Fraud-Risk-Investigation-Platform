import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Vercel's Python runtime, Hugging Face Spaces' Docker runtime, and
# Antideploy (built on Google Cloud Run) all deploy onto a filesystem that's
# either read-only outside /tmp or gets wiped on every cold start/redeploy.
# Vercel sets VERCEL=1, HF Spaces sets SPACE_ID, and Cloud Run itself sets
# K_SERVICE on every service unconditionally (it's how Cloud Run identifies
# itself, documented and unlikely to change) — any one of these flips this
# flag, redirecting the SQLite DB and log files to /tmp instead of losing
# data unpredictably or crashing on the first write. Render (and local dev)
# use a real writable, persistent process, so BASE_DIR is fine there.
IS_RESTRICTED_FS = bool(os.getenv("VERCEL") or os.getenv("SPACE_ID") or os.getenv("K_SERVICE"))
IS_SERVERLESS = IS_RESTRICTED_FS  # kept as an alias — existing call sites (api/main.py) use this name
RUNTIME_DIR = Path("/tmp") if IS_RESTRICTED_FS else BASE_DIR

# Settings
SQLITE_DB_PATH = RUNTIME_DIR / "razor_risk.db"
# SQLite-only — see db/database.py for why the earlier Postgres-via-
# DATABASE_URL path was removed rather than kept half-wired. This is now
# only used for the startup log line showing where the DB actually lives.
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

# Rolling window (days) for the amount_zscore_prior feature (both offline
# training in ml/train_tabular_model.py and live scoring in
# ml/risk_aggregator.py use this SAME constant) — a user's "normal" amount
# is judged against their trailing 90 days, not their entire lifetime.
# Keeping this in one place is what keeps train-time and live-scoring-time
# feature computation from silently drifting apart.
PRIOR_AMOUNT_WINDOW_DAYS = int(os.getenv("PRIOR_AMOUNT_WINDOW_DAYS", 90))
