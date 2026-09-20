import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

IS_RESTRICTED_FS = bool(os.getenv("VERCEL") or os.getenv("SPACE_ID") or os.getenv("K_SERVICE"))
IS_SERVERLESS = IS_RESTRICTED_FS
RUNTIME_DIR = Path("/tmp") if IS_RESTRICTED_FS else BASE_DIR

SQLITE_DB_PATH = RUNTIME_DIR / "razor_risk.db"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgrespassword@localhost:5432/razor_risk")
DATABASE_BACKEND = "postgresql" if DATABASE_URL.startswith(("postgresql://", "postgres://", "postgresql+")) else "sqlite"
DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", "2"))
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", "10"))
DB_CONNECT_TIMEOUT_SECONDS = int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "5"))
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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")
TYPESAFE_API_BASE = os.getenv("TYPESAFE_API_BASE", "https://api.typesafe.ai")
TYPESAFE_MODEL = os.getenv("TYPESAFE_MODEL", "jev-latest")
TYPESAFE_TIMEOUT_SECONDS = float(os.getenv("TYPESAFE_TIMEOUT_SECONDS", "5"))
JEV_AUTO_RESOLVE_MIN_CONFIDENCE = float(os.getenv("JEV_AUTO_RESOLVE_MIN_CONFIDENCE", "0.85"))

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
API_KEY = os.getenv("API_KEY", "")
HIGH_RISK_THRESHOLD = 70.0
WATCHLIST_TTL_HOURS = int(os.getenv("WATCHLIST_TTL_HOURS", 24))
WATCHLIST_SCORE_MULTIPLIER = float(os.getenv("WATCHLIST_SCORE_MULTIPLIER", 1.2))
PRIOR_AMOUNT_WINDOW_DAYS = int(os.getenv("PRIOR_AMOUNT_WINDOW_DAYS", 90))
