import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Settings
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'razor_risk.db'}")
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

APP_LOG_PATH = LOG_DIR / "app.log"
RISK_ENGINE_LOG_PATH = LOG_DIR / "risk_engine.log"
AGENT_LOG_PATH = LOG_DIR / "agent_investigations.log"
ML_TRAINING_LOG_PATH = LOG_DIR / "ml_training.log"
GRAPH_LOG_PATH = LOG_DIR / "graph.log"
DATABASE_LOG_PATH = LOG_DIR / "database.log"
PIPELINE_LOG_PATH = LOG_DIR / "pipeline.log"
FRONTEND_LOG_PATH = LOG_DIR / "frontend_client.log"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") # reserved, not currently used
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
