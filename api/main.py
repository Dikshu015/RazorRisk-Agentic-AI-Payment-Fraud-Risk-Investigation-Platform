from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from config import BASE_DIR, IS_SERVERLESS, SQLITE_DB_PATH
from db.database import init_db, get_raw_sqlite_connection
from ml.graph_builder import graph_builder
from utils.logger import get_logger

from api.routes_transactions import router as transactions_router
from api.routes_graph import router as graph_router
from api.routes_agent import router as agent_router
from api.routes_logs import router as logs_router
from api.routes_admin import router as admin_router

logger = get_logger("api_main")

app = FastAPI(
    title="RazorRisk — Agentic AI Payment Fraud & Risk Investigation Platform",
    description="Graph Neural Network + Tabular ML + LangGraph Agentic Investigation Engine",
    version="1.0.0"
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(transactions_router)
app.include_router(graph_router)
app.include_router(agent_router)
app.include_router(logs_router)
app.include_router(admin_router)

# Mount Static Files for Dashboard
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/dashboard", StaticFiles(directory=str(static_dir), html=True), name="static")

@app.on_event("startup")
def on_startup():
    logger.info("Starting RazorRisk FastAPI Backend Engine...")
    init_db()

    # On Vercel or Hugging Face Spaces, DATABASE_URL points at /tmp (see
    # config.py) — a fresh,
    # empty file on every cold start, since /tmp doesn't persist between
    # invocations. Pre-trained model weights ARE bundled with the deployment
    # (ml/models/*.joblib, *.npz — read-only access is fine), so no retraining
    # is needed here, but without any seed data the fraud-ring demo presets
    # (USER_RING1_1, etc.) would have no graph relationships to actually
    # demonstrate. Seed a small synthetic dataset once per cold start so the
    # demo works the same way it does locally, just regenerated each time
    # instead of persisted.
    if IS_SERVERLESS:
        conn = get_raw_sqlite_connection()
        txn_count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        conn.close()
        if txn_count == 0:
            logger.info(f"Serverless cold start with empty DB at {SQLITE_DB_PATH} — seeding a small synthetic dataset...")
            from data.generate_synthetic_data import generate_dataset
            generate_dataset(num_users=150, num_transactions=800, seed=42)

    # Populate the in-memory entity graph + community detection from whatever
    # is currently in the database. Without this, the process boots with an
    # empty graph and every live risk score silently loses its graph signal
    # (shared-device/shared-IP/community features) until a model retrain
    # happens to touch graph_builder.build_graph() as a side effect.
    logger.info("Building in-memory transaction graph from current database state...")
    graph_builder.build_graph()
    graph_builder.detect_communities()
    logger.info(
        f"Graph ready: {graph_builder.G.number_of_nodes()} nodes, "
        f"{graph_builder.G.number_of_edges()} edges."
    )

from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    # Bare domain root has no content of its own — without this, hosts like
    # antideploy.com that put the app straight at the domain root return a
    # plain FastAPI {"detail":"Not Found"} for anyone who visits the base URL
    # instead of /dashboard/ directly.
    return RedirectResponse(url="/dashboard/")

@app.get("/health")
def health_check():
    return {"status": "HEALTHY", "system": "RazorRisk AI Engine", "version": "1.0.0"}

@app.get("/api/v1/stats")
def get_system_stats():
    conn = get_raw_sqlite_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM transactions")
    total_txns = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM devices")
    total_devices = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM risk_scores WHERE risk_tier IN ('HIGH', 'CRITICAL')")
    high_risk_txns = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM investigation_reports")
    investigations_count = cursor.fetchone()[0]

    conn.close()

    return {
        "total_transactions": total_txns,
        "total_users": total_users,
        "total_devices": total_devices,
        "high_risk_transactions": high_risk_txns,
        "investigations_conducted": investigations_count
    }
