from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from config import BASE_DIR
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
