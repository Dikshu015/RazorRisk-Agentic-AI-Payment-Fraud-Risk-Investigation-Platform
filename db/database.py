import sqlite3
import os
from config import DATABASE_URL, BASE_DIR, SQLITE_DB_PATH
from utils.logger import get_logger

logger = get_logger("database")

# Historical note: an earlier version carried an unused SQLAlchemy/Postgres ORM
# path alongside the real sqlite3 implementation. It was removed because every
# application query already used raw sqlite3 and the two paths were not actually
# interoperable. RazorRisk intentionally has one application-owned SQLite data
# layer today. See PROJECT_WORKFLOW.md for the migration history.

def init_db():
    """Initialize database tables using schema.sql (SQLite only — see note above)."""
    logger.info(f"Initializing database using connection target: {DATABASE_URL}")
    schema_path = BASE_DIR / "db" / "schema.sql"
    db_file = str(SQLITE_DB_PATH)
    conn = sqlite3.connect(db_file)
    with open(schema_path, "r", encoding="utf-8") as f:
        conn.executescript(f.read())

    # Lightweight SQLite migrations for databases created by older RazorRisk
    # versions. schema.sql handles fresh installs; these ALTERs keep an
    # existing demo DB usable after upgrading the application.
    def add_column_if_missing(table, column, definition):
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            logger.info(f"Database migration: added {table}.{column}")

    add_column_if_missing("transactions", "velocity_enabled", "BOOLEAN DEFAULT 0")
    add_column_if_missing("transactions", "velocity_source", "TEXT DEFAULT 'BACKEND'")
    add_column_if_missing("risk_scores", "stacker_calibrated_score", "FLOAT NOT NULL DEFAULT 0.0")

    conn.commit()
    conn.close()
    logger.info(f"SQLite database schema applied successfully at {db_file}")

def get_raw_sqlite_connection():
    """Get direct sqlite connection for pandas / graph queries.
    Uses the same path config.py's DATABASE_URL was built from — was
    previously hardcoded to BASE_DIR (read-only on Vercel/HF Spaces/
    Antideploy), silently ignoring DATABASE_URL entirely."""
    return sqlite3.connect(str(SQLITE_DB_PATH))
