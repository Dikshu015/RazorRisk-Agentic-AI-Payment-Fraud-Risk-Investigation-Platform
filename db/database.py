import sqlite3
import os
from config import DATABASE_URL, BASE_DIR, SQLITE_DB_PATH
from utils.logger import get_logger

logger = get_logger("database")

# Earlier versions of this file wired up a parallel SQLAlchemy engine + ORM
# models (db/models.py) intended to support Postgres via DATABASE_URL. It
# was never actually load-bearing: init_db() was the only thing that ever
# touched it, and every real query in the app (all of api/, ml/, data/,
# agent/) goes through get_raw_sqlite_connection() below, using raw sqlite3
# syntax that doesn't even work against Postgres. A platform that
# auto-detects infrastructure from dependencies (e.g. antideploy.com seeing
# asyncpg/psycopg2-binary in requirements.txt) would provision a Postgres
# database that the app would then silently never write a single row to.
# Removed rather than left half-wired — see PROJECT_WORKFLOW.md for the
# full account of this if you want to actually add Postgres support properly
# (it needs schema.sql's SQLite-specific syntax ported, and every "?"
# placeholder in every raw query rewritten to "%s").

def init_db():
    """Initialize database tables using schema.sql (SQLite only — see note above)."""
    logger.info(f"Initializing database using connection target: {DATABASE_URL}")
    schema_path = BASE_DIR / "db" / "schema.sql"
    db_file = str(SQLITE_DB_PATH)
    conn = sqlite3.connect(db_file)
    with open(schema_path, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()
    logger.info(f"SQLite database schema applied successfully at {db_file}")

def get_raw_sqlite_connection():
    """Get direct sqlite connection for pandas / graph queries.
    Uses the same path config.py's DATABASE_URL was built from — was
    previously hardcoded to BASE_DIR (read-only on Vercel/HF Spaces/
    Antideploy), silently ignoring DATABASE_URL entirely."""
    return sqlite3.connect(str(SQLITE_DB_PATH))
