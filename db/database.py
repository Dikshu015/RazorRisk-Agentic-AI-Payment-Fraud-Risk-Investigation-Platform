import sqlite3
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config import DATABASE_URL, BASE_DIR, SQLITE_DB_PATH
from utils.logger import get_logger

logger = get_logger("database")

Base = declarative_base()

# SQLite or PostgreSQL setup
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency for API database session management"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables using schema.sql"""
    logger.info(f"Initializing database using connection target: {DATABASE_URL}")
    schema_path = BASE_DIR / "db" / "schema.sql"
    
    if DATABASE_URL.startswith("sqlite"):
        db_file = DATABASE_URL.replace("sqlite:///", "")
        conn = sqlite3.connect(db_file)
        with open(schema_path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()
        conn.close()
        logger.info(f"SQLite database schema applied successfully at {db_file}")
    else:
        Base.metadata.create_all(bind=engine)
        logger.info("SQLAlchemy models synced with database target.")

def get_raw_sqlite_connection():
    """Get direct sqlite connection for pandas / graph queries if using sqlite.
    Uses the same path config.py's DATABASE_URL was built from — was
    previously hardcoded to BASE_DIR (read-only on Vercel), silently
    ignoring DATABASE_URL entirely."""
    return sqlite3.connect(str(SQLITE_DB_PATH))
