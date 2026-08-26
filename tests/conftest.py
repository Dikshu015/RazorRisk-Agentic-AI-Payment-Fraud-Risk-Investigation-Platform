"""Pytest bootstrap for a clean, directly runnable RazorRisk test suite."""
from db.database import init_db


def pytest_sessionstart(session):
    """Ensure schema exists before any test module touches raw SQLite."""
    init_db()
