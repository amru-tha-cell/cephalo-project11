"""Standalone script: `python -m app.db.init_db` to (re)create tables."""
from app.db.database import init_db

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
