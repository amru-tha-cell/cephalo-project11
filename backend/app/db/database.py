"""
SQLAlchemy engine/session setup. Works out of the box with SQLite and
transparently upgrades to PostgreSQL by changing DATABASE_URL in .env
(e.g. postgresql://user:pass@host:5432/cephalo_db).
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import settings

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session per-request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables if they don't exist. Called on app startup."""
    from app.models import db_models  # noqa: F401 (registers models with Base)
    Base.metadata.create_all(bind=engine)
