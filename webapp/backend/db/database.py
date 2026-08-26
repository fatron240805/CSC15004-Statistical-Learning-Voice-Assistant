"""SQLite engine + session — tạo bảng lúc startup (webapp/backend/app.db)."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.config import DATA_DIR
from backend.db.models import Base

DB_PATH = DATA_DIR / "app.db"

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Tạo bảng nếu chưa có. Gọi lúc startup app."""
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    return SessionLocal()
