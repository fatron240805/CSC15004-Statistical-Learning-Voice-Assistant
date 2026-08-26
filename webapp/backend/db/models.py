"""SQLAlchemy models — schema theo mục 7 spec_ModuleB_web_app.md."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    embedding = Column(JSON, nullable=True)  # 192-dim float list, đã L2-normalize
    enrollment_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Preference(Base):
    __tablename__ = "preferences"

    user_id = Column(Integer, ForeignKey("users.user_id"), primary_key=True)
    favorite_tracks = Column(JSON, nullable=True)  # list track_id/tên bài, khớp Deezer


class ActionLog(Base):
    __tablename__ = "action_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    action = Column(String, nullable=False)  # "unlock_door" | "start_engine" | "play_playlist" | ...
    verified = Column(Boolean, nullable=False)  # kết quả verify()/identify()
    score = Column(Float, nullable=True)  # cosine similarity thực tế, dùng để debug/báo cáo
