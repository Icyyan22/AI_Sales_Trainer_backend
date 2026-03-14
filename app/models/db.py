from __future__ import annotations

import datetime
import json

from sqlalchemy import JSON, DateTime, Integer, String, Text, func, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import settings


class Base(DeclarativeBase):
    pass


class UserRecord(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(200), nullable=False)
    display_name: Mapped[str] = mapped_column(String(50), default="")
    avatar_path: Mapped[str | None] = mapped_column(String(200), nullable=True)
    role: Mapped[str] = mapped_column(String(20), default="user")  # user / admin / super_admin
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )


class SessionRecord(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scenario_id: Mapped[str] = mapped_column(String(100), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    difficulty: Mapped[str] = mapped_column(String(20), default="normal")
    status: Mapped[str] = mapped_column(String(20), default="active")
    final_coverage: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    final_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    completed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, nullable=True
    )


class MessageRecord(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    analysis: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    strategy: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    turn: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )


class FeedbackRecord(Base):
    __tablename__ = "feedbacks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    message_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    human_labels: Mapped[dict] = mapped_column(JSON, nullable=False)
    ai_labels: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    agreement_rate: Mapped[float | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )


engine = create_async_engine(settings.database_url, echo=False)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Incremental migrations for existing databases
    _migrations = [
        ("users", "role", "ALTER TABLE users ADD COLUMN role VARCHAR(20) DEFAULT 'user'"),
        ("sessions", "user_id", "ALTER TABLE sessions ADD COLUMN user_id VARCHAR(36)"),
    ]
    async with engine.begin() as conn:
        for table, column, sql in _migrations:
            try:
                await conn.execute(text(sql))
            except Exception:
                pass  # Column already exists


async def get_db() -> AsyncSession:
    async with async_session_factory() as session:
        yield session
