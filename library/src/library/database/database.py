import os
from dotenv import load_dotenv
from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.pool import QueuePool

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]


_engine = None  # Lazy initialization for better process safety


def get_engine():
    """Get or create the database engine (lazy, process-safe)."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_recycle=3600,
            pool_timeout=30,
            echo=False,
        )
    return _engine


def create_tables():
    """Create all tables"""
    SQLModel.metadata.create_all(get_engine())
    print("Database tables created successfully")


def get_session():
    """Get database session"""
    return Session(get_engine())
