import os
from dotenv import load_dotenv
from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.pool import QueuePool

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    poolclass=QueuePool,
    pool_size=10,  # Number of permanent connections
    max_overflow=20,  # Additional connections when pool is full
    pool_recycle=3600,  # Recycle connections after 1 hour
    pool_timeout=30,  # Timeout for getting connection from pool
    echo=False,  # Set to True for SQL debugging
)


def create_tables():
    """Create all tables"""
    SQLModel.metadata.create_all(engine)
    print("Database tables created successfully")


def get_session():
    """Get database session"""
    return Session(engine)
