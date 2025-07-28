"""
Database Models and Operations
Simple models for paper scraping queue management
"""

import os
from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field, create_engine, Session, select
from sqlalchemy import Column, String
import logfire

# Use environment variable for database URL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://u4axloluqibskgvdikuy:g2rXgpHSbztokCbFxSyR@bk8htvifqendwt1wlzat-postgresql.services.clever-cloud.com:7327/bk8htvifqendwt1wlzat"
)

class PoliciesAbstractsAll(SQLModel, table=True):
    """
    Model for the existing policies_abstracts_all table
    Adjust fields based on actual table structure
    """
    __tablename__ = "policies_abstracts_all"
    
    openalex_id: str = Field(sa_column=Column(String, primary_key=True))
    doi: Optional[str] = Field(default=None, sa_column=Column(String))

class ScrapingQueue(SQLModel, table=True):
    """
    Simple queue table to track papers that need to be scraped
    """
    __tablename__ = "scraping_queue"
    
    openalex_id: str = Field(sa_column=Column(String, primary_key=True))
    doi: Optional[str] = Field(default=None, sa_column=Column(String))
    
    # Simple status tracking
    scraped: bool = Field(default=False)
    assigned_folder: Optional[int] = Field(default=None)  # 0-11 for the 12 folders
    pdf_path: Optional[str] = Field(default=None, sa_column=Column(String))
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    scraped_at: Optional[datetime] = Field(default=None)
    
    # Error tracking
    error_message: Optional[str] = Field(default=None, sa_column=Column(String))

# Database engine and session management
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def create_tables():
    """Create all tables"""
    SQLModel.metadata.create_all(engine)
    logfire.info("Database tables created successfully")

def get_session():
    """Get database session"""
    return Session(engine)

def populate_scraping_queue(limit: Optional[int] = None) -> int:
    """
    Populate scraping queue from policies_abstracts_all table
    Returns the number of papers added to queue
    """
    with get_session() as session:
        try:
            # Create tables if they don't exist
            create_tables()
            
            # Query policies_abstracts_all table
            policies_stmt = select(PoliciesAbstractsAll)
            if limit:
                policies_stmt = policies_stmt.limit(limit)
                
            all_policies = session.exec(policies_stmt).all()
            
            added_count = 0
            for policy in all_policies:
                # Check if already exists in queue
                existing_stmt = select(ScrapingQueue).where(
                    ScrapingQueue.openalex_id == policy.openalex_id
                )
                existing = session.exec(existing_stmt).first()
                
                if not existing:
                    # Add to scraping queue
                    queue_item = ScrapingQueue(
                        openalex_id=policy.openalex_id,
                        doi=policy.doi
                    )
                    session.add(queue_item)
                    added_count += 1
                    
            session.commit()
            logfire.info(f"Added {added_count} papers to scraping queue")
            return added_count
            
        except Exception as e:
            session.rollback()
            logfire.error(f"Failed to populate scraping queue: {e}")
            raise

def get_papers_to_scrape(limit: Optional[int] = None):
    """Get papers that need to be scraped"""
    with get_session() as session:
        stmt = select(ScrapingQueue).where(ScrapingQueue.scraped == False)
        if limit:
            stmt = stmt.limit(limit)
        return session.exec(stmt).all()

def mark_paper_scraped(openalex_id: str, pdf_path: str, folder_id: int):
    """Mark a paper as scraped with its file path and folder"""
    with get_session() as session:
        try:
            stmt = select(ScrapingQueue).where(ScrapingQueue.openalex_id == openalex_id)
            paper = session.exec(stmt).first()
            
            if paper:
                paper.scraped = True
                paper.pdf_path = pdf_path
                paper.assigned_folder = folder_id
                paper.scraped_at = datetime.utcnow()
                session.add(paper)
                session.commit()
                logfire.info(f"Marked {openalex_id} as scraped")
            
        except Exception as e:
            session.rollback()
            logfire.error(f"Failed to mark paper as scraped: {e}")

def mark_paper_failed(openalex_id: str, error_message: str):
    """Mark a paper as failed to scrape"""
    with get_session() as session:
        try:
            stmt = select(ScrapingQueue).where(ScrapingQueue.openalex_id == openalex_id)
            paper = session.exec(stmt).first()
            
            if paper:
                paper.error_message = error_message
                session.add(paper)
                session.commit()
                logfire.info(f"Marked {openalex_id} as failed: {error_message}")
                
        except Exception as e:
            session.rollback()
            logfire.error(f"Failed to mark paper as failed: {e}")

def reset_failed_papers() -> int:
    """Reset failed papers to try again (clear error message)"""
    with get_session() as session:
        try:
            stmt = select(ScrapingQueue).where(
                (ScrapingQueue.scraped == False) &
                (ScrapingQueue.error_message.isnot(None))
            )
            papers_to_reset = session.exec(stmt).all()
            
            reset_count = 0
            for paper in papers_to_reset:
                paper.error_message = None
                session.add(paper)
                reset_count += 1
                
            session.commit()
            logfire.info(f"Reset {reset_count} failed papers for retry")
            return reset_count
            
        except Exception as e:
            session.rollback()
            logfire.error(f"Failed to reset failed papers: {e}")
            return 0

def get_scraping_stats():
    """Get scraping statistics"""
    with get_session() as session:
        try:
            total_stmt = select(ScrapingQueue)
            total = len(session.exec(total_stmt).all())
            
            scraped_stmt = select(ScrapingQueue).where(ScrapingQueue.scraped == True)
            scraped = len(session.exec(scraped_stmt).all())
            
            failed_stmt = select(ScrapingQueue).where(ScrapingQueue.error_message.isnot(None))
            failed = len(session.exec(failed_stmt).all())
            
            return {
                "total": total,
                "scraped": scraped,
                "failed": failed,
                "pending": total - scraped - failed
            }
            
        except Exception as e:
            logfire.error(f"Failed to get scraping stats: {e}")
            return {} 