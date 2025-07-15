"""
Database Models and Operations
Simple models for paper scraping queue management
"""

import os
from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field, create_engine, Session, select
from sqlalchemy import Column, String
from sqlalchemy.pool import QueuePool
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
    assigned_folder: Optional[str] = Field(default=None, sa_column=Column(String))  # 0-11 for the 12 folders
    pdf_path: Optional[str] = Field(default=None, sa_column=Column(String))
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    scraped_at: Optional[datetime] = Field(default=None)
    
    # Error tracking
    error_message: Optional[str] = Field(default=None, sa_column=Column(String))

# Database engine with optimized connection pooling
engine = create_engine(
    DATABASE_URL, 
    pool_pre_ping=True,
    poolclass=QueuePool,
    pool_size=10,  # Number of permanent connections
    max_overflow=20,  # Additional connections when pool is full
    pool_recycle=3600,  # Recycle connections after 1 hour
    pool_timeout=30,  # Timeout for getting connection from pool
    echo=False  # Set to True for SQL debugging
)

def create_tables():
    """Create all tables"""
    SQLModel.metadata.create_all(engine)
    logfire.info("Database tables created successfully")

def get_session():
    """Get database session"""
    return Session(engine)



def get_papers_to_scrape(limit: Optional[int] = None):
    """Get papers that need to be scraped from PoliciesAbstractsAll, excluding already processed ones"""
    with get_session() as session:
        # Get all OpenAlex IDs that have been successfully scraped (in ScrapingQueue with scraped=True)
        processed_ids_stmt = select(ScrapingQueue.openalex_id).where(ScrapingQueue.scraped == True)
        processed_ids = set(session.exec(processed_ids_stmt).all())
        
        # Get papers from PoliciesAbstractsAll that haven't been successfully scraped yet
        policies_stmt = select(PoliciesAbstractsAll)
        if limit:
            # Get extra papers since some might be already processed
            policies_stmt = policies_stmt.limit(limit * 3)
        else:
            policies_stmt = policies_stmt.limit(1000)  # Reasonable default batch size
            
        all_policies = session.exec(policies_stmt).all()
        
        # Filter out already processed papers
        papers_to_scrape = []
        for policy in all_policies:
            if policy.openalex_id not in processed_ids:
                papers_to_scrape.append(policy)
                
                # Stop if we have enough papers
                if limit and len(papers_to_scrape) >= limit:
                    break
        
        logfire.info(f"Found {len(papers_to_scrape)} papers to scrape (excluded {len(processed_ids)} already processed)")
        return papers_to_scrape

def mark_paper_scraped(openalex_id: str, pdf_path: str, folder_id: int):
    """Mark a paper as scraped with its file path and folder - creates or updates entry in ScrapingQueue"""
    with get_session() as session:
        try:
            stmt = select(ScrapingQueue).where(ScrapingQueue.openalex_id == openalex_id)
            paper = session.exec(stmt).first()
            
            if paper:
                # Update existing entry
                paper.scraped = True
                paper.pdf_path = pdf_path
                paper.assigned_folder = folder_id
                paper.scraped_at = datetime.utcnow()
                paper.error_message = None  # Clear any previous error
                session.add(paper)
            else:
                # Create new entry for successful paper
                # Get DOI from source table if available
                policies_stmt = select(PoliciesAbstractsAll).where(PoliciesAbstractsAll.openalex_id == openalex_id)
                source_paper = session.exec(policies_stmt).first()
                doi = source_paper.doi if source_paper else None
                
                scraped_paper = ScrapingQueue(
                    openalex_id=openalex_id,
                    doi=doi,
                    scraped=True,
                    pdf_path=pdf_path,
                    assigned_folder=folder_id,
                    scraped_at=datetime.utcnow()
                )
                session.add(scraped_paper)
                
            session.commit()
            logfire.info(f"Marked {openalex_id} as scraped")
            
        except Exception as e:
            session.rollback()
            logfire.error(f"Failed to mark paper as scraped: {e}")

def mark_paper_failed(openalex_id: str, error_message: str):
    """Mark a paper as failed to scrape - creates or updates entry in ScrapingQueue"""
    with get_session() as session:
        try:
            stmt = select(ScrapingQueue).where(ScrapingQueue.openalex_id == openalex_id)
            paper = session.exec(stmt).first()
            
            if paper:
                # Update existing entry
                paper.error_message = error_message
                paper.scraped = False
                session.add(paper)
            else:
                # Create new entry for failed paper
                # Get DOI from source table if available
                policies_stmt = select(PoliciesAbstractsAll).where(PoliciesAbstractsAll.openalex_id == openalex_id)
                source_paper = session.exec(policies_stmt).first()
                doi = source_paper.doi if source_paper else None
                
                failed_paper = ScrapingQueue(
                    openalex_id=openalex_id,
                    doi=doi,
                    scraped=False,
                    error_message=error_message
                )
                session.add(failed_paper)
                
            session.commit()
            logfire.info(f"Marked {openalex_id} as failed: {error_message}")
                
        except Exception as e:
            session.rollback()
            logfire.error(f"Failed to mark paper as failed: {e}")


def clear_scraping_queue() -> int:
    """Clear the entire scraping queue table"""
    with get_session() as session:
        try:
            # Count entries before deletion
            count_stmt = select(ScrapingQueue)
            before_count = len(session.exec(count_stmt).all())
            
            if before_count == 0:
                return 0
            
            # Delete all entries
            from sqlalchemy import delete
            delete_stmt = delete(ScrapingQueue)
            session.exec(delete_stmt)
            session.commit()
            
            logfire.info(f"Cleared scraping queue: {before_count} entries")
            return before_count
            
        except Exception as e:
            session.rollback()
            logfire.error(f"Failed to clear scraping queue: {e}")
            raise

def get_scraping_stats():
    """Get scraping statistics from both source and tracking tables"""
    with get_session() as session:
        try:
            # Total papers from source table
            total_stmt = select(PoliciesAbstractsAll)
            total = len(session.exec(total_stmt).all())
            
            # Successfully scraped papers from tracking table
            scraped_stmt = select(ScrapingQueue).where(ScrapingQueue.scraped == True)
            scraped = len(session.exec(scraped_stmt).all())
            
            # Failed papers from tracking table (scraped=False with error message)
            failed_stmt = select(ScrapingQueue).where(
                (ScrapingQueue.scraped == False) & 
                (ScrapingQueue.error_message.isnot(None))
            )
            failed = len(session.exec(failed_stmt).all())
            
            # Pending = total papers - successfully scraped papers
            pending = total - scraped
            
            return {
                "total": total,
                "scraped": scraped,
                "failed": failed,
                "pending": pending
            }
            
        except Exception as e:
            logfire.error(f"Failed to get scraping stats: {e}")
            return {}

def get_paper_doi(openalex_id: str) -> Optional[str]:
    """Get the DOI for a specific OpenAlex ID from the source table"""
    with get_session() as session:
        try:
            stmt = select(PoliciesAbstractsAll.doi).where(PoliciesAbstractsAll.openalex_id == openalex_id)
            result = session.exec(stmt).first()
            return result if result else None
        except Exception as e:
            logfire.error(f"Failed to get DOI for {openalex_id}: {e}")
            return None

 