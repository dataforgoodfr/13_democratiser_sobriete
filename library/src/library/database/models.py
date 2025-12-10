"""
Database Models and Operations
Simple models for paper scraping queue management.
"""

from datetime import datetime
from sqlmodel import SQLModel, Field


class OpenAlexWorks(SQLModel, table=True):
    """
    Model for the main library table containing Open Alex works.
    """

    __tablename__ = "openalex_works"

    # in SQLModel, primary_key must be optional to allow autoincrement
    id: int | None = Field(default=None, primary_key=True)
    openalex_id: str = Field(unique=True, index=True)
    doi: str | None = None
    title: str
    abstract: str | None = None
    language: str
    publication_date: datetime | None = None
    publication_type: str
    fwci: float | None = None
    is_oa: str
    oa_status: str
    landing_page_url: str | None = None
    pdf_url: str | None = None


class ScrapingQueue(SQLModel, table=True):
    """
    Queue tracking works to scrape.
    """

    __tablename__ = "scraping_queue"

    id: int | None = Field(default=None, primary_key=True)

    # basic info from Open Alex
    openalex_id: str = Field(foreign_key="openalex_works.openalex_id")
    landing_page_url: str | None = None
    pdf_url: str | None = None

    # scraping status
    scraping_attempted: bool = False
    scraping_successful: bool | None = None
    download_path: str | None = None
    scraped_at: datetime | None = None
    error_message: str | None = None
