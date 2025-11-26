"""
Database Package
Handles database models and queue management for paper scraping
"""

from .models import (
    ScrapingQueue,
    PoliciesAbstractsAll,
    get_session,
    create_tables,
    get_papers_to_scrape,
    mark_paper_scraped,
    mark_paper_failed,
    get_scraping_stats,
    clear_scraping_queue,
    get_paper_doi
)

__all__ = [
    'ScrapingQueue',
    'PoliciesAbstractsAll',
    'get_session',
    'create_tables',
    'get_papers_to_scrape',
    'mark_paper_scraped',
    'mark_paper_failed',
    'get_scraping_stats',
    'clear_scraping_queue',
    'get_paper_doi'
] 