"""
Scraping Package
Handles paper downloading and distribution from OpenAlex
"""

from .targeted_scraper import TargetedPaperScraper
from .test_scraping import main as test_scraping_main

__all__ = ['TargetedPaperScraper', 'test_scraping_main'] 