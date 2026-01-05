from datetime import datetime, UTC
import logging
from sqlmodel import select, or_

from library.database import get_session
from library.database.models import ScrapingQueue


def get_papers_to_scrape(
    limit: int | None = None, resume_from: datetime | None = None
) -> list[ScrapingQueue]:
    with get_session() as session:
        if resume_from:
            # sometimes the script enters an error mode where all downloads fail
            # this mode is to retry all papers that were scraped after a certain datetime
            stmt = select(ScrapingQueue).where(
                or_(
                    ScrapingQueue.scraped_at >= resume_from,
                    ScrapingQueue.scraping_attempted == False,  # noqa: E712
                )
            )
        else:
            stmt = select(ScrapingQueue).where(ScrapingQueue.scraping_attempted == False)  # noqa: E712

        stmt = stmt.limit(limit)
        papers_to_scrape = session.exec(stmt).all()
        return papers_to_scrape


def mark_paper_scraped(openalex_id: str, download_path: str, used_selenium: bool):
    with get_session() as session:
        try:
            stmt = select(ScrapingQueue).where(ScrapingQueue.openalex_id == openalex_id)
            paper = session.exec(stmt).first()

            if paper:
                paper.scraping_attempted = True
                paper.scraping_successful = True
                paper.download_path = download_path
                paper.required_selenium = used_selenium
                paper.scraped_at = datetime.now(UTC)
                paper.error_message = None  # Clear any previous error
                session.add(paper)

            session.commit()
            logging.info(f"Marked {openalex_id} as scraped")

        except Exception as e:
            session.rollback()
            logging.error(f"Failed to mark paper as scraped: {e}")


def mark_paper_failed(openalex_id: str, error_message: str, used_selenium: bool):
    with get_session() as session:
        try:
            stmt = select(ScrapingQueue).where(ScrapingQueue.openalex_id == openalex_id)
            paper = session.exec(stmt).first()

            if paper:
                paper.scraping_attempted = True
                paper.scraping_successful = False
                paper.required_selenium = used_selenium
                paper.scraped_at = datetime.now(UTC)
                paper.error_message = error_message
                session.add(paper)

            session.commit()
            logging.info(f"Marked {openalex_id} as failed: {error_message}")

        except Exception as e:
            session.rollback()
            logging.error(f"Failed to mark paper as failed: {e}")


def get_scraping_stats():
    with get_session() as session:
        stmt = select(ScrapingQueue.id)
        total = len(session.exec(stmt).all())

        scraped_stmt = select(ScrapingQueue.id).where(ScrapingQueue.scraping_successful == True)  # noqa: E712
        scraped = len(session.exec(scraped_stmt).all())

        failed_stmt = select(ScrapingQueue.id).where(ScrapingQueue.scraping_successful == False)  # noqa: E712
        failed = len(session.exec(failed_stmt).all())

        # Pending = total papers - successfully scraped papers
        pending = total - scraped

        return {"total": total, "scraped": scraped, "failed": failed, "pending": pending}
