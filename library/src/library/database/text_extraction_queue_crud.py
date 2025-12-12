from datetime import datetime, UTC
import logging
from sqlmodel import select

from library.database import get_session
from library.database.models import TextExtractionQueue


def get_already_processed_ids() -> set[str]:
    with get_session() as session:
        stmt = select(TextExtractionQueue.openalex_id).where(
            TextExtractionQueue.attempted == False  # noqa
        )
        results = session.exec(stmt).all()
        return set(results)


def mark_paper_processed(openalex_id: str, s3_folder: str):
    folder = s3_folder.split("/")[-1]

    with get_session() as session:
        try:
            stmt = select(TextExtractionQueue).where(
                TextExtractionQueue.openalex_id == openalex_id
            )
            paper = session.exec(stmt).first()

            if paper:
                paper.s3_folder = folder
                paper.attempted = True
                paper.successful = True
                paper.processed_at = datetime.now(UTC)
                paper.error_message = None  # Clear any previous error

            else:
                paper = TextExtractionQueue(
                    openalex_id=openalex_id,
                    s3_folder=folder,
                    attempted=True,
                    successful=True,
                    processed_at=datetime.now(UTC),
                )

            session.add(paper)
            session.commit()
            logging.info(f"Marked {openalex_id} as scraped")

        except Exception as e:
            session.rollback()
            logging.error(f"Failed to mark paper as scraped: {e}")


def mark_paper_failed(openalex_id: str, s3_folder: str, error_message: str):
    folder = s3_folder.split("/")[-1]

    with get_session() as session:
        try:
            stmt = select(TextExtractionQueue).where(
                TextExtractionQueue.openalex_id == openalex_id
            )
            paper = session.exec(stmt).first()

            if paper:
                paper.s3_folder = folder
                paper.attempted = True
                paper.successful = False
                paper.processed_at = datetime.now(UTC)
                paper.error_message = error_message
            else:
                paper = TextExtractionQueue(
                    openalex_id=openalex_id,
                    s3_folder=folder,
                    attempted=True,
                    successful=False,
                    processed_at=datetime.now(UTC),
                    error_message=error_message,
                )

            session.add(paper)
            session.commit()
            logging.info(f"Marked {openalex_id} as failed: {error_message}")

        except Exception as e:
            session.rollback()
            logging.error(f"Failed to mark paper as failed: {e}")
