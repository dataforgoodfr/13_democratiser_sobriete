from datetime import datetime, UTC
import logging
from typing import Literal
from sqlmodel import select

from library.database import get_session
from library.database.models import TextExtractionQueue


def get_already_processed_ids(mode: Literal["all", "txt", "md"] = "all") -> set[str]:
    with get_session() as session:
        stmt = select(
            TextExtractionQueue.openalex_id
        )  # no filtering, as records as added as processed
        if mode == "txt":
            stmt = stmt.where(TextExtractionQueue.raw_text != None)  # noqa: E711
        elif mode == "md":
            stmt = stmt.where(TextExtractionQueue.markdown != None)  # noqa: E711
        results = session.exec(stmt).all()
        return set(results)


def mark_paper_processed(openalex_id: str, s3_folder: str, mode: Literal["txt", "md"]):
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
                paper.processed_at = datetime.now(UTC)
                paper.error_message = None  # Clear any previous error

            else:
                paper = TextExtractionQueue(
                    openalex_id=openalex_id,
                    s3_folder=folder,
                    attempted=True,
                    processed_at=datetime.now(UTC),
                )

            if mode == "txt":
                paper.raw_text = True
            elif mode == "md":
                paper.markdown = True

            session.add(paper)
            session.commit()
            logging.info(f"Marked {openalex_id} as scraped for mode {mode}")

        except Exception as e:
            session.rollback()
            logging.error(f"Failed to mark paper as scraped: {e}")


def mark_paper_failed(
    openalex_id: str, s3_folder: str, error_message: str, mode: Literal["txt", "md"]
):
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
                paper.processed_at = datetime.now(UTC)
                paper.error_message = error_message
            else:
                paper = TextExtractionQueue(
                    openalex_id=openalex_id,
                    s3_folder=folder,
                    attempted=True,
                    processed_at=datetime.now(UTC),
                    error_message=error_message,
                )

            if mode == "txt":
                paper.raw_text = False
            elif mode == "md":
                paper.markdown = False

            session.add(paper)
            session.commit()
            logging.info(f"Marked {openalex_id} as failed for mode {mode}: {error_message}")

        except Exception as e:
            session.rollback()
            logging.error(f"Failed to mark paper as failed: {e}")
