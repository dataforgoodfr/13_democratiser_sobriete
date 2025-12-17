import logging
import os
from typing import Any, Dict, List

import pymupdf4llm
from .clients.database_client import DatabaseClient
from .clients.qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


def fetch_policies_with_texts(
    db_client: DatabaseClient, qdrant_client: QdrantClient, limit: int = 10, offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Fetch policies from PostgreSQL and their corresponding texts from Qdrant

    Args:
        db_client: Database client instance
        qdrant_client: Qdrant client instance
        limit: Number of policies to fetch
        offset: Number of policies to skip

    Returns:
        List of dictionaries containing policy data with their corresponding texts
    """
    # Fetch policies from PostgreSQL
    policies = db_client.query_policies_abstracts_all(limit=limit, offset=offset)

    # Extract openalex_ids
    openalex_ids = [policy["openalex_id"] for policy in policies if policy.get("openalex_id")]

    # Fetch texts from Qdrant in batch
    qdrant_texts = {}
    if openalex_ids:
        qdrant_data = qdrant_client.get_texts_by_ids(openalex_ids)
        for data in qdrant_data:
            qdrant_texts[data["id"]] = data

    # Combine policy data with Qdrant texts
    enriched_policies = []
    for policy in policies:
        enriched_policy = policy.copy()
        openalex_id = policy.get("openalex_id")

        if openalex_id and openalex_id in qdrant_texts:
            qdrant_data = qdrant_texts[openalex_id]
            enriched_policy["qdrant_text"] = qdrant_data["payload"].get("text", "")
            enriched_policy["qdrant_payload"] = qdrant_data["payload"]
            enriched_policy["has_text"] = True
        else:
            enriched_policy["qdrant_text"] = ""
            enriched_policy["qdrant_payload"] = {}
            enriched_policy["has_text"] = False

        enriched_policies.append(enriched_policy)

    return enriched_policies


def get_policy_text(qdrant_client: QdrantClient, policy: Dict[str, Any]) -> str:
    """
    Get the text of a policy from Qdrant
    """
    openalex_id = policy.get("openalex_id")
    logger.info(f"OpenAlex ID: {openalex_id}")
    qdrant_data = qdrant_client.get_text_by_id(openalex_id)
    if qdrant_data:
        logger.info(
            f"  Found text in Qdrant with payload keys: {list(qdrant_data['payload'].keys())}"
        )

        # Extract text content from payload
        if "text" in qdrant_data["payload"]:
            text_content = qdrant_data["payload"]["text"]
            # Truncate for display
            preview = text_content[:200] + "..." if len(text_content) > 200 else text_content
            logger.info(f"  Text preview: {preview}")
        else:
            logger.info(
                f"  No 'text' field found in payload. Available fields: {list(qdrant_data['payload'].keys())}"
            )
    else:
        logger.warning(f"  No text found in Qdrant for openalex_id: {openalex_id}")
    return qdrant_data["payload"]["text"]


def get_pymupdf4llm(
    pdf_path: str,
    bool_write_images: bool = False,
    bool_embed_images: bool = False,
) -> list[dict]:
    """
    Extract the content from a PDF file using pymupdf4llm.
    Args:
        pdf_path (str): The path to the PDF file.
        bool_write_images (bool): Whether to write images to disk.
        bool_embed_images (bool): Whether to embed images in the markdown.
    Returns:
        list[dict]: The extracted content.
    """
    # Get the name of the file from the path, without the extension
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Extract the content from the PDF
    content_md = pymupdf4llm.to_markdown(
        doc=pdf_path,
        page_chunks=True,
        write_images=bool_write_images,
        embed_images=bool_embed_images,
        image_path=os.path.join("images_pdf", file_name),
        show_progress=False,
    )

    return content_md
