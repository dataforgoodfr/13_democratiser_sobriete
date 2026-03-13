from concurrent.futures import ThreadPoolExecutor

from pyalex import Works
from qdrant_client import QdrantClient

from .config import settings
from .models import DocumentChunk, Publication


qdrant_client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)

executor = ThreadPoolExecutor(max_workers=2)


def get_publications_from_chunks(chunks: list[DocumentChunk]) -> list[Publication]:
    """Fetch publications from OpenAlex for the given chunks."""
    ids = [chunk.openalex_id for chunk in chunks]
    if not ids:
        return []

    fields = [
        "id",
        "title",
        "doi",
        "abstract_inverted_index",
        "open_access",
        "authorships",
        "publication_year",
    ]
    works = Works().filter(openalex_id="|".join(ids)).select(fields).get()
    publications = []
    for work in works:
        authors = [
            f"{auth.get('author', {}).get('display_name')}"
            for auth in work.get("authorships", [])
        ]
        openalex_id = work["id"].split("/")[-1]
        publication = Publication(
            openalex_id=openalex_id,
            doi=work.get("doi"),
            title=work["title"],
            abstract=work["abstract"],
            authors=authors,
            publication_year=work["publication_year"],
            url=work.get("open_access", {}).get("oa_url"),
            retrieved_chunks=[chunk for chunk in chunks if chunk.openalex_id == openalex_id],
        )
        publications.append(publication)

    return publications