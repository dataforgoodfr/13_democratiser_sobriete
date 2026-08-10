import os
import time
from itertools import chain
from typing import Generator
import pyalex
from pyalex import Works, Work


DESIRED_FIELDS = [
    "id",
    "doi",
    "title",
    "abstract_inverted_index",
    "language",
    "publication_date",
    "type",
    "open_access",
    "best_oa_location",
    "has_fulltext",
    "fwci",
]
BASE_FILTERS = {
    "is_paratext": False,
    "is_retracted": False,
}


class OpenAlexConnector:
    """Connector for OpenAlex API using pyalex library."""

    def __init__(self, email: str | None = None):
        # A real mailto puts us in OpenAlex's polite pool — the anonymous pool
        # throttles datacenter IPs aggressively (429 storms from Scaleway).
        self.email = email or os.environ.get(
            "OPENALEX_MAILTO", "example@thesufficiencylab.org"
        )
        pyalex.config.email = self.email
        # Since Feb 2026 OpenAlex requires an API key for production use:
        # ~100k credits/day free with a key vs ~100 without (searches cost 10).
        if os.environ.get("OPENALEX_API_KEY"):
            pyalex.config.api_key = os.environ["OPENALEX_API_KEY"]
        # Exponential backoff on 429/5xx instead of failing whole themes.
        pyalex.config.max_retries = int(os.environ.get("OPENALEX_MAX_RETRIES", "8"))
        pyalex.config.retry_backoff_factor = float(os.environ.get("OPENALEX_BACKOFF", "0.5"))
        pyalex.config.retry_http_codes = [429, 500, 503]
        # Pause between pages to stay under the 10 req/s polite-pool limit.
        self.page_delay = float(os.environ.get("OPENALEX_PAGE_DELAY", "0.15"))

    def count_works(self, query: str, filters: dict = BASE_FILTERS) -> int:
        """Get the count of works matching the query."""
        query = self._sanitize_query(query)
        return Works().search(query).filter(**filters).count()

    def fetch_works(
        self,
        query: str,
        per_page: int = 200,
        filters: dict = BASE_FILTERS,
        fields: list[str] = DESIRED_FIELDS,
    ) -> tuple:
        """Fetch works matching the query. Returns an iterator and the total count."""
        query = self._sanitize_query(query)
        works_pager = Works().search(query).filter(**filters).select(fields)
        total_count = works_pager.count()

        def work_generator():
            for page in works_pager.paginate(per_page=per_page, n_max=None):
                for work in page:
                    yield work
                time.sleep(self.page_delay)

        work_iterator = work_generator()
        return work_iterator, total_count

    def fetch_work_ids(
        self, query: str, per_page: int = 200, filters: dict = BASE_FILTERS
    ) -> tuple:
        """Fetch only the IDs of works matching the query. Returns an iterator and the total count."""
        query = self._sanitize_query(query)
        works_pager = Works().search(query).filter(**filters).select(["id"])
        total_count = works_pager.count()

        def id_generator():
            for page in works_pager.paginate(per_page=per_page, n_max=None):
                for work in page:
                    yield self.get_entity_id_from_url(work["id"])
                time.sleep(self.page_delay)

        id_iterator = id_generator()
        return id_iterator, total_count

    def get_works_from_ids(
        self,
        ids: list,
        per_page: int = 100,
        filters: dict = None,
        fields: list[str] = DESIRED_FIELDS,
    ) -> Generator[Work, None, None]:
        """Fetch works given a list of OpenAlex IDs."""
        for i in range(0, len(ids), per_page):
            batch_ids = ids[i : i + per_page]
            works = (
                Works().filter(openalex_id="|".join(batch_ids)).filter(**filters).select(fields)
            )
            for work in chain(*works.paginate(per_page=per_page, n_max=None)):
                yield work
            time.sleep(self.page_delay)

    def get_entity_id_from_url(self, url: str) -> str:
        """Extract OpenAlex entity ID from its URL."""
        return url.split("/")[-1]

    def _sanitize_query(self, query: str) -> str:
        """Sanitize the query string for OpenAlex API."""
        # handle different types of quotes
        q = query.replace("“", '"').replace("”", '"').replace('" ', '"').replace(' "', '"')

        # restore spaces around and uppercase for logical operators
        q = q.replace('"OR"', '" OR "').replace('"AND"', '" AND "')
        q = q.replace('"or"', '" OR "').replace('"and"', '" AND "')

        return q
