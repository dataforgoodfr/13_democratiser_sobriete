from itertools import chain
import pyalex


class OpenAlexConnector:
    """Connector for OpenAlex API using pyalex library."""
    
    def __init__(self, email: str = "example@thesufficiencylab.org"):
        self.email = email
        pyalex.config.email = email
    
    def count_works(self, query: str) -> int:
        """Get the count of works matching the query."""
        query = self._sanitize_query(query)
        return pyalex.Works().search(query).count()
    
    def fetch_works(self, query: str, per_page: int = 200) -> tuple:
        """Fetch works matching the query. Returns an iterator and the total count."""
        query = self._sanitize_query(query)
        works_pager = pyalex.Works().search(query)
        total_count = works_pager.count()

        def work_generator():
            for page in works_pager.paginate(per_page=per_page, n_max=None):
                for work in page:
                    yield work

        work_iterator = work_generator()
        return work_iterator, total_count
        
    def fetch_work_ids(self, query: str, per_page: int = 200) -> tuple:
        """Fetch only the IDs of works matching the query. Returns an iterator and the total count."""
        query = self._sanitize_query(query)
        works_pager = pyalex.Works().search(query).select(['id'])
        total_count = works_pager.count()

        def id_generator():
            for page in works_pager.paginate(per_page=per_page, n_max=None):
                for work in page:
                    yield self.get_entity_id_from_url(work['id'])

        id_iterator = id_generator()
        return id_iterator, total_count

    def get_works_from_ids(self, ids: list, per_page: int = 200):
        """Fetch works given a list of OpenAlex IDs."""
        for i in range(0, len(ids), per_page):
            batch_ids = ids[i:i + per_page]
            works = pyalex.Works().filter(id='|'.join(batch_ids))
            for work in chain(*works.paginate(per_page=per_page, n_max=None)):
                yield work

    def get_entity_id_from_url(self, url: str) -> str:
        """Extract OpenAlex entity ID from its URL."""
        return url.split('/')[-1]

    def _sanitize_query(self, query: str) -> str:
        """Sanitize the query string for OpenAlex API."""
        # handle different types of quotes
        q = query.replace('“', '"').replace('”', '"').replace('" ', '"').replace(' "', '"')

        # restore spaces around and uppercase for logical operators
        q = q.replace('"OR"', '" OR "').replace('"AND"', '" AND "')  
        q = q.replace('"or"', '" OR "').replace('"and"', '" AND "')
        
        return q
