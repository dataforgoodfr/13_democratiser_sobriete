from typing import List

from src.domain.paper import OpenAlexPaper, PaperWithText


class OpenAlexPaperIngestionUseCase:

    def __init__(self, open_alex_client, llm_client, paper_repository, pdf_store):
        self.open_alex_client = open_alex_client
        self.llm_client = llm_client
        self.paper_repository = paper_repository
        self.pdf_store = pdf_store


    def ingest_mobility_papers(self):
        papers_list: List[OpenAlexPaper] = self.open_alex_client.get_papers(domain="mobility")
        extracted_text_list: List[PaperWithText] = self.pdf_store.extract_text(papers_list)
        papers_with_taxonomy = []
        for paper in extracted_text_list:
            paper_with_taxonomy = self.llm_client.get_taxonomy_from_paper(paper)
            papers_with_taxonomy.embeddings = self.llm_client.get_embeddings(paper.extract_text)
            papers_with_taxonomy.append(paper_with_taxonomy)
        self.paper_repository.save_all(papers_with_taxonomy)
