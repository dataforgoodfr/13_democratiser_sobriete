import os
from typing import List, Optional

from icecream import ic
from taxonomy.paper_taxonomy import OpenAlexPaper, PaperWithText, PaperTaxonomy


class OpenAlexPaperIngestionUseCase:

    def __init__(
        self,
        open_alex_client,
        llm_client,
        pdf_extractor,
        tax_folder
    ):
        self.open_alex_client = open_alex_client
        self.llm_client = llm_client
        self.pdf_extractor = pdf_extractor
        self.tax_folder = tax_folder

    def ingest_papers_with_query(self, query: str = "mobility", limit: Optional[int] = 10, model: str = "smollm:135m", prompt_type: str = "basic"):
        papers_list: List[OpenAlexPaper] = self.open_alex_client.get_papers(
            query_requested=query, limit=limit
        )
        extracted_text_list: List[PaperWithText] = self.pdf_extractor.extract_text(papers_list)
        papers_with_taxonomy = []
        os.makedirs(self.tax_folder, exist_ok=True)
        
        for pix, paper in enumerate(extracted_text_list[:limit]):
            print(f"Processing paper text to fill taxonomy : {pix + 1}/{len(extracted_text_list)}")
            # TODO : In Pydantic classes : Add the taxonomy to the input object, for enrichment in one entity
            paper_name = paper.openalex_paper.paper_name
            paper_with_taxonomy: PaperTaxonomy = self.llm_client.get_taxonomy_from_paper(paper, model, prompt_type)

            # save the extracted tax
            output_path = os.path.join(self.tax_folder, (paper_name + "_tax.json"))
            with open(output_path, "w") as f:
                f.write(paper_with_taxonomy.model_dump_json())

            # papers_with_taxonomy.embeddings = self.llm_client.get_embeddings(paper.extract_text)
            papers_with_taxonomy.append(paper_with_taxonomy)
        
        ic(len(papers_with_taxonomy))
        
