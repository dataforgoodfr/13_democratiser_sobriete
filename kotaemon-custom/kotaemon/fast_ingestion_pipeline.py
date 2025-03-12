from typing import List

from kotaemon.base import Document, Param, lazy
from kotaemon.base.component import BaseComponent
from kotaemon.base.schema import LLMInterface
from kotaemon.embeddings import OpenAIEmbeddings
from kotaemon.indices import VectorIndexing
from kotaemon.indices.vectorindex import VectorRetrieval
from kotaemon.llms.chats.openai import ChatOpenAI
from kotaemon.storages import LanceDBDocumentStore
from kotaemon.storages.vectorstores.qdrant import QdrantVectorStore

from wsl_library.pdfextraction import TAXS
from wsl_library.pdfextraction.llm import prompts
from wsl_library.pdfextraction.llm.ollama_extraction import extract_ollama_from_paper
from wsl_library.pdfextraction.pdf.pymu import get_pymupdf4llm

PROMPT_FCTS = {name: obj for name, obj in prompts.__dict__.items() if callable(obj)}
TAXS_NAME = list(TAXS.keys())


class IndexingPipeline(VectorIndexing):
    vector_store: QdrantVectorStore = Param(
        lazy(QdrantVectorStore).withx(
            url="http://localhost:6333",
            api_key="None",
            collection_name="default",
        ),
        ignore_ui=True,  # usefull ?
    )
    doc_store: LanceDBDocumentStore = Param(
        lazy(LanceDBDocumentStore).withx(
            path="./kotaemon-custom/kotaemon/ktem_app_data/user_data/docstore",
        ),
        ignore_ui=True,
    )
    embedding: OpenAIEmbeddings = Param(
        lazy(OpenAIEmbeddings).withx(
            # base_url="http://172.17.0.1:11434/v1/",
            base_url="http://localhost:11434/v1/",
            model="snowflake-arctic-embed2",
            api_key="ollama",
        ),
        ignore_ui=True,
    )
    pdf_path: str
    model_name: str = "llama3.2:latest"
    taxonomy_name: str = "PaperTaxonomy"
    prompt_type: str = "main_parts_prompt"

    taxonomy = TAXS.get(taxonomy_name)
    if prompt_type in PROMPT_FCTS:
        prompt_fct: callable = PROMPT_FCTS.get(prompt_type)
    else:
        raise ValueError(f"{prompt_type} has not been implemented")

    def run(self, text: str | list[str], metadatas: dict | list[dict] | None) -> Document:
        """Normally, this indexing pipeline returns nothing. For demonstration,
        we want it to return something, so let's return the number of documents
        in the vector store
        """
        # --- LOGIC INGESTION PIPELINE ----

        # TODO

        # core extraction
        # inference metadatas
        # summarization
        #

        # --- END - LOGIC INGESTION PIPELINE ----

        # Final ingestion (=> to 3 databases)
        super().run(text, metadatas)

        return Document(self.vector_store.count())

    def run_pdf(self, pdf_name: str) -> Document:
        """
        ETL pipeline for a single pdf file
        1. Extract text and taxonomy from pdf
        2. Transform taxonomy (flattening)
        3. Ingest text and taxonomy into the vector store

        For demonstration, we want it to return something, so let's return the number of documents in the vector store
        """

        # get the text from the pdf
        content = get_pymupdf4llm(
            self.pdf_path + pdf_name
        )  # List of pdf node, could be used latter as paragraphs
        text = "\n".join([c["text"] for c in content])

        # make the prompt
        prompt = self.prompt_fct(text)

        # extract the taxonomy from the extracted texts
        paper_tax = extract_ollama_from_paper(
            prompt, self.model_name, self.taxonomy
        ).model_dump(mode="json")  #  Python dict

        # Final ingestion (=> to 3 databases)
        # Later we want to downscale this ingestion to paragraph level
        # Yet it's at paper level
        super().run([text], [paper_tax])

        return Document(self.vector_store.count())

    def run_saved():
        """
        ETL pipeline that reuses the saved extracted text and taxonomy from wsl_library.pdfextraction.main
        """
        # TODO
        pass


class Pipeline(BaseComponent):
    """
    from simple_pipeline.py, a better RAG pipeline must exist somewhere
    """

    llm: ChatOpenAI = ChatOpenAI.withx(
        base_url="http://localhost:11434/v1/",
        model="gemma2:2b",
        api_key="ollama",
    )

    retrieval_pipeline: VectorRetrieval

    def run(self, text: str) -> LLMInterface:
        matched_texts: List[Document] = self.retrieval_pipeline(text)
        return self.llm("\n".join(map(str, matched_texts)))


if __name__ == "__main__":
    indexing_pipeline = IndexingPipeline(pdf_path="./tests/pdfextraction/pdf/")
    indexing_pipeline.run_pdf("1-s2.0-S2211467X23001748-main.pdf")
    indexing_pipeline.run_pdf("1-s2.0-S0094119008001095-main.pdf")

    rag_pipeline = Pipeline(retrieval_pipeline=indexing_pipeline.to_retrieval_pipeline())
    print(
        rag_pipeline.run(
            "Who wrote research papers abouts the impacts of digitalization and societal changes on energy transition ?"
        )
    )
