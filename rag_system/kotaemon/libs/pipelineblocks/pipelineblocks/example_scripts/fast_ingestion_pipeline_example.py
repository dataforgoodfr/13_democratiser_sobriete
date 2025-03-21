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

from pipelineblocks.extraction.pdfextractionblock.pdf_to_markdown import PdfExtractionToMarkdownBlock
from pipelineblocks.llm.ingestionblock.openai import OpenAIMetadatasLLMInference

# Taxonomy example (pydantic schema)
from taxonomy.paper_taxonomy import PaperTaxonomy


# (For dev) temporary settings
OLLAMA_DEPLOYMENT = 'docker'
VECTOR_STORE_DEPLOYMENT = 'docker'

PDF_FOLDER = "./pipeline_scripts/pdf_test/"

# (For dev)(temporary) ------------- #

ollama_host = '172.17.0.1' if OLLAMA_DEPLOYMENT == 'docker' else 'localhost'
qdrant_host = '172.17.0.1' if VECTOR_STORE_DEPLOYMENT == 'docker' else 'localhost'


class IndexingPipeline(VectorIndexing):

    # --- Different blocks (pipeline blocks library) --- 

    pdf_extraction_block : PdfExtractionToMarkdownBlock = Param(
        lazy(PdfExtractionToMarkdownBlock).withx(
        )
    )

    # At least, one taxonomy = one llm_inference_block
    # (Multiply the number of llm_inference_block when you need handle more than one taxonomy
    metadatas_llm_inference_block : OpenAIMetadatasLLMInference = Param(
        lazy(OpenAIMetadatasLLMInference).withx(
            llm = ChatOpenAI(
                base_url=f"http://{ollama_host}:11434/v1/",
                model="gemma2:2b",
                api_key="ollama",
                ),
            taxonomy = PaperTaxonomy
            )
    )

    # --- Final Kotaemon ingestion ----

    vector_store: QdrantVectorStore = Param(
        lazy(QdrantVectorStore).withx(
            url=f"http://{qdrant_host}:6333",
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
            base_url=f"http://{ollama_host}:11434/v1/",
            model="snowflake-arctic-embed2",
            api_key="ollama",
        ),
        ignore_ui=True,
    )

    pdf_path: str
    
    def run(self, pdf_name: str) -> None:
        """
        ETL pipeline for a single pdf file
        1. Extract text and taxonomy from pdf
        2. Transform taxonomy (flattening)
        3. Ingest text and taxonomy into the vector store

        Return nothing
        """

        text_md = self.pdf_extraction_block.run(self.pdf_path + pdf_name, method = 'group_all')

        metadatas = self.metadatas_llm_inference_block.run(text_md,  
                                                        doc_type  = 'entire_doc', 
                                                        inference_type = 'scientific')
        

        metadatas_json = metadatas.model_dump()
    
        super().run(text=[text_md], metadatas=[metadatas_json])

        return None

# ----------------Retrive (Crash) version -------------- #
# TODO Convert it with refacto pipelineblocks too #
# TODO Build another script file with it... Here it's just a draft

class RetrievePipeline(BaseComponent):
    """
    from simple_pipeline.py, a better RAG pipeline must exist somewhere

    
    """

    llm: ChatOpenAI = ChatOpenAI.withx(
        base_url=f"http://{ollama_host}:11434/v1/",
        model="gemma2:2b",
        api_key="ollama",
    )

    retrieval_pipeline: VectorRetrieval

    def run(self, text: str) -> LLMInterface:
        matched_texts: List[Document] = self.retrieval_pipeline(text)
        return self.llm("\n".join(map(str, matched_texts)))


if __name__ == "__main__":


    indexing_pipeline = IndexingPipeline(pdf_path=PDF_FOLDER)
    indexing_pipeline.run("1-s2.0-S2211467X23001748-main.pdf")
    indexing_pipeline.run("1-s2.0-S0094119008001095-main.pdf")



    # Just for test

    rag_pipeline = RetrievePipeline(retrieval_pipeline=indexing_pipeline.to_retrieval_pipeline())
    print(
        rag_pipeline.run(
            "Who wrote research papers abouts the impacts of digitalization and societal changes on energy transition ?"
        )
    )
