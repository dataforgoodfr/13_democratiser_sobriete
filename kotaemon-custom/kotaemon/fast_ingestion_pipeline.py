import os
from typing import List

from kotaemon.base import Document, Param, lazy
from kotaemon.indices import VectorIndexing
from kotaemon.embeddings import OpenAIEmbeddings
from kotaemon.storages import LanceDBDocumentStore, QdrantVectorStore
#

class IndexingPipeline(VectorIndexing):

    COLLECTION_NAME = 'WSL_collection_test'

    vector_store: QdrantVectorStore = Param(
        lazy(QdrantVectorStore).withx(url="http://localhost:6333",
                                      collection_name=COLLECTION_NAME,
                                      api_key="None",
                                      ),
        ignore_ui=True,
    )
    doc_store: LanceDBDocumentStore = Param(
        lazy(LanceDBDocumentStore).withx(
            path= './ktem_app_data/user_data/docstore',
            collection_name= COLLECTION_NAME),
        ignore_ui=True,
    )
    embedding: OpenAIEmbeddings = Param(
        lazy(OpenAIEmbeddings).withx(
        base_url = "http://172.17.0.1:11434/v1/",
        model="snowflake-arctic-embed2",
        api_key= "ollama"
    ),
    ignore_ui=True,
    )


    def run(self, text: str | list[str], metadatas: dict | list[dict] | None) -> Document:
        """Normally, this indexing pipeline returns nothing. For demonstration,
        we want it to return something, so let's return the number of documents
        in the vector store
        """
        # --- LOGIC INGESTION PIPELINE ----

        #TODO

        # core extraction
        # inference metadatas
        # summarization
        # 

        # --- END - LOGIC INGESTION PIPELINE ----

        # Final ingestion (=> to 3 databases)
        super().run(text, metadatas)

        return Document(self.vector_store._collection.count())
    

pipeline = IndexingPipeline()

pipeline.run(text=["feedback to further \nimprove his skills and knowledge", "yes ok yes it's a test ok"], 
             metadatas=[{"key_1":"test"}, {"key_1":"test"}])

