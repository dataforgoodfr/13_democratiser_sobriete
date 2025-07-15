import os
from typing import Any, List, Optional, cast

from .base import LlamaIndexVectorStore

VECTORSTORE_URL = os.getenv("VECTOSTORE_URL", "")
default_api_key = os.getenv("API_KEY", "")

class QdrantVectorStore(LlamaIndexVectorStore):
    _li_class = None

    def _get_li_class(self):
        try:
            from llama_index.vector_stores.qdrant import (
                QdrantVectorStore as LIQdrantVectorStore,
            )
        except ImportError:
            raise ImportError(
                "Please install missing package: "
                "'pip install llama-index-vector-stores-qdrant'"
            )

        return LIQdrantVectorStore

    def __init__(
        self,
        collection_name,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        client_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ):
        self._collection_name = collection_name
        self._url = VECTORSTORE_URL
        self._api_key = default_api_key
        self._client_kwargs = client_kwargs
        self._kwargs = kwargs
        print(f"url: {self._url}")

        super().__init__(
            collection_name=collection_name,
            url=VECTORSTORE_URL,
            api_key=default_api_key,
            client_kwargs=client_kwargs,
            **kwargs,
        )
        from llama_index.vector_stores.qdrant import (
            QdrantVectorStore as LIQdrantVectorStore,
        )

        self._client = cast(LIQdrantVectorStore, self._client)

    def delete(self, ids: List[str], **kwargs):
        """Delete vector embeddings from vector stores

        Args:
            ids: List of ids of the embeddings to be deleted
            kwargs: meant for vectorstore-specific parameters
        """
        from qdrant_client import models

        self._client.client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(
                points=ids,
            ),
            **kwargs,
        )

    def drop(self):
        """Delete entire collection from vector stores"""
        self._client.client.delete_collection(self._collection_name)

    def count(self) -> int:
        return self._client.client.count(
            collection_name=self._collection_name, exact=True
        ).count

    def __persist_flow__(self):
        return {
            "collection_name": self._collection_name,
            "url": self._url,
            "api_key": self._api_key,
            "client_kwargs": self._client_kwargs,
            **self._kwargs,
        }
