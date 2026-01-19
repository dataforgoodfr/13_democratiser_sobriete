import typing as t

import hdbscan
from bertopic import BERTopic
from bertopic.representation import BaseRepresentation, KeyBERTInspired
from sentence_transformers import SentenceTransformer
from umap import UMAP


class BERTopicModelTrainer:
    def __init__(
        self,
        embedding_model: t.Any = SentenceTransformer(
            "tomaarsen/static-retrieval-mrl-en-v1", truncate_dim=256
        ),
        umap_n_components=10,
        umap_n_neighbors=30,
        umap_random_state=42,
        umap_metric="cosine",
        hdbscan_min_samples=20,
        hdbscan_min_cluster_size=20,
        representation_model: BaseRepresentation = KeyBERTInspired(),
    ):
        self.embedding_model = embedding_model
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_random_state = umap_random_state
        self.umap_metric = umap_metric
        self.hdbscan_min_samples = hdbscan_min_samples
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.representation_model = representation_model
        self.umap_model = None
        self.hdbscan_model = None
        self.topic_model = None

    def initialize_models(self, vectorizer_model):
        self.umap_model = UMAP(
            n_components=self.umap_n_components,
            n_neighbors=self.umap_n_neighbors,
            random_state=self.umap_random_state,
            metric=self.umap_metric,
            verbose=True,
        )
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_samples=self.hdbscan_min_samples,
            gen_min_span_tree=True,
            prediction_data=True,
            min_cluster_size=self.hdbscan_min_cluster_size,
        )
        self.vectorizer_model = vectorizer_model

    def train(self, docs: t.List[str], embeddings: t.Any = None):
        self.initialize_models(vectorizer_model=self.vectorizer_model)
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            verbose=True,
            representation_model=self.representation_model,
        ).fit(docs, embeddings=embeddings)
        return self.topic_model

    def save_model(self, model_path: str, **kwargs):
        if self.topic_model is not None:
            self.topic_model.save(path=model_path, **kwargs)

    def load_model(self, model_path: str):
        self.topic_model = BERTopic.load(path=model_path)
        return self.topic_model
