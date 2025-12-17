from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

import hdbscan

from policy_analysis.policies_clustering.embeddings import TfidfEmbedder, SentenceBERTEmbedder
from policy_analysis.policies_clustering.reduction import UMAPReducer
from policy_analysis.policies_clustering.report import plot_clusters_2d


class HDBSCANClusterer(BaseEstimator):
    def __init__(self, min_cluster_size=50, min_samples=None, metric="euclidean"):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.clusterer = None

    def fit(self, X, y=None):
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method="eom",
            core_dist_n_jobs=-1
        )
        self.labels_ = self.clusterer.fit_predict(X)
        self.probabilities_ = self.clusterer.probabilities_
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_


def build_hdbscan_pipeline(
    embedding="tfidf",
    n_components=10,
    min_cluster_size=50
):
    steps = []

    if embedding == "tfidf":
        steps.append(("embed", TfidfEmbedder()))
        steps.append(("svd", TruncatedSVD(n_components=n_components, random_state=42)))
        steps.append(("norm", Normalizer(copy=False)))

    elif embedding == "sbert":
        steps.append(("embed", SentenceBERTEmbedder()))

    else:
        raise ValueError("embedding must be 'tfidf' or 'sbert'")

    steps.append(("umap", UMAPReducer(n_components=n_components)))
    steps.append(("cluster", HDBSCANClusterer(min_cluster_size=min_cluster_size)))

    return Pipeline(steps)


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    root = Path().cwd()
    fp = root / "data/conclusions&pollitiques_synthetiques.jsonl"
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    df = pd.read_json(fp, lines=True)
    texts = df["response"].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    pipe = build_hdbscan_pipeline(
        embedding="sbert",
        n_components=5,
        min_cluster_size=2
    )

    pipe.fit(embeddings)

    umap_model = UMAP(n_components=2, n_neighbors=15, random_state=42,
                    metric="cosine", verbose=True)
    labels = pipe.named_steps["cluster"].labels_
    reduced_embeddings_2d = umap_model.fit_transform(embeddings)

    plot_clusters_2d(reduced_embeddings_2d, labels)
