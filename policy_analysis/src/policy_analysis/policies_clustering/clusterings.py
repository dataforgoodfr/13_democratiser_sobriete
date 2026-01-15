from typing import Literal
from loguru import logger
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan

from policy_analysis.policies_clustering.embeddings import TfidfEmbedder, SentenceBERTEmbedder
from policy_analysis.policies_clustering.reduction import UMAPReducer
from policy_analysis.policies_clustering.report import plot_clusters_2d


class HDBSCANClusterer(BaseEstimator):
    def __init__(self, min_cluster_size: int=50, min_samples: int | None = None, metric="euclidean"):
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


class KMeansClusterer(BaseEstimator):
    def __init__(self, n_clusters: int = 8, random_state: int = 42, max_iter: int = 300):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.clusterer = None

    def fit(self, X, y=None):
        self.clusterer = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=10
        )
        self.labels_ = self.clusterer.fit_predict(X)
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_


def evaluate_clustering(X, labels, metric: str = "silhouette") -> float:
    """
    Evaluate clustering quality using various metrics.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        metric: One of 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'coherence'
    
    Returns:
        Score (higher is better for all metrics)
    """
    mask = labels != -1
    if mask.sum() < 2:
        return -1.0
    
    X_filtered = X[mask]
    labels_filtered = labels[mask]

    if len(np.unique(labels_filtered)) < 2:
        return -1.0
    
    if metric == "silhouette":
        return silhouette_score(X_filtered, labels_filtered)
    elif metric == "coherence":
        coherence = 0.0
        unique_labels = np.unique(labels_filtered)
        
        for label in unique_labels:
            cluster_mask = labels_filtered == label
            cluster_points = X_filtered[cluster_mask]
            
            if len(cluster_points) > 1:
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(cluster_points)
                cluster_coherence = (similarities.sum() - len(cluster_points)) / (len(cluster_points) * (len(cluster_points) - 1))
                coherence += cluster_coherence * len(cluster_points)
        
        coherence /= len(labels_filtered)
        return coherence
    else:
        raise ValueError(f"Unknown metric: {metric}")


def find_optimal_clusters(
    X,
    cluster_range: list[int] | None= None,
    metric: str = "silhouette",
    random_state: int = 42
) -> dict:
    """
    Find optimal number of clusters using cross-validation.
    
    Args:
        X: Feature matrix
        cluster_range: Range of cluster numbers to try
        metric: Evaluation metric ('silhouette', 'coherence')
        random_state: Random state for reproducibility
    
    Returns:
        Dictionary with results including best_n_clusters and scores
    """
    scores = []

    cluster_range = cluster_range or [50, 100, 150, 200]
    
    for n_clusters in cluster_range:
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = clusterer.fit_predict(X)
        score = evaluate_clustering(X, labels, metric=metric)
        scores.append(score)
        logger.info(f"n_clusters={n_clusters}, {metric}={score:.4f}")
    
    best_idx = np.argmax(scores)
    best_n_clusters = list(cluster_range)[best_idx]
    
    return {
        "best_n_clusters": best_n_clusters,
        "best_score": scores[best_idx],
        "all_scores": scores,
        "cluster_range": list(cluster_range)
    }


def build_clustering_pipeline(
    embedding: Literal["tfidf", "sbert"] = "sbert",
    clustering: Literal["hdbscan", "kmeans"] = "hdbscan",
    n_components: int = 10,
    min_cluster_size: int = 50,
    min_samples: int | None = None,
    n_clusters: int = 8,
    optimize_clusters: bool = False,
    cluster_range: range = range(2, 21),
    optimization_metric: str = "silhouette"
):
    """
    Build a clustering pipeline with flexible options.
    
    Args:
        embedding: Type of embedding ('tfidf' or 'sbert')
        clustering: Type of clustering ('hdbscan' or 'kmeans')
        n_components: Number of dimensions for reduction
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples for HDBSCAN
        n_clusters: Number of clusters for KMeans (if not optimizing)
        optimize_clusters: Whether to find optimal n_clusters via CV (KMeans only)
        cluster_range: Range of clusters to test if optimizing
        optimization_metric: Metric for optimization ('silhouette', 'calinski_harabasz', 'davies_bouldin', 'coherence')
    
    Returns:
        Pipeline object and optional optimization results
    """
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

    if clustering == "hdbscan":
        steps.append(("cluster", HDBSCANClusterer(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )))
    elif clustering == "kmeans":
        steps.append(("norm", Normalizer(copy=False)))
        steps.append(("cluster", KMeansClusterer(
            n_clusters=n_clusters,
            random_state=42
        )))
    else:
        raise ValueError("clustering must be 'hdbscan' or 'kmeans'")
    
    pipeline = Pipeline(steps)
    
    return pipeline


def build_hdbscan_pipeline(
    embedding="tfidf",
    n_components=10,
    min_cluster_size=50,
    min_samples=None
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
    steps.append(("cluster", HDBSCANClusterer(min_cluster_size=min_cluster_size, min_samples=min_samples)))

    return Pipeline(steps)


def run_clustering_experiment(
    dataset_name: str = "madoss/wsl_library_filtered",
    dataset_split: str = "train",
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    embedding_type: Literal["tfidf", "sbert"] = "sbert",
    clustering_method: Literal["hdbscan", "kmeans", "kmeans_cv"] = "kmeans_cv",
    n_components: int = 10,
    n_clusters: int = 10,
    min_cluster_size: int = 100,
    min_samples: int | None = 20,
    cluster_range: list[int] | None = None,
    cv_metric: str = "silhouette",
    output_dir: str | None = None,
    visualize: bool = True
):
    """
    Run a complete clustering experiment with configurable parameters.
    
    Args:
        dataset_name: HuggingFace dataset name
        dataset_split: Dataset split to use
        embedding_model: SentenceTransformer model name
        embedding_type: Type of embedding ('tfidf' or 'sbert')
        clustering_method: Clustering method ('hdbscan', 'kmeans', or 'kmeans_cv')
        n_components: Number of dimensions for UMAP reduction
        n_clusters: Number of clusters for KMeans (if not using CV)
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples for HDBSCAN
        cluster_range_min: Minimum clusters to test in CV
        cluster_range_max: Maximum clusters to test in CV
        cv_metric: Metric for CV optimization
        output_dir: Directory to save results
        visualize: Whether to create visualizations
    
    Returns:
        Dictionary with results including labels, pipeline, and CV results if applicable
    """
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from pathlib import Path

    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=dataset_split)
    texts = dataset["text"]
    logger.info(f"Loaded {len(texts)} texts")
    cluster_range = cluster_range or [50, 100, 150, 200]
    if embedding_type == "sbert":
        logger.info(f"Generating embeddings with {embedding_model}...")
        if dataset_name == "madoss/wsl_library_filtered":
            dataset.set_format(type="numpy", columns=["embedding"])
            embeddings = np.array(dataset["embedding"])
        else:
            model = SentenceTransformer(embedding_model)
            embeddings = model.encode(texts, show_progress_bar=True)
        logger.info(f"Embeddings shape: {embeddings.shape}")
    else:
        embeddings = texts
    
    results = {"texts": texts, "embeddings": embeddings if isinstance(embeddings, np.ndarray) else None}
    
    if clustering_method == "hdbscan":
        logger.info("\n=== Running HDBSCAN ===")
        pipe = build_clustering_pipeline(
            embedding=embedding_type,
            clustering="hdbscan",
            n_components=n_components,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        pipe.fit(embeddings)
        labels = pipe.named_steps["cluster"].labels_
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"HDBSCAN found {n_clusters_found} clusters")
        results.update({"labels": labels, "pipeline": pipe, "n_clusters": n_clusters_found})
        
    elif clustering_method == "kmeans":
        logger.info(f"\n=== Running KMeans (n_clusters={n_clusters}) ===")
        pipe = build_clustering_pipeline(
            embedding=embedding_type,
            clustering="kmeans",
            n_components=n_components,
            n_clusters=n_clusters
        )
        pipe.fit(embeddings)
        labels = pipe.named_steps["cluster"].labels_
        logger.info(f"KMeans created {len(set(labels))} clusters")
        results.update({"labels": labels, "pipeline": pipe, "n_clusters": n_clusters})
        
    elif clustering_method == "kmeans_cv":
        logger.info("\n=== Running KMeans with CV optimization ===")

        pipe_for_reduction = build_clustering_pipeline(
            embedding=embedding_type,
            clustering="kmeans",
            n_components=n_components,
            n_clusters=2
        )
        
        reduced_embeddings = embeddings
        for name, step in pipe_for_reduction.steps[:-1]:
            reduced_embeddings = step.fit_transform(reduced_embeddings)

        cv_results = find_optimal_clusters(
            reduced_embeddings,
            cluster_range=cluster_range,
            metric=cv_metric
        )
        
        logger.info(f"Optimal number of clusters: {cv_results['best_n_clusters']}")
        logger.info(f"Best {cv_metric} score: {cv_results['best_score']:.4f}")

        pipe = build_clustering_pipeline(
            embedding=embedding_type,
            clustering="kmeans",
            n_components=n_components,
            n_clusters=cv_results['best_n_clusters']
        )
        pipe.fit(embeddings)
        labels = pipe.named_steps["cluster"].labels_
        
        results.update({
            "labels": labels,
            "pipeline": pipe,
            "n_clusters": cv_results['best_n_clusters'],
            "cv_results": cv_results
        })

    if visualize:
        logger.info("\n=== Creating visualization ===")
        umap_2d = UMAP(n_components=2, n_neighbors=15, random_state=42, metric="cosine", verbose=True)
        reduced_2d = (umap_2d.fit_transform(embeddings if isinstance(embeddings, np.ndarray) else
        results["pipeline"].transform(embeddings)[:len(embeddings)])
        )
        
        title = f"{clustering_method.upper()} Clustering"
        if clustering_method == "kmeans_cv":
            title += f" (optimal n={results['n_clusters']})"
        elif clustering_method == "kmeans":
            title += f" (n={n_clusters})"
        
        plot_clusters_2d(reduced_2d, labels, title=title)
        results["reduced_2d"] = reduced_2d

    if output_dir:
        from pathlib import Path
        import pickle
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        np.save(output_path / "labels.npy", labels)
        logger.info(f"Saved labels to {output_path / 'labels.npy'}")

        if "cv_results" in results:
            with open(output_path / "cv_results.pkl", "wb") as f:
                pickle.dump(results["cv_results"], f)
            logger.info(f"Saved CV results to {output_path / 'cv_results.pkl'}")
    
    return results


def main():
    """CLI entry point for clustering experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run clustering experiments on text datasets")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="madoss/wsl_library_filtered",
                       help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to use")
    
    # Embedding arguments
    parser.add_argument("--embedding-type", type=str, choices=["tfidf", "sbert"], default="sbert",
                       help="Type of embedding to use")
    parser.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                       help="SentenceTransformer model name (for sbert)")
    
    # Clustering arguments
    parser.add_argument("--clustering", type=str, choices=["hdbscan", "kmeans", "kmeans_cv"], 
                       default="kmeans_cv", help="Clustering method")
    parser.add_argument("--n-components", type=int, default=10,
                       help="Number of UMAP components")
    parser.add_argument("--n-clusters", type=int, default=10,
                       help="Number of clusters for KMeans (if not using CV)")
    
    # HDBSCAN arguments
    parser.add_argument("--min-cluster-size", type=int, default=100,
                       help="Minimum cluster size for HDBSCAN")
    parser.add_argument("--min-samples", type=int, default=20,
                       help="Minimum samples for HDBSCAN")
    
    # CV arguments
    parser.add_argument("--cluster-range", type=int, nargs="+", default=[50, 100, 150, 200],
                       help="Minimum clusters to test in CV")

    parser.add_argument("--cv-metric", type=str, 
                       choices=["silhouette", "calinski_harabasz", "coherence"],
                       default="silhouette", help="Metric for CV optimization")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save results")
    parser.add_argument("--no-visualize", action="store_true",
                       help="Disable visualization")
    
    args = parser.parse_args()

    results = run_clustering_experiment(
        dataset_name=args.dataset,
        dataset_split=args.split,
        embedding_model=args.embedding_model,
        embedding_type=args.embedding_type,
        clustering_method=args.clustering,
        n_components=args.n_components,
        n_clusters=args.n_clusters,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_range=args.cluster_range,
        cv_metric=args.cv_metric,
        output_dir=args.output_dir,
        visualize=not args.no_visualize
    )
    
    logger.info("\n=== Experiment Complete ===")
    logger.info(f"Total clusters found: {results['n_clusters']}")
    if "cv_results" in results:
        logger.info(f"Best {args.cv_metric} score: {results['cv_results']['best_score']:.4f}")

    return results


if __name__ == "__main__":
    main()
