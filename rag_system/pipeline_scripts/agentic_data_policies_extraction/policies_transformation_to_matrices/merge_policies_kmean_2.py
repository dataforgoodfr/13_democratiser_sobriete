from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from typing import List, Tuple
import pandas as pd

class CONFIG_Kmean:
    n_words: int = 3
    max_clusters: int = 500
    random_seed: int = 42

CFG_Kmean = CONFIG_Kmean()    

def get_cluster_keywords(cluster_tfidf: np.ndarray,
                        feature_names: np.ndarray,
                        n_words: int = 3) -> str:
    """
    Extract the most representative words for a cluster.

    Args:
        cluster_tfidf: Summed TF-IDF vectors for the cluster.
        feature_names: Array of feature names from the vectorizer.
        n_words: Number of top words to return.

    Returns:
        String of top words separated by spaces.
    """
    top_indices = cluster_tfidf.argsort()[-n_words:][::-1]
    return ' '.join([feature_names[i] for i in top_indices])

def cluster_from_kmeans_2(texts: List[str],
                       max_clusters: int = 50) -> Tuple[KMeans, np.ndarray, np.ndarray]:
    """
    Perform K-means clustering on text data.

    Args:
        texts: List of text documents to cluster.
        max_clusters: Maximum number of clusters.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (fitted KMeans model, TF-IDF matrix, feature names).
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=max_clusters, random_state=CFG_Kmean.random_seed)
    kmeans.fit(X)
    return kmeans, X, vectorizer.get_feature_names_out()

def merge_policies_kmeans_2(df: pd.DataFrame,
                        texts: pd.Series,
                        max_clusters: int = CFG_Kmean.max_clusters) -> pd.DataFrame:
    """
    Add cluster names based on representative keywords to a DataFrame.

    Args:
        df: DataFrame containing the data.
        texts: Series of text documents to cluster.
        max_clusters: Maximum number of clusters.

    Returns:
        DataFrame with added cluster columns.
    """
    kmeans, X, feature_names = cluster_from_kmeans_2(texts, max_clusters)

    # Add cluster labels to DataFrame
    df['cluster_id_KMEAN'] = kmeans.labels_

    # Get top words for each cluster
    cluster_keywords = {}
    for cluster_id in range(max_clusters):
        cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
        if len(cluster_indices) > 0:  # Only process non-empty clusters
            cluster_tfidf = X[cluster_indices].sum(axis=0).A1
            cluster_keywords[cluster_id] = get_cluster_keywords(cluster_tfidf, feature_names)

    # Map cluster IDs to keywords, handle missing clusters
    df['policy_KMEAN'] = df['cluster_id_KMEAN'].map(lambda cluster_id: cluster_keywords.get(cluster_id, f"cluster_{cluster_id}"))

    return df

def prepare_evaluation_features(df: pd.DataFrame,
                               text_column: str = 'policy') -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and labels for clustering evaluation.

    Args:
        df: DataFrame containing the data
        text_column: Name of the column containing text to evaluate

    Returns:
        Tuple of (TF-IDF matrix, original cluster labels)
    """
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df[text_column])
    labels = df["cluster_id"].values

    return X, labels