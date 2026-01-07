import numpy as np
import matplotlib.pyplot as plt


def plot_clusters_2d(X_2d: np.ndarray, labels, title="HDBSCAN Clusters"):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, s=5)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig("cluster.png")
    plt.close()
