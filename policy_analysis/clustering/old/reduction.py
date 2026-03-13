from sklearn.base import BaseEstimator, TransformerMixin
import umap


class UMAPReducer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that performs dimensionality reduction using UMAP.
    This class wraps the UMAP (Uniform Manifold Approximation and Projection) algorithm
    to reduce high-dimensional data while preserving local and global structure.

    Parameters :

    **n_components**: int, default=10
        The dimension of the space to embed into. This defaults to 10 for visualization
        purposes, but can be set to any integer value in the range 2 to 100.

    **n_neighbors**: int, default=15
        The size of the local neighborhood (in terms of number of neighboring sample points)
        used for manifold approximation. Larger values result in more global views of the
        manifold, while smaller values result in more local data being preserved. Generally,
        values should be in the range 2 to 100.

    **min_dist**: float, default=0.0
        The effective minimum distance between embedded points. Smaller values will result
        in a more clustered/clumped embedding where nearby points on the manifold are drawn
        closer together, while larger values will result in a more even dispersal of points.
        The value should be set relative to the spread value. Generally, values should be
        in the range 0.0 to 0.99.

    **random_state**: int, default=42
        Random seed used for the random number generator for reproducibility.
        Do not use if you want to use paralellism (njobs > 1).
    """
    
    def __init__(self, n_components=10, n_neighbors=15, min_dist=0.0, random_state=None):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.reducer = None

    def fit(self, X, y=None):
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        self.reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric="cosine",
            random_state=self.random_state,
        ).fit(X_dense)
        return self

    def transform(self, X):
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        if self.reducer is None:
            raise ValueError("UMAPReducer must be fitted first")
        return self.reducer.transform(X_dense)
