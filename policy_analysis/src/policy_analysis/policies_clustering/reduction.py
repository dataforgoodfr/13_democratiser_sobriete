from sklearn.base import BaseEstimator, TransformerMixin
import umap


class UMAPReducer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10, n_neighbors=15, min_dist=0.0, random_state=42):
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
            random_state=self.random_state
        ).fit(X_dense)
        return self

    def transform(self, X):
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        if self.reducer is None:
            raise ValueError("UMAPReducer must be fitted first")
        return self.reducer.transform(X_dense)