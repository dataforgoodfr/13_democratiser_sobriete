
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class TfidfEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=20000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english"
        )

    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X)


class SentenceBERTEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2", batch_size=64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y=None):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed")
        self.model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X):
        embeddings = self.model.encode(
            X,
            batch_size=self.batch_size,
            show_progress_bar=True
        )
        return np.asarray(embeddings)
