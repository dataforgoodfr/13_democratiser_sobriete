from typing import Optional, List
from sentence_transformers import CrossEncoder

from kotaemon.rerankings import BaseReranking
from kotaemon.base import Document, Param

class CrossEncoderReranking(BaseReranking):
    """Uses a local Hugging Face Cross-Encoder model for reranking."""

    model_name: str = Param(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",  # Default cross-encoder model
        help="Hugging Face Cross-Encoder model ID.",
    )
    is_truncated: Optional[bool] = Param(True, help="Whether to truncate the inputs.")
    max_tokens: Optional[int] = Param(
        512, help="Maximum tokens per document (truncated if needed)."
    )
    batch_size: Optional[int] = Param(6, help="Batch size for processing documents.")
    model: Optional[CrossEncoder] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = str(self.model_name)  # Ensure it's a valid string
        self.load_model()  # Load model

    def load_model(self):
        """Loads the Cross-Encoder model locally."""
        self.model = CrossEncoder(self.model_name)  # Initialize model

    def client(self, query: str, texts: List[str]) -> List[float]:
        """Computes relevance scores using the Cross-Encoder model locally."""
        if self.is_truncated:
            texts = [text[: self.max_tokens] for text in texts]  # Truncate if needed

        pairs = [(query, text) for text in texts]  # Create query-document pairs
        scores = self.model.predict(pairs)  # Get relevance scores
        return scores

    def run(self, documents: List[Document], query: str) -> List[Document]:
        """Re-ranks documents based on relevance scores from Cross-Encoder."""
        if not documents:
            return []

        ranked_docs = []
        num_batch = max(len(documents) // self.batch_size, 1)

        for i in range(num_batch):
            if i == num_batch - 1:
                mini_batch = documents[self.batch_size * i :]
            else:
                mini_batch = documents[self.batch_size * i : self.batch_size * (i + 1)]

            _docs = [doc.content for doc in mini_batch]
            scores = self.client(query, _docs)

            for doc, score in zip(mini_batch, scores):
                doc.metadata["reranking_score"] = score
                ranked_docs.append(doc)

        ranked_docs.sort(key=lambda x: x.metadata["reranking_score"], reverse=True)
        return ranked_docs
