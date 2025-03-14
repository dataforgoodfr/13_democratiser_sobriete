import math
from typing import List, Optional, Dict, ClassVar, Literal


class RetrievalMetricResult:
    def __init__(self, score: float, metadata: Optional[Dict] = None):
        self.score = score
        self.metadata = metadata or {}


class BaseRetrievalMetric:
    metric_name: ClassVar[str]

    def compute(
        self,
        query: Optional[str] = None,
        expected_ids: Optional[List[str]] = None,
        retrieved_ids: Optional[List[str]] = None,
    ) -> RetrievalMetricResult:
        raise NotImplementedError


class HitRate(BaseRetrievalMetric):
    metric_name = "hit_rate"
    use_granular_hit_rate: bool = False

    def compute(
        self, expected_ids: List[str], retrieved_ids: List[str]
    ) -> RetrievalMetricResult:
        if not retrieved_ids or not expected_ids:
            raise ValueError("Retrieved ids and expected ids must be provided")

        if self.use_granular_hit_rate:
            hits = sum(1 for doc_id in retrieved_ids if doc_id in expected_ids)
            score = hits / len(expected_ids)
        else:
            score = 1.0 if any(id in expected_ids for id in retrieved_ids) else 0.0

        return RetrievalMetricResult(score=score)


class MRR(BaseRetrievalMetric):
    metric_name = "mrr"
    use_granular_mrr: bool = False

    def compute(
        self, expected_ids: List[str], retrieved_ids: List[str]
    ) -> RetrievalMetricResult:
        if not retrieved_ids or not expected_ids:
            raise ValueError("Retrieved ids and expected ids must be provided")

        if self.use_granular_mrr:
            reciprocal_rank_sum = sum(
                1.0 / (i + 1)
                for i, doc_id in enumerate(retrieved_ids)
                if doc_id in expected_ids
            )
            relevant_docs_count = sum(
                1 for doc_id in retrieved_ids if doc_id in expected_ids
            )
            score = (
                reciprocal_rank_sum / relevant_docs_count
                if relevant_docs_count > 0
                else 0.0
            )
        else:
            score = next(
                (
                    1.0 / (i + 1)
                    for i, id in enumerate(retrieved_ids)
                    if id in expected_ids
                ),
                0.0,
            )

        return RetrievalMetricResult(score=score)


class Precision(BaseRetrievalMetric):
    metric_name = "precision"

    def compute(
        self, expected_ids: List[str], retrieved_ids: List[str]
    ) -> RetrievalMetricResult:
        if not retrieved_ids or not expected_ids:
            raise ValueError("Retrieved ids and expected ids must be provided")

        precision = len(set(retrieved_ids) & set(expected_ids)) / len(retrieved_ids)
        return RetrievalMetricResult(score=precision)


class Recall(BaseRetrievalMetric):
    metric_name = "recall"

    def compute(
        self, expected_ids: List[str], retrieved_ids: List[str]
    ) -> RetrievalMetricResult:
        if not retrieved_ids or not expected_ids:
            raise ValueError("Retrieved ids and expected ids must be provided")

        recall = len(set(retrieved_ids) & set(expected_ids)) / len(expected_ids)
        return RetrievalMetricResult(score=recall)


class AveragePrecision(BaseRetrievalMetric):
    metric_name = "ap"

    def compute(
        self, expected_ids: List[str], retrieved_ids: List[str]
    ) -> RetrievalMetricResult:
        if not retrieved_ids or not expected_ids:
            raise ValueError("Retrieved ids and expected ids must be provided")

        relevant_count, total_precision = 0, 0.0
        for i, retrieved_id in enumerate(retrieved_ids, start=1):
            if retrieved_id in expected_ids:
                relevant_count += 1
                total_precision += relevant_count / i

        score = total_precision / len(expected_ids)
        return RetrievalMetricResult(score=score)


class NDCG(BaseRetrievalMetric):
    metric_name = "ndcg"
    mode: Literal["linear", "exponential"] = "linear"

    def compute(
        self, expected_ids: List[str], retrieved_ids: List[str]
    ) -> RetrievalMetricResult:
        if not retrieved_ids or not expected_ids:
            raise ValueError("Retrieved ids and expected ids must be provided")

        def discounted_gain(rel: float, i: int, mode: str) -> float:
            if rel == 0:
                return 0
            if mode == "linear":
                return rel / math.log2(i + 1)
            elif mode == "exponential":
                return (2**rel - 1) / math.log2(i + 1)

        expected_set = set(expected_ids)
        dcg = sum(
            discounted_gain(docid in expected_set, i, self.mode)
            for i, docid in enumerate(retrieved_ids, start=1)
        )
        idcg = sum(
            discounted_gain(True, i, self.mode) for i in range(1, len(expected_ids) + 1)
        )
        score = dcg / idcg if idcg > 0 else 0.0
        return RetrievalMetricResult(score=score)


METRIC_REGISTRY: Dict[str, BaseRetrievalMetric] = {
    "hit_rate": HitRate(),
    "mrr": MRR(),
    "precision": Precision(),
    "recall": Recall(),
    "ap": AveragePrecision(),
    "ndcg": NDCG(),
}


def resolve_metrics(metrics: List[str]) -> List[BaseRetrievalMetric]:
    """Resolve metrics from list of metric names."""
    for metric in metrics:
        if metric not in METRIC_REGISTRY:
            raise ValueError(f"Invalid metric name: {metric}")

    return [METRIC_REGISTRY[metric] for metric in metrics]
