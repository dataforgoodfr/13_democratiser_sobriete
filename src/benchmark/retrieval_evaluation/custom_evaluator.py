import os
import json
import asyncio
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from kotaemon.indices import VectorRetrieval
from kotaemon.base import RetrievedDocument
from benchmark.retrieval_evaluation.metrics import BaseRetrievalMetric, RetrievalMetricResult
from llama_index.core.bridge.pydantic import BaseModel

class RetrievalEvalMode:
    """Retrieval evaluation mode constants."""

    TEXT = "text"


class RetrievalEvalResult:
    """Stores evaluation results for retrieval tasks."""

    def __init__(
        self,
        query: str,
        expected_ids: List[str],
        expected_texts: Optional[List[str]],
        retrieved_ids: List[str],
        retrieved_texts: List[str],
        mode: str,
        metric_dict: Dict[str, dict[str, RetrievalMetricResult]],
    ):
        self.query = query
        self.expected_ids = expected_ids
        self.expected_texts = expected_texts
        self.retrieved_ids = retrieved_ids
        self.retrieved_texts = retrieved_texts
        self.mode = mode
        self.metric_dict = metric_dict


class EmbeddingQAFinetuneDataset(BaseModel):
    """Embedding QA Finetuning Dataset.

    Args:
        queries (Dict[str, str]): Dict id -> query.
        corpus (Dict[str, str]): Dict id -> string.
        relevant_docs (Dict[str, List[str]]): Dict query id -> list of doc ids.
    """

    queries: Dict[str, str]
    corpus: Dict[str, str]
    relevant_docs: Dict[str, List[str]]
    mode: str = "text"

    @property
    def query_docid_pairs(self) -> List[Tuple[str, List[str]]]:
        """Get query, relevant doc ids."""
        return [
            (query, self.relevant_docs[query_id])
            for query_id, query in self.queries.items()
        ]

    def save_json(self, path: str) -> None:
        """Save the dataset to a JSON file.

        Args:
            path (str): The file path to save the JSON.
        """
        data = {
            "queries": self.queries,
            "corpus": self.corpus,
            "relevant_docs": self.relevant_docs,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "EmbeddingQAFinetuneDataset":
        """Load the dataset from a JSON file.

        Args:
            path (str): The file path to load the JSON from.

        Returns:
            EmbeddingQAFinetuneDataset: The loaded dataset.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def save_results(
    name: str,
    eval_results: List[RetrievalEvalResult],
    output_file: str = None,
):
    """Display results from evaluation and save them to a file."""

    if not eval_results:
        raise ValueError("The evaluation results list is empty!")

    # Initialize storage for results
    metric_dicts = []

    for eval_result in eval_results:
        metric_dict = eval_result.metric_dict

        # From results as {metric_name: {k: RetrievalMetricResult}}
        # To {metric_name: {k: score}}
        d = {}
        for metric_name, dict_result in metric_dict.items():
            for k, result in dict_result.items():
                score = np.round(result.score, 3)
                if metric_name not in d:
                    d[metric_name] = {}
                d[metric_name][k] = score

        metric_dicts.append(d)

    # Convert to a DataFrame with multi-indexed columns (metric, k)
    full_df = pd.DataFrame(metric_dicts)

    # Flatten the DataFrame: convert {metric: {k: value}} -> (metric, k) columns
    full_df = full_df.apply(
        lambda row: pd.Series(
            {(m, k): v for m, k_dict in row.items() for k, v in k_dict.items()}
        ),
        axis=1,
    )

    metrics = list(set(metric for metric, _ in full_df.columns))

    # Filter columns to only include relevant metrics
    selected_columns = [(m, k) for m, k in full_df.columns if m in metrics]
    full_df = full_df[selected_columns]

    # Compute mean scores per metric per k
    metrics_mean = full_df.mean().to_dict()

    # Construct a DataFrame for the current results
    result_df = pd.DataFrame(
        {
            ("retriever", "@k"): [name],
            **metrics_mean,
        }
    )

    # If output file is specified, load previous data and append
    if output_file:
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file, header=[0, 1])

            # Ensure the second level is string for concatenation
            existing_df.columns = pd.MultiIndex.from_tuples(
                [(m, str(k)) for m, k in existing_df.columns]
            )
            result_df.columns = pd.MultiIndex.from_tuples(
                [(m, str(k)) for m, k in result_df.columns]
            )

            result_df = pd.concat(
                [existing_df, result_df], ignore_index=True, axis=0, join="outer"
            )

        # Save the updated results
        result_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

    return result_df


class BaseRetrievalEvaluator(ABC):
    """Base class for retrieval evaluators."""

    def __init__(self, metrics: List[BaseRetrievalMetric], k_values: List[int]):
        self.metrics = metrics
        self.k_values = k_values

    @abstractmethod
    async def _aget_retrieved_ids_and_texts(
        self, query: str, mode: str = RetrievalEvalMode.TEXT
    ) -> Tuple[List[str], List[str]]:
        """Retrieve document IDs and texts."""
        raise NotImplementedError

    def evaluate(
        self,
        query: str,
        expected_ids: List[str],
        expected_texts: Optional[List[str]] = None,
        mode: str = RetrievalEvalMode.TEXT,
        **kwargs: Any,
    ) -> RetrievalEvalResult:
        """Run evaluation for a given query."""
        return asyncio.run(
            self.aevaluate(query, expected_ids, expected_texts, mode, **kwargs)
        )

    async def aevaluate(
        self,
        query: str,
        expected_ids: List[str],
        expected_texts: Optional[List[str]] = None,
        mode: str = RetrievalEvalMode.TEXT,
        **kwargs: Any,
    ) -> RetrievalEvalResult:
        """Asynchronous evaluation."""
        retrieved_ids, retrieved_texts = await self._aget_retrieved_ids_and_texts(
            query, mode
        )

        if not expected_ids or not retrieved_ids:
            raise ValueError(
                f"Missing IDs -> Retrieved: {retrieved_ids}, Expected: {expected_ids}"
            )
        k_values = self.k_values
        if k_values:
            metric_dict = {
                metric.metric_name: {
                    k: metric.compute(
                        expected_ids=expected_ids, retrieved_ids=retrieved_ids[:k]
                    )
                    for k in k_values
                }
                for metric in self.metrics
            }  # {metric_name : {k : metric_at_k}}
        else:
            metric_dict = {
                metric.metric_name: {
                    0: metric.compute(
                        expected_ids=expected_ids, retrieved_ids=retrieved_ids
                    )
                }
                for metric in self.metrics
            }  # {metric_name : {k : metric_at_k}}

        return RetrievalEvalResult(
            query,
            expected_ids,
            expected_texts,
            retrieved_ids,
            retrieved_texts,
            mode,
            metric_dict,
        )

    async def aevaluate_dataset(
        self,
        dataset: EmbeddingQAFinetuneDataset,
        workers: int = 2,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[RetrievalEvalResult]:
        """Run evaluation on a dataset."""
        semaphore = asyncio.Semaphore(workers)

        async def eval_worker(
            query: str, expected_ids: List[str], mode: str, **kwargs
        ) -> RetrievalEvalResult:
            async with semaphore:
                return await self.aevaluate(
                    query=query, expected_ids=expected_ids, mode=mode, **kwargs
                )

        response_jobs = []
        mode = dataset.mode  # Assuming dataset has a mode attribute
        for query_id, query in dataset.queries.items():
            expected_ids = dataset.relevant_docs[query_id]  # doi
            response_jobs.append(eval_worker(query, expected_ids, mode))

        if show_progress:
            from tqdm.asyncio import tqdm_asyncio

            eval_results = await tqdm_asyncio.gather(*response_jobs)
        else:
            eval_results = await asyncio.gather(*response_jobs)

        return eval_results


class CustomRetrieverEvaluator(BaseRetrievalEvaluator):
    """Evaluator for vector-based retrievers."""

    retriever: VectorRetrieval
    node_postprocessor: Optional[Any] = None

    def __init__(
        self,
        metrics: List[BaseRetrievalMetric],
        retriever: VectorRetrieval,
        node_postprocessor: Optional[Any] = None,
        k_values: List[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize parameters."""
        super().__init__(metrics, k_values)
        self.retriever = retriever
        self.node_postprocessor = node_postprocessor
        self.kwargs = kwargs  # Store additional parameters for later use

    async def _aget_retrieved_ids_and_texts(
        self, query: str, mode: str = RetrievalEvalMode.TEXT, **kwargs
    ) -> Tuple[List[str], List[str]]:

        # Merge passed kwargs with stored kwargs from initialization
        retrieval_kwargs = {**self.kwargs, **kwargs}

        """Retrieve document IDs and texts, applying post-processors if needed."""
        retrieved_documents: List[RetrievedDocument] = self.retriever.run(
            query, **retrieval_kwargs
        )
        # print("Number of Retrieved Documents : ", len(retrieved_documents))
        if self.node_postprocessor:
            return self.node_postprocessor(retrieved_documents)  # (doi , texts)
        else:
            return (
                [
                    str(hash(doc.content)) for doc in retrieved_documents
                ],  # Simulated unique IDs
                [doc.content for doc in retrieved_documents],
            )


def postprocess_nodes(
    list_documents: List[RetrievedDocument],
) -> Tuple[List[str], List[str]]:
    """
    Postprocess retrieved documents to extract DOIs and texts.
    Returns:
        Tuple[List[str], List[str]]: DOIs and corresponding texts.
    """

    return (
        [doc.metadata.get("doi", "N/A") for doc in list_documents],
        [doc.content for doc in list_documents],
    )
