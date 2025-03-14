### Utils functions for Query Generation ###
import json
import re
import uuid
import warnings
from tqdm import tqdm
from typing import Dict, List, Tuple
from tqdm import tqdm
import random
import pandas as pd
import itertools

from kotaemon.llms import ChatLLM

from llama_index.core.llms.utils import LLM
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode

from .prompts import (GROUNDNESS_CRITIC_PROMPT, 
                      RELEVANCE_CRITIC_PROMPT,
                      STANDALONE_CRITIC_PROMPT,
                      DEFAULT_QA_GENERATE_PROMPT_TMPL
)

from benchmark.retrieval_evaluation.custom_evaluator import (
    EmbeddingQAFinetuneDataset,
)


def load_corpus(
    folder_path, num_files_limit, pdf_reader, chunk_size, chunk_overlap, verbose=False
):
    if verbose:
        print(f"Loading PDFs from {folder_path}")

    reader = SimpleDirectoryReader(
        input_dir=folder_path,
        file_extractor={".pdf": pdf_reader},
        recursive=False,
        required_exts=[".pdf"],
        num_files_limit=num_files_limit,
        raise_on_error=False,
    )
    docs = reader.load_data(num_workers=4, show_progress=True)
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)
    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes


def load_existing_data(
    path: str,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    """Load existing data from a JSON file if it exists.

    Args:
        path (str): The file path to load the JSON from.

    Returns:
        Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]: The loaded queries, corpus, and relevant_docs.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["queries"], data["corpus"], data["relevant_docs"]
    except FileNotFoundError:
        return {}, {}, {}


def random_prompt_generator(list_prompts: List[str] = None):
    if list_prompts:
        return random.choice(list_prompts)
    else:
        return DEFAULT_QA_GENERATE_PROMPT_TMPL


def extract_scores_and_justifications(response: str):
    """
    Extracts the evaluation justification and total score from a given response.

    Parameters:
        response (str): The text response containing 'Evaluation:' and 'Total score:'.

    Returns:
        dict: A dictionary with 'evaluation' (str) and 'score' (int) or None if not found.
    """
    evaluation_pattern = r"Evaluation:\s*(.*?)\s*(?=Total score:|$)"
    score_pattern = r"Total score:\s*(\d+)"

    evaluation_match = re.search(evaluation_pattern, response, re.DOTALL)
    score_match = re.search(score_pattern, response)

    evaluation = evaluation_match.group(1).strip() if evaluation_match else None
    score = int(score_match.group(1)) if score_match else None

    return {"evaluation": evaluation, "score": score}


def llm_as_a_judge(
    qa_dataset: EmbeddingQAFinetuneDataset, llm: ChatLLM, nb_query: int = 1
):
    """
    Filters questions using an LLM based on groundedness, relevance, and standalone criteria.

    Parameters:
        qa_dataset (EmbeddingQAFinetuneDataset): The dataset containing queries, corpus, and relevant docs.
        llm (ChatLLM): The language model used for evaluation.
        nb_query (int, optional): Number of queries to evaluate. Defaults to evaluating all.

    Returns:
        EmbeddingQAFinetuneDataset: Filtered dataset with low-quality queries removed.
    """
    queries, corpus = qa_dataset.queries, qa_dataset.corpus
    nb_query = (
        len(queries.keys()) if nb_query <= 0 else min(nb_query, len(queries.keys()))
    )

    inverse_dict = {query: query_id for query_id, query in queries.items()}
    results = []

    def evaluate_query(prompt_template, query, context=""):
        """Helper function to evaluate a query using LLM with error handling."""
        try:
            prompt = prompt_template.format(context=context, question=query)
            response = llm.run(prompt)
            return extract_scores_and_justifications(response.content)
        except Exception as e:
            return {"evaluation": "Extraction failed", "score": None, "error": str(e)}

    for query_id, query in tqdm(
        itertools.islice(queries.items(), nb_query),
        desc="LLM-as-a-judge query filtering",
    ):
        query_context = corpus.get(query_id, "")

        ground_result = evaluate_query(GROUNDNESS_CRITIC_PROMPT, query, query_context)
        relevance_result = evaluate_query(RELEVANCE_CRITIC_PROMPT, query)
        stand_result = evaluate_query(STANDALONE_CRITIC_PROMPT, query)

        results.append(
            {
                "query": query,
                "groundedness_eval": ground_result["evaluation"],
                "groundedness_score": ground_result["score"],
                "relevance_eval": relevance_result["evaluation"],
                "relevance_score": relevance_result["score"],
                "standalone_eval": stand_result["evaluation"],
                "standalone_score": stand_result["score"],
            }
        )

    # Convert results to DataFrame and filter out low-scoring queries
    df = pd.DataFrame(results)
    query_to_remove = df.loc[
        (df["groundedness_score"] < 4)
        | (df["relevance_score"] < 4)
        | (df["standalone_score"] < 4)
    ]["query"].tolist()

    # Remove queries and corresponding relevant docs
    queries_relevant_docs_keys_to_remove = {inverse_dict[q] for q in query_to_remove}

    filtered_queries = {
        k: v
        for k, v in queries.items()
        if k not in queries_relevant_docs_keys_to_remove
    }
    filtered_relevant_docs = {
        k: v
        for k, v in qa_dataset.relevant_docs.items()
        if k not in queries_relevant_docs_keys_to_remove
    }

    return EmbeddingQAFinetuneDataset(
        queries=filtered_queries,
        corpus=qa_dataset.corpus,
        relevant_docs=filtered_relevant_docs,
    )


def query_selection(generated_query: str) -> bool:
    """
    Filter out generated queries based on simple syntaxic rules.
    """
    # Strip leading/trailing whitespace
    generated_query = generated_query.strip()

    # Define disallowed query prefixes
    disallowed_phrases = ["Here are", "Here are two", "Here's"]

    # Check if the query contains any disallowed phrase
    if any(generated_query.startswith(phrase) for phrase in disallowed_phrases):
        return False

    # Regex to ensure the first character is an uppercase letter
    uppercase_pattern = re.compile(r"^[A-Z]")

    # Validate that query starts with an uppercase letter
    if not uppercase_pattern.match(generated_query):
        return False

    return True


# function to clean the dataset
# Source : https://www.llamaindex.ai/blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83
def syntax_filtering_dataset(qa_dataset: EmbeddingQAFinetuneDataset):
    """
    Filters out queries from the qa_dataset that contain certain phrases and the corresponding
    entries in the relevant_docs, and creates a new EmbeddingQAFinetuneDataset object with
    the filtered data.

    :param qa_dataset: An object that has 'queries', 'corpus', and 'relevant_docs' attributes.
    :return: An EmbeddingQAFinetuneDataset object with the filtered queries, corpus and relevant_docs.
    """

    # Extract keys from queries and relevant_docs that need to be removed
    queries_relevant_docs_keys_to_remove = {
        k for k, v in qa_dataset.queries.items() if not query_selection(v)
    }

    # Filter queries and relevant_docs using dictionary comprehensions
    filtered_queries = {
        k: v
        for k, v in qa_dataset.queries.items()
        if k not in queries_relevant_docs_keys_to_remove
    }
    filtered_relevant_docs = {
        k: v
        for k, v in qa_dataset.relevant_docs.items()
        if k not in queries_relevant_docs_keys_to_remove
    }

    # Create a new instance of EmbeddingQAFinetuneDataset with the filtered data
    return EmbeddingQAFinetuneDataset(
        queries=filtered_queries,
        corpus=qa_dataset.corpus,
        relevant_docs=filtered_relevant_docs,
    )


def generate_qa_embedding_pairs(
    nodes: List[TextNode],
    llm: LLM,
    qa_generate_prompt_tmpl: str = None,
    num_questions_per_chunk: int = 2,
    retry_limit: int = 3,
    on_failure: str = "continue",  # options are "fail" or "continue"
    save_every: int = 500,
    output_path: str = "qa_finetune_dataset.json",
    verbose: bool = False,
    query_filtering_fcts: list[str] = None,
) -> EmbeddingQAFinetuneDataset:
    """Generate QA pairs from a set of nodes and save periodically.

    Args:
        nodes (List[TextNode]): List of TextNode objects to process.
        llm (LLM): The large language model to use for generating questions.
        qa_generate_prompt_tmpl (str): The template for generating QA prompts.
        num_questions_per_chunk (int): Number of questions to generate per chunk of text.
        retry_limit (int): Number of times to retry on failure.
        on_failure (str): Action to take on repeated failures ('fail' or 'continue').
        save_every (int): Number of nodes to process before saving the dataset.
        output_path (str): The file path to save the JSON output.
        verbose (bool): If True, print debugging messages.
        query_filtering_fcts (list[str]) : filtering functions

    Returns:
        EmbeddingQAFinetuneDataset: The generated dataset.
    """
    queries, corpus, relevant_docs = load_existing_data(output_path)
    # relevant_docs is expected to be a dict mapping question IDs to a list of DOIs

    node_dict = {
        node.node_id: node.get_content(metadata_mode=MetadataMode.NONE)
        for node in nodes
    }  # {node_id : chunk_text}
    node_dict_doi = {
        node.node_id: node.metadata.get("doi", "N/A") for node in nodes
    }  # {node_id : doi}

    start_index = len(corpus)

    save_counter = start_index

    for node_id, text in tqdm(
        list(node_dict.items())[start_index:], initial=start_index
    ):

        if qa_generate_prompt_tmpl is None:
            qa_generate_prompt_tmpl = random_prompt_generator(
                list_prompts=[
                    POLITICIAN_QA_GENERATE_PROMPT_TMPL,
                    CITIZEN_QA_GENERATE_PROMPT_TMPL,
                    SCIENTIST_QA_GENERATE_PROMPT_TMPL,
                ]
            )
        query = qa_generate_prompt_tmpl.format(
            context_str=text, num_questions_per_chunk=num_questions_per_chunk
        )

        retry_count = 0
        success = False
        while retry_count < retry_limit:
            try:
                response = llm.run(query)
                success = True
                break
            except Exception as e:
                retry_count += 1
                if verbose:
                    print(
                        f"Error querying LLM: {e}. Retrying {retry_count}/{retry_limit}..."
                    )

        if not success:
            if on_failure == "fail":
                raise RuntimeError(f"Failed to query LLM after {retry_limit} retries.")
            elif on_failure == "continue":
                if verbose:
                    print(f"Skipping Node ID {node_id} after {retry_limit} retries.")
                continue

        query_pattern = r"(?:^|\s)([^?!.]*\?)"  # original pattern r"^\d+[\).\s]"
        result = str(response).strip().split("\n")
        questions = []
        for res in result:
            for q in re.findall(query_pattern, res):
                questions.append(q)

        questions = [question for question in questions if len(question) > 0][
            :num_questions_per_chunk
        ]

        num_questions_generated = len(questions)
        if num_questions_generated < num_questions_per_chunk:
            warnings.warn(
                f"Fewer questions generated ({num_questions_generated}) "
                f"than requested ({num_questions_per_chunk})."
            )

        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            if question_id not in relevant_docs:
                relevant_docs[question_id] = []
            relevant_docs[question_id].append(
                node_dict_doi[node_id]
            )  # Store DOI instead of node_id // relevant_docs : {"query_id" : ["doi"]}

        corpus[node_id] = text  # Store text under node_id

        save_counter += 1
        if save_counter % save_every == 0:
            dataset = EmbeddingQAFinetuneDataset(
                queries=queries, corpus=corpus, relevant_docs=relevant_docs
            )
            dataset.save_json(output_path)
            if verbose:
                print(f"Saved progress at {save_counter} entries.")

    # Save final dataset
    dataset = EmbeddingQAFinetuneDataset(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs
    )
    if query_filtering_fcts:
        if "syntax" in query_filtering_fcts:
            dataset = syntax_filtering_dataset(dataset)
        if "llm-as-a-judge" in query_filtering_fcts:
            dataset = llm_as_a_judge(qa_dataset=dataset, llm=llm, nb_query=0)

    dataset.save_json(output_path)
    if verbose:
        print("Final dataset saved.")

    return dataset
