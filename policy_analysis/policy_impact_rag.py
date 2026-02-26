"""
WARNING: understand what you're doing before running.
The script is currently set for assessing negative impacts only, although you can switch back easily to impacts in general.
This was done to save on costs after a first run mostly returned positive impacts, while we wanted to also find negative or neutral ones.
The final script should aim to find all three (positive, negative, and neutral).
Since each full run is costly (dozens of millions of input tokens sent to Scaleway API), we did a second run for negative impacts only.
"""


from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import requests
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

# TODO: save to a jsonl file, it would be cleaner
OUTPUT_FILE = 'negative_impact_evaluation_results_justif.pkl'


BACKEND_URL = "http://localhost:8000"

RESOURCE_TAXONOMY_DETAILS = """Consider the following natural resources: freshwater, marine resources, wetlands, metals and ores, non-metallic minerals, fossil fuels, agricultural land, forests, urban land, biomass.
Positive means the policy contributes to the preservation or regeneration of one or more of these resources. Negative means the opposite."""

WELLBEING_TAXONOMY_DETAILS = "Consider the following aspects of wellbeing: housing, jobs, education, civic engagement, life satisfaction, work-life balance, income, community, environment, health, safety."

JUSTICE_TAXONOMY_DETAILS = "Consider the following types of justice: distributional, procedural, corrective, recognitional, transitional."

PLANETARY_BOUNDARIES_TAXONOMY_DETAILS = """Consider the following planetary boundaries: land system change, climate change, biosphere integrity, biogeochemical flows, ocean acidification, freshwater use, atmospheric aerosol loading, ozone depletion, introduction of novel entities.
Positive means the policy is contributes to keeping human activities within these boundaries. Negative means the opposite."""


impacts = {
    "natural_resources": RESOURCE_TAXONOMY_DETAILS,
    "wellbeing": WELLBEING_TAXONOMY_DETAILS,
    "justice_considerations": JUSTICE_TAXONOMY_DETAILS,
    "planetary_boundaries": PLANETARY_BOUNDARIES_TAXONOMY_DETAILS,
}


def build_question(policy: str, impact_dimension: str, precision: str) -> str:
    return f"What is the impact of {policy} on {impact_dimension.replace('_', ' ')}? {precision}"


def build_negative_question(policy: str, impact_dimension: str, precision: str) -> str:
    return f"What are the NEGATIVE (or adverse, detrimental, etc) impact of {policy} on {impact_dimension.replace('_', ' ')}? {precision} Really focus on negative impacts, or at best mixed/neutral impacts, even if the overall impact is positive."


class Evidence(BaseModel):
    evidence: str = Field(description="Very concise (1-2 sentences MAX) summary of the evidence from the document supporting the impact assessment.")
    openalex_ids: list[str] = Field(description="List of OpenAlex IDs of documents supporting this evidence.")
    impact: Literal["positive", "neutral", "negative"] = Field(
        description="The likely impact direction of the policy on the given impact dimension, based on the evidence. "
        "Use 'positive' when evidence suggests overall beneficial effects on the dimension, 'negative' for harmful effects, 'neutral' when effects are mixed, unclear, or not supported by evidence."
    )


class StructuredImpactResponse(BaseModel):
    evidences: list[Evidence] = Field(
         description="List of evidences from retrieved documents. Not all documents need to appear: unhelpful documents can be ignored. "
         "Likewise, not all dimensions of the impact need to be covered: if the evidence only supports an assessment on a subset of dimensions, that's fine. "
         "Do not list an unsupported evidence."
         )
    overall_impact: Literal["positive", "neutral", "negative", "unknown"] = Field(description="Use unknown only when there is no evidence available")


def _post_json(url: str, payload: dict) -> dict:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def evaluate_single_impact(
    policy: str,
    impact_dimension: Literal[
        "natural_resources",
        "wellbeing",
        "justice_considerations",
        "planetary_boundaries",
    ],
    backend_url: str = BACKEND_URL,
) -> str:
    details = impacts[impact_dimension]
    #question = build_question(policy, impact_dimension, details)
    question = build_negative_question(policy, impact_dimension, details)
    payload = {
        "messages": [{"role": "user", "content": question}],
        "output_schema": StructuredImpactResponse.model_json_schema(),
        "schema_name": "ImpactSchema",
        "fetch_pubs": True,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 1024,
        "timeout": 60,
    }

    try:
        response = _post_json(f"{backend_url}/internal/structured-output", payload)
        return response["output"]
    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        response_text = e.response.text if e.response is not None else ""
        s = f"error:http:{status_code}:{response_text}"
        print(s)
        return s
    except Exception as e:
        return f"error:exception:{e}"

def evaluate_policy(
    policy: str,
    backend_url: str = BACKEND_URL,
) -> dict[str, str]:
    results = {}
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                evaluate_single_impact,
                policy=policy,
                impact_dimension=dimension,
                backend_url=backend_url,
            ): dimension
            for dimension in impacts.keys()
        }
        for future in futures:
            dimension = futures[future]
            results[dimension] = future.result()
    return results


def main():
    max_concurrent_requests = 10
    max_consecutive_errors = 20
    nmax = None

    # if output file exists, load it to avoid redoing work
    try:
        with open(OUTPUT_FILE, 'rb') as f_in:
            results = pickle.load(f_in)
            results = [r for r in results if isinstance(r[-1], dict)]
            print(f"Loaded {len(results)} existing results from {OUTPUT_FILE}.")
    except FileNotFoundError:
        results = []
        print("No existing results found. Starting fresh.")

    df = pd.read_parquet('sample10k_results/sample_10k_policy_clusters_llm_2026-02-12.parquet')
    print(f'Loaded {len(df)} policy clusters.')
    

    def _eval_one(idx, policy, impact, details):
        evaluation = evaluate_single_impact(policy, impact, backend_url='http://localhost:8000')
        #question = build_question(policy, impact, details)
        question = build_negative_question(policy, impact, details)
        return (idx, policy, impact, question, evaluation)

    # Build task args once, filtering out already existing results
    existing_tasks = set((r[0], r[2]) for r in results)
    task_args = [
        (idx, row["cluster"], impact, details)
        for idx, row in df.iloc[:nmax].iterrows()
        for impact, details in impacts.items()
        if (idx, impact) not in existing_tasks
    ]
    print(f"Prepared {len(task_args)} tasks for evaluation ( {len(df)} x {len(impacts)} - {len(results)} = {len(df)*len(impacts) - len(results)} ).")
    print('Will save to ', OUTPUT_FILE)

    consecutive_errors = 0
    with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
        futures = [executor.submit(_eval_one, *args) for args in task_args]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Evaluating impacts"):
            r = f.result()
            if isinstance(r[-1], str) and r[-1].startswith("error:"):
                consecutive_errors += 1
                print(f"Error encountered: {r[-1]}. Consecutive errors: {consecutive_errors}")
                if consecutive_errors >= max_consecutive_errors:
                    print(f"Reached maximum consecutive errors ({max_consecutive_errors}). Stopping execution.")
                    break
            else:
                consecutive_errors = 0
            
            results.append(r)
            if len(results) % 100 == 0:
                with open(OUTPUT_FILE, 'wb') as f_out:
                    pickle.dump(results, f_out)

    with open(OUTPUT_FILE, 'wb') as f_out:
        pickle.dump(results, f_out)

if __name__ == "__main__":
    main()
