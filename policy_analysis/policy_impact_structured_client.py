from pydantic import BaseModel, Field
import requests
from typing import Literal
from concurrent.futures import ThreadPoolExecutor


BACKEND_URL = "http://localhost:8000"

RESOURCE_TAXONOMY_DETAILS = """Consider the following natural resources: freshwater, marine resources, wetlands, metals and ores, non-metallic minerals, fossil fuels, agricultural land, forests, urban land, biomass.
Positive means the policy contributes to the preservation or regeneration of one or more of these resources"""

WELLBEING_TAXONOMY_DETAILS = "Consider the following aspects of wellbeing: housing, jobs, education, civic engagement, life satisfaction, work-life balance, income, community, environment, health, safety."

JUSTICE_TAXONOMY_DETAILS = "Consider the following types of justice: distributional, procedural, corrective, recognitional, transitional."

PLANETARY_BOUNDARIES_TAXONOMY_DETAILS = """Consider the following planetary boundaries: land system change, climate change, biosphere integrity, biogeochemical flows, ocean acidification, freshwater use, atmospheric aerosol loading, ozone depletion, introduction of novel entities.
Positive means the policy is contributes to keeping human activities within these boundaries."""


impacts = {
    "natural_resources": RESOURCE_TAXONOMY_DETAILS,
    "wellbeing": WELLBEING_TAXONOMY_DETAILS,
    "justice_considerations": JUSTICE_TAXONOMY_DETAILS,
    "planetary_boundaries": PLANETARY_BOUNDARIES_TAXONOMY_DETAILS,
}


def build_question(policy: str, impact_dimension: str, precision: str) -> str:
    return f"What is the impact of {policy} on {impact_dimension}? {precision}"


class ImpactSchema(BaseModel):
    impact: Literal["positive", "neutral", "negative", "unknown"] = Field(
        description="The likely impact direction of the policy on the given impact dimension, based on retrieved scientific evidence. "
        "Use 'positive' when evidence suggests overall beneficial effects on the dimension, 'negative' for harmful effects, 'neutral' when effects are mixed, unclear, or not supported by evidence, and 'unknown' when there is no evidence available."
    )


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
    impact_str = impact_dimension.replace("_", " ")
    details = impacts[impact_dimension]
    question = build_question(policy, impact_str, details)
    payload = {
        "messages": [{"role": "user", "content": question}],
        "output_schema": ImpactSchema.model_json_schema(),
        "schema_name": "ImpactSchema",
        "fetch_pubs": True,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 64,
        "timeout": 60,
    }

    response = _post_json(f"{backend_url}/internal/structured-output", payload)
    return response["output"]["impact"]


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
