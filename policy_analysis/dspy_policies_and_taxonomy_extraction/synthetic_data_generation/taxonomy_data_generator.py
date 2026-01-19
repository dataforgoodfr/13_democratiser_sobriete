from openai import OpenAI
import json
import time
import os
import re
from dotenv import load_dotenv
import random


from taxonomy_extraction.geographical_taxonomy import (
    Geographical_scope,
    Studied_country,
    Regional_group,
)

from taxonomy_extraction.Impact_taxonomy import (
    Human_needs,
    Natural_ressource,
    Wellbeing,
    Justice_consideration,
    Planetary_boundaries,
)


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_json(text):
    """Extract the first JSON object from a string."""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return None


def generate_synthetic_qa(existing_qa, n_samples=5, max_retries=3):
    synthetic_data = []

    for i in range(n_samples):
        guidance_examples = random.sample(existing_qa, 4)

        for attempt in range(max_retries):
                
            prompt = (
                "You are generating synthetic scientific-policy text for a climate and environmental research dataset.\n\n"

                "IMPORTANT: ALL taxonomy fields MUST strictly follow the allowed values below.\n\n"

                "=== GEOGRAPHY TAXONOMY ===\n\n"

                "Regional group (EXCLUSIVE, one value or null):\n"
                f"{[e.value for e in Regional_group]}, the one impacted in the text if any\n\n"

                "Geographical scopes (MULTIPLE CHOICE,none or one or more):\n"
                f"{[e.value for e in Geographical_scope]} the ones impacted in the text if any\n\n"

                "Studied countries (MULTIPLE CHOICE,none or one or more):\n"
                f"{[e.value for e in Studied_country]}  the ones impacted in the text if any \n\n"

                "Geography rules:\n"
                "- Use ONLY values from the lists above\n"
                "- Do NOT invent new regions or country names\n"
                "- If the scope is global or regional, the country list may be empty\n"
                "- regional_group must be null ONLY if it cannot be inferred\n\n"

                "=== IMPACT TAXONOMY ===\n\n"

                "Human needs (MULTIPLE CHOICE,none or one or more):\n"
                f"{[e.value for e in Human_needs]}, the ones impacted explicitly in the text if any\n\n"

                "Natural resources (MULTIPLE CHOICE,none or one or more):\n"
                f"{[e.value for e in Natural_ressource]},the ones impacted explicitly in the text if any \n\n"

                "Wellbeing (MULTIPLE CHOICE,none or one or more):\n"
                f"{[e.value for e in Wellbeing]},the ones impacted explicitly in the text if any \n\n"

                "Justice considerations (MULTIPLE CHOICE,none or one or more):\n"
                f"{[e.value for e in Justice_consideration]},the ones impacted explicitly in the text if any \n\n"

                "Planetary boundaries (MULTIPLE CHOICE,none or one or more):\n"
                f"{[e.value for e in Planetary_boundaries]},the ones impacted explicitly in the text if any \n\n"

                "Impact rules:\n"
                "- Use ONLY values from the lists above\n"
                "- Do NOT invent new labels\n"
                "- Use empty arrays when no impact category applies\n"
                "- Do NOT use null for impact fields\n\n"

                "Generate ONE NEW entry that is:\n"
                "- Written in a formal academic or policy-report tone\n"
                "- Similar in length and structure to the examples\n"
                "- Phrased as conclusions, findings, or implications (not a literal question)\n"
                "- Novel in wording and focus\n\n"

                "Output a VALID JSON object with EXACTLY these fields:\n"
                "- question (string)\n"
                "- regional_group (string or null)\n"
                "- geographical_scopes (array of strings)\n"
                "- main_country_focus (array of strings)\n"
                "- human_needs (array of strings)\n"
                "- natural_ressource (array of strings)\n"
                "- wellbeing (array of strings)\n"
                "- justice_consideration (array of strings)\n"
                "- planetary_boundaries (array of strings)\n\n"

                "Annotated examples:\n"
                f"{json.dumps(guidance_examples, ensure_ascii=False, indent=2)}\n\n"
                "Output ONLY the JSON object. No markdown. No explanation."
            )


            

            response = client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8
            )

            text = response.choices[0].message.content
            parsed = extract_json(text)

            print(f"Attempt {attempt + 1} for sample {i}:\n{text}\n")

            if parsed and isinstance(parsed, dict):
                synthetic_data.append(parsed)
                break
            else:
                print("Warning: Invalid JSON. Retrying...")
                time.sleep(0.5)

        else:
            print(f"Skipping sample {i} after {max_retries} failed attempts.")

    return synthetic_data


# ---- EXAMPLE USAGE ----

original = []
with open("taxonomy_extraction/transformed_data_all.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        original.append(json.loads(line))

guidance_examples = original[:32]

synthetic = generate_synthetic_qa(guidance_examples, n_samples=50)

with open("synthetic_data-all.jsonl", "w", encoding="utf-8") as f:
    for item in synthetic:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Synthetic dataset saved to synthetic_data-all.jsonl")
