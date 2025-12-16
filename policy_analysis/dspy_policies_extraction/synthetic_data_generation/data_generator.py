from openai import OpenAI
import json
import time
import os
from pathlib import Path
import re
from dotenv import load_dotenv
import random

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

def generate_synthetic_qa(existing_qa, n_samples=5, max_retries=3):
    synthetic_data = []

    for i in range(n_samples):
        existing_qa_sample = random.sample(existing_qa, 4)
        for attempt in range(max_retries):
            prompt = (
                "You are a creative data generator. Given the following examples of question–response pairs, "
                "generate ONE new question–response pair that is related to the same domain but introduces "
                "some novelty in topic, phrasing, or perspective. Try to make it distinct from the examples, "
                "while keeping the difficulty and style appropriate.\n\n"
                f"Examples:\n{json.dumps(existing_qa_sample)}\n\n"
                "Output ONLY a JSON object with fields: question, response. "
                "Do not repeat any content from the examples."
            )

            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8
            )

            text = response.choices[0].message.content
            parsed = extract_json(text)
            print(f"Attempt {attempt+1} for sample {i}: {text}")
            if parsed:
                synthetic_data.append(parsed)
                break  # success, go to next sample
            else:
                print(f"Warning: Failed to parse JSON for sample {i}, attempt {attempt+1}. Retrying...")
                time.sleep(0.5)
        else:
            print(f"Skipping sample {i} after {max_retries} failed attempts.")

    return synthetic_data

# ---- EXAMPLE USAGE ----
# Read JSONL
root = Path(__file__).parent
original = []
extracted_politics_examples_fp = root / "extracted_politics_examples.jsonl"
with extracted_politics_examples_fp.open("r", encoding="utf-8") as f:
    for line in f:
        original.append(json.loads(line))

guidance_examples = original[:32]  # On utiliser tous les exemples gold que l'on a pour la génération de données synthétiques

synthetic = generate_synthetic_qa(guidance_examples, n_samples=100)

# Write to JSONL
synthetic_qa_fp = root / "synthetic_qa2.jsonl"
with synthetic_qa_fp.open( "w", encoding="utf-8") as f:
    for item in synthetic:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Synthetic dataset saved to {synthetic_qa_fp}")
