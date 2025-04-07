import dotenv
dotenv.load_dotenv()

import os
import requests
import pandas as pd
import random

# Load Mistral API key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Load data
BASE_DIR = "/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete/data"

non_pertinent_upper = pd.read_excel(os.path.join(BASE_DIR, "extract1000_article_non-pertinent_upperlim.xlsx")).sample(10, random_state=42)
non_pertinent_lower = pd.read_excel(os.path.join(BASE_DIR, "extract1000_article_non-pertinent_lowerlim.xlsx")).sample(10, random_state=42)
df_mobility = pd.read_csv(os.path.join(BASE_DIR, "df_mobility_all_articles_copy_for_classification_test_subset.csv")).sample(20, random_state=42)

data_subset = pd.concat([non_pertinent_upper, non_pertinent_lower, df_mobility], ignore_index=True)

# Improved few-shot prompt with real examples
FEW_SHOT_PROMPT_TEMPLATE = """
You are a scientific research analyst. Your task is to classify abstracts based on the following sufficiency framework.

---
## Definition of Sufficiency
Sufficiency refers to a set of policy measures and daily practices which avoid the demand for energy, materials, land, water, and other natural resources, while delivering well-being for all within planetary boundaries.

## Classification Criteria
A paper should be classified as **"About Sufficiency"** if it meets EITHER of the following:

### Combination 1:
- **Policy Measures & Practices** (required): Mentions a concrete policy, lifestyle choice, public policy, private action, or system designed to reduce resource consumption
- **Resource Demand Avoidance** (required): Explicitly addresses avoiding or reducing demand for resources (energy, materials, land, water, etc.)

### Combination 2:
- **Policy Measures & Practices** (required)
- **Well-being Delivery** (required): Human well-being, equity, social welfare, etc.
- **Planetary Boundaries Context** (required): Environmental limits, sustainability, climate change, or similar

---
## Examples

### Example 1
**Title**: Sustainable Urban Transport Planning
**Abstract**: This study discusses strategies such as walking, cycling, and public transportation integration to reduce urban car usage and associated emissions.
**Label**: About Sufficiency

### Example 2
**Title**: Smart Grids for Better Electricity Distribution
**Abstract**: This paper presents technical advancements in smart grids to enhance electricity efficiency without discussing demand-side reduction or behavior change.
**Label**: Not About Sufficiency

### Example 3
**Title**: Low-consumption Lifestyles in Households
**Abstract**: We analyze how voluntary simplicity, energy-saving practices, and community initiatives contribute to reduced energy and material use in residential settings.
**Label**: About Sufficiency

---
## Now, classify the following abstract:

**Title**: {title}

**Abstract**: {abstract}

**Label**:
"""

# Function to classify using Mistral API
def classify_abstract(title, abstract):
    prompt = FEW_SHOT_PROMPT_TEMPLATE.format(title=title, abstract=abstract)
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-large-latest",
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(MISTRAL_API_URL, headers=headers, json=data)

    try:
        response_data = response.json()
    except Exception as e:
        print("Failed to decode JSON response:", e)
        print(response.text)
        return "Error"

    if "choices" not in response_data:
        print("Unexpected API response (no 'choices'):")
        print(response_data)
        return "Error"

    return response_data['choices'][0]['message']['content'].strip()

# Safe classification with logging
def safe_classify(row):
    try:
        print(f"Classifying: {row['title'][:60]}...")
        return classify_abstract(row['title'], row['abstract'])
    except Exception as e:
        print(f"❌ Error while classifying: {e}")
        return "Error"

# Run the classification
data_subset['Mistral_Classification'] = data_subset.apply(safe_classify, axis=1)

# Save results
output_path = os.path.join(BASE_DIR, "classified_subset_results_mistral.csv")
data_subset.to_csv(output_path, index=False)

print(f"✅ Classification complete. Results saved to '{output_path}'")
