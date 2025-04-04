# example of few-shot learning script with GPT

import openai
import pandas as pd
import random

# Set up OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Load a subset of the provided files
non_pertinent_upper = pd.read_excel("extract1000_article_non-pertinent_upperlim.xlsx").sample(10, random_state=42)
non_pertinent_lower = pd.read_excel("extract1000_article_non-pertinent_lowerlim.xlsx").sample(10, random_state=42)
df_mobility = pd.read_csv("df_mobility_all_articles_copy_for_classification_test_subset - df_mobility_all_articles.csv").sample(10, random_state=42)

# Merge the subsets
data_subset = pd.concat([non_pertinent_upper, non_pertinent_lower, df_mobility], ignore_index=True)

# Define the few-shot prompt template
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
**Title**: Sharing Vehicles: A Path to Sustainable Urban Transport  
**Abstract**: This paper examines the rise of car-sharing initiatives in European cities as a means to reduce private car ownership...  
**Label**: About Sufficiency

### Example 2  
**Title**: Advances in Battery Storage for Grid Efficiency  
**Abstract**: The study evaluates the efficiency improvements in lithium-ion battery storage...  
**Label**: Not About Sufficiency

---
## Now, classify the following abstract:

**Title**: {title}

**Abstract**: {abstract}

**Label**:
"""

# Function to classify an abstract using GPT-4
def classify_abstract(title, abstract):
    prompt = FEW_SHOT_PROMPT_TEMPLATE.format(title=title, abstract=abstract)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response['choices'][0]['message']['content'].strip()

# Apply classification to the subset
data_subset['GPT_Classification'] = data_subset.apply(lambda row: classify_abstract(row['primary_title'], row['abstract']), axis=1)

# Save results
data_subset.to_csv("classified_subset_results.csv", index=False)

print("Classification complete. Results saved to 'classified_subset_results.csv'")
