# prompt to ask an LLM to classify papers as about sufficiency or not about sufficiency (binary) based on their abstract
# approach: provide the official definition of sufficiency to Claude using Sonnet 3.7 asking to derive a prompt to classify scientific papers
# steps: 1st iterated on the prompt to ensure Claude, (2) tested with Mistral Large 7 to classify the documents, it returned most of them as not about sufficiency (about 90% false negative, same results with GPT4)
# provided a list of 200 manually classified documents that were not classified "about sufficiency" and 200 "about sufficiency" and reran a slightly optimized queries (keyword change): the results were comparable
# Then attempted to do few-shots learning, with an adjusted prompt using a few examples from the manually classified datasets for the mobility sector:

You are a scientific research analyst. Your task is to classify abstracts based on the following sufficiency framework.

---

## Definition of Sufficiency
Sufficiency refers to a set of policy measures and daily practices which avoid the demand for energy, materials, land, water, and other natural resources, while delivering well-being for all within planetary boundaries.

---

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
**Abstract**: This paper examines the rise of car-sharing initiatives in European cities as a means to reduce private car ownership. By shifting transportation behaviors and reducing the number of vehicles per capita, these policies aim to reduce the demand for land use, fossil fuel consumption, and urban infrastructure while maintaining mobility access for all citizens.  
**Label**: About Sufficiency

---

### Example 2  
**Title**: The Impact of Indoor Farming on Resource Consumption  
**Abstract**: We analyze how compact indoor farming systems can decrease the need for extensive agricultural land, fertilizer inputs, and water usage. The study discusses emerging technologies and policy frameworks that support urban food self-sufficiency and local well-being.  
**Label**: About Sufficiency

---

### Example 3  
**Title**: Collaborative Housing Models and Urban Sustainability  
**Abstract**: This research explores co-housing communities that aim to reduce per-capita residential space and shared appliance use. The paper also discusses how such initiatives foster community, reduce energy needs, and contribute to lower ecological footprints.  
**Label**: About Sufficiency

---

### Example 4  
**Title**: Advances in Battery Storage for Grid Efficiency  
**Abstract**: The study evaluates the efficiency improvements in lithium-ion battery storage and their potential to enhance renewable energy integration. Modeling shows promising gains in discharge efficiency and reduced system cost.  
**Label**: Not About Sufficiency

---

### Example 5  
**Title**: Machine Learning for Climate Prediction  
**Abstract**: We present a machine learning framework for improving long-term climate forecasting using large-scale satellite data. This paper does not evaluate human behavior, demand reduction, or public policies.  
**Label**: Not About Sufficiency

---

### Example 6  
**Title**: Material Substitution in Steel Manufacturing  
**Abstract**: This paper discusses how switching to lower-carbon materials in steel production can reduce emissions. It focuses on technological substitutions but does not address demand-side reductions or lifestyle change.  
**Label**: Not About Sufficiency

---

## Now, classify the following abstract:

**Title**: {{INSERT_TITLE_HERE}}

**Abstract**: {{INSERT_ABSTRACT_HERE}}

**Label**:


## import openai
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

