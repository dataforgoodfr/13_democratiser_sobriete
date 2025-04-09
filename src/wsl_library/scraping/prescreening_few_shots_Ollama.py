# example of few-shot learning script with GPT

import os

import ollama
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

tqdm.pandas()

# Load a subset of the provided files
data_path = "src/wsl_library/scraping/data/"
non_pertinent_lowerlim_df = pd.read_excel(
    os.path.join(
        data_path,
        "extract1000_article_non-pertinent_lowerlim.xlsx",
    ),
    usecols=["abstract", "primary_title"],
).sample(10, random_state=42)
non_pertinent_upperlim_df = pd.read_excel(
    os.path.join(
        data_path,
        "extract1000_article_non-pertinent_upperlim.xlsx",
    ),
    usecols=["abstract", "primary_title"],
).sample(10, random_state=42)
df_mobility_df = (
    pd.read_csv(
        os.path.join(
            data_path,
            "df_mobility_all_articles_copy_for_classification_test_subset - df_mobility_all_articles.csv",
        ),
        usecols=["abstract", "title"],
    )
    .rename(columns={"title": "primary_title"})
    .sample(10, random_state=42)
)

# Add manual filter column
non_pertinent_lowerlim_df["origin"] = "lowerlim"
non_pertinent_upperlim_df["origin"] = "upperlim"
non_pertinent_lowerlim_df["true_label"] = "Not About Sufficiency"
non_pertinent_upperlim_df["true_label"] = "Not About Sufficiency"
df_mobility_df["true_label"] = "About Sufficiency"
df_mobility_df["origin"] = "pertinent"

# Merge the subsets
data_subset = pd.concat(
    [non_pertinent_lowerlim_df, non_pertinent_upperlim_df, df_mobility_df], ignore_index=True
)

# Define the few-shot prompt template
FEW_SHOT_PROMPT_TEMPLATE = """
You are a scientific research analyst. Your task is to classify abstracts based on whether they relate to **sufficiency in the context of sustainability**.

---
## ✅ Definition of Sufficiency
Sufficiency refers to a set of **policy measures and daily practices** which **avoid the demand** for energy, materials, land, water, and other natural resources, **while delivering well-being** for all within **planetary boundaries**.

---
## ✅ Classify as "About Sufficiency" if:
- It satisfies the definition of sufficiency given before.
- OR, it discusses **policies or actions** (public, private, or lifestyle) to **reduce resource demand**.
- OR, it includes all three:
   - **Policies/Actions** + **Well-being delivery** + **Environmental sustainability context**

---
## ❌ Classify as "Not About Sufficiency" if:
- It does not satisfy the definition of sufficiency given before.
- OR, the abstract is unrelated to sustainability, mobility, environmental limits, or social equity.
- OR, it discusses efficiency or technology without behavioral or systemic reduction in demand.

---
## 🔎 Examples

### ✅ Example 1  
**Title**: Regulating Shared Scooters for Spatial Equity  
**Abstract**: This study investigates how experts, policymakers, and private operators assess different regulation models for e-scooters to guarantee spatial and social equity of access to the service.  
**Label**: About Sufficiency

### ✅ Example 2  
**Title**: Leisure Time Activities and Energy Consumption  
**Abstract**: This paper examines energy use during leisure time to identify ways to reduce consumption while maintaining well-being.  
**Label**: About Sufficiency

### ✅ Example 3  
**Title**: Encouraging Children's Active Commuting  
**Abstract**: The study explores how driving children to school influences their long-term travel habits and identifies policies to encourage more sustainable commuting choices.  
**Label**: About Sufficiency

### ✅ Example 4  
**Title**: Development Patterns and Transport Energy  
**Abstract**: We show how spatial development patterns affect energy consumption for transportation and recommend urban planning approaches to reduce demand.  
**Label**: About Sufficiency

### ✅ Example 5  
**Title**: Public Investment in Bikesharing Systems  
**Abstract**: This research analyzes how public funding contributes to expanding bikesharing systems and reducing private vehicle usage.  
**Label**: About Sufficiency

### ❌ Example 6  
**Title**: High-Performance Architectural Structures  
**Abstract**: This paper reviews innovative architectural materials and techniques for aesthetic optimization.  
**Label**: Not About Sufficiency

### ❌ Example 7  
**Title**: Employee Safety in Railway Maintenance  
**Abstract**: The research proposes a new framework for improving occupational safety protocols in railway infrastructure maintenance.  
**Label**: Not About Sufficiency

### ❌ Example 8  
**Title**: Gender Wage Gap in Italian Labor Market  
**Abstract**: Using regression analysis, we explore the effects of reservation wages across genders in Italy.  
**Label**: Not About Sufficiency

### ❌ Example 9  
**Title**: Resilient Water Cycles in the Anthropocene  
**Abstract**: This paper reviews the evolution of water management across civilizations.  
**Label**: Not About Sufficiency

---

## Now classify the following:

**Title**: {title}  
**Abstract**: {abstract}  
**Label**:
"""


# Function to classify an abstract using the Ollama library
def classify_abstract(title, abstract):
    prompt = FEW_SHOT_PROMPT_TEMPLATE.format(title=title, abstract=abstract)
    response = ollama.chat(
        # model="gemma3:12b",
        # model="deepseek-r1:14b",
        model="deepseek-r1:32b",
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.2,
            "num_ctx": 128000,
            "gsdqgsg": 10000,  # not raising any error
        },  # I'm not sure at all that these parameters are used
    )
    content = response.message.content
    # Take reasoning into account
    if "</think>" in content:
        content = content.split("</think>")[-1]  # Get rid of the reasoning part
    else:
        content = content.split("\n")[0]  # Get the first line of the response with the label
    print(content)
    if "Not About Sufficiency" in content:
        return "Not About Sufficiency"
    elif "About Sufficiency" in content:
        return "About Sufficiency"
    else:
        # Handle unexpected responses
        print(f"Unexpected response: {content}")
        return f"Unknown \n\n {content}"


# Apply classification to the subset
data_subset["GPT_Classification"] = (
    data_subset.progress_apply(  # To async for allowing ollama to batch
        lambda row: classify_abstract(row["primary_title"], row["abstract"]), axis=1
    )
)

# Save results
data_subset.to_csv(os.path.join(data_path, "classified_subset_results.csv"), index=False)

print("Classification complete. Results saved to 'classified_subset_results.csv'")

true_labels = data_subset["true_label"]
predicted_labels = data_subset["GPT_Classification"]

# Generate classification report
print("Classification Report:")
print(classification_report(true_labels, predicted_labels))

# Generate confusion matrix
conf_matrix = confusion_matrix(
    true_labels, predicted_labels, labels=["About Sufficiency", "Not About Sufficiency"]
)
print("Confusion Matrix:")
print(conf_matrix)
