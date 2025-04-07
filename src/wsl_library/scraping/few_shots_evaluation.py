import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the results
BASE_DIR = "/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete/data"
df = pd.read_csv(f"{BASE_DIR}/classified_subset_results_mistral.csv")

# Optional: Preview a few rows to ensure column names are correct
print(df[['Mistral_Classification', 'manual filter']].head())

# Normalize model predictions to binary (1 = About Sufficiency, 0 = Not)
def normalize_prediction(pred):
    pred = str(pred).lower()
    if "not" in pred:
        return 0
    return 1

df["Predicted_Label"] = df["Mistral_Classification"].apply(normalize_prediction)
df["True_Label"] = df["manual filter"]

# Evaluation metrics
print("\n📋 Classification Report:")
print(classification_report(df["True_Label"], df["Predicted_Label"], target_names=["Not About", "About"]))

# Confusion Matrix
cm = confusion_matrix(df["True_Label"], df["Predicted_Label"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not About", "About"])

# Plot
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Mistral Classification")
plt.show()
