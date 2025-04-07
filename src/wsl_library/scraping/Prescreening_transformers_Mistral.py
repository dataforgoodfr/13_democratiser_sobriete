import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer

# Define a base path and load data
BASE_DIR = "/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete/data"

non_pertinent_upperlim_df = pd.read_excel(os.path.join(BASE_DIR, "extract1000_article_non-pertinent_upperlim.xlsx"))
non_pertinent_lowerlim_df = pd.read_excel(os.path.join(BASE_DIR, "extract1000_article_non-pertinent_lowerlim.xlsx"))
df_mobility_df = pd.read_csv(os.path.join(BASE_DIR, "df_mobility_all_articles_copy_for_classification_test_subset.csv"))

# Select relevant columns and combine the data
non_pertinent_lowerlim_df = non_pertinent_lowerlim_df[['abstract', 'manual filter']]
non_pertinent_upperlim_df = non_pertinent_upperlim_df[['abstract', 'manual filter']]
df_mobility_df = df_mobility_df[['abstract', 'manual filter']]

# Combine all dataframes
combined_df = pd.concat([non_pertinent_lowerlim_df, non_pertinent_upperlim_df, df_mobility_df], ignore_index=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_df['abstract'], combined_df['manual filter'], test_size=0.2, random_state=42)

# Ensure no missing values in X_train and X_test
X_train = X_train.dropna()
X_test = X_test.dropna()
y_train = y_train.loc[X_train.index]
y_test = y_test.loc[X_test.index]

# Load the Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Encode the sentences
X_train_embeddings = model.encode(X_train.astype(str).tolist())
X_test_embeddings = model.encode(X_test.astype(str).tolist())

# Train a Logistic Regression model
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_embeddings, y_train)

# Predict and evaluate the model
y_pred = classifier.predict(X_test_embeddings)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Classification Report:\n", report)

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming y_test and y_pred are already defined
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not About Sufficiency', 'About Sufficiency'], yticklabels=['Not About Sufficiency', 'About Sufficiency'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
