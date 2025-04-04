# approach provided by Mistral, using regression models and sentence transformers


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer

# Load the data
non_pertinent_lowerlim_df = pd.read_excel('extract1000_article_non-pertinent_lowerlim.xlsx')
non_pertinent_upperlim_df = pd.read_excel('extract1000_article_non-pertinent_upperlim.xlsx')
df_mobility_df = pd.read_csv('df_mobility_all_articles_copy_for_classification_test_subset - df_mobility_all_articles.csv')

# Add manual filter column
non_pertinent_lowerlim_df['manual filter'] = 'No'
non_pertinent_upperlim_df['manual filter'] = 'No'
df_mobility_df['manual filter'] = 'Yes'

# Select relevant columns and combine the data
non_pertinent_lowerlim_df = non_pertinent_lowerlim_df[['abstract', 'primary_title', 'manual filter']]
non_pertinent_upperlim_df = non_pertinent_upperlim_df[['abstract', 'primary_title', 'manual filter']]
df_mobility_df = df_mobility_df[['abstract', 'title', 'manual filter']]

# Rename columns to match
non_pertinent_lowerlim_df.rename(columns={'primary_title': 'title'}, inplace=True)
non_pertinent_upperlim_df.rename(columns={'primary_title': 'title'}, inplace=True)

# Combine all dataframes
combined_df = pd.concat([non_pertinent_lowerlim_df, non_pertinent_upperlim_df, df_mobility_df], ignore_index=True)

# Combine abstract and title into a single text column
combined_df['text'] = combined_df['abstract'].fillna('') + " " + combined_df['title'].fillna('')

# Map manual filter to binary labels
combined_df['label'] = combined_df['manual filter'].map({'Yes': 1, 'No': 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_df['text'], combined_df['label'], test_size=0.2, random_state=42)

# Load the Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Encode the sentences
X_train_embeddings = model.encode(X_train)
X_test_embeddings = model.encode(X_test)

# Train a Logistic Regression model
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_embeddings, y_train)

# Predict and evaluate the model
y_pred = classifier.predict(X_test_embeddings)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

