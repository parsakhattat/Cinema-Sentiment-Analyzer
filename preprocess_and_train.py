import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import joblib
import os

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")

# Clean text with progress and negation handling
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = re.sub(r'<br\s*/><br\s*/>', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    # Simple negation handling: combine "not" with next word
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i].lower() == 'not':
            new_tokens.append(tokens[i] + "_" + tokens[i + 1])
            i += 2
        else:
            if tokens[i].isalpha() and tokens[i].lower() not in stop_words:
                new_tokens.append(tokens[i])
            i += 1
    return ' '.join(new_tokens)

tqdm.pandas(desc="Cleaning Reviews")
df['cleaned_review'] = df['review'].progress_apply(clean_text)

# Convert to TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Add bi-grams
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment'].map({'positive': 1, 'negative': 0})

# Print progress
print(f"TF-IDF matrix shape: {X.shape}")
print(f"Sample features: {vectorizer.get_feature_names_out()[:10]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load or train model
model_file = "sentiment_model_improved.joblib"
vectorizer_file = "vectorizer_improved.joblib"
if os.path.exists(model_file) and os.path.exists(vectorizer_file):
    print("Loading saved model and vectorizer...")
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
else:
    print("Training model... (This may take a moment)")
    max_iter = 100
    model = LogisticRegression(max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    print("Training completed, saving model and vectorizer...")
    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)

# Evaluate with custom threshold
y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability of Positive
threshold = 0.4  # Lower threshold for stricter Positive classification
y_pred = (y_pred_prob >= threshold).astype(int)
print(f"Accuracy (threshold {threshold}): {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix_improved.png')
plt.show()