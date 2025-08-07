import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import pandas as pd

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Movie Review Sentiment", layout="wide")

#  CSS codes
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- TITLE & SIDEBAR --------------------
st.title("Movie Review Sentiment Predictor")
st.markdown("---")

st.sidebar.header("About")
st.sidebar.write(
    "A Logistic Regression model to predict movie review sentiment "
    "(Accuracy: 0.89, Threshold: 0.4)."
)

# -------------------- NLTK DATA CHECK --------------------
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# -------------------- LOAD MODEL & VECTORIZER --------------------
try:
    model = joblib.load("sentiment_model_improved.joblib")
    vectorizer = joblib.load("vectorizer_improved.joblib")
except FileNotFoundError:
    st.error(
        "Error: Model or vectorizer files not found. Please ensure 'sentiment_model_improved.joblib' and 'vectorizer_improved.joblib' are in the project directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or vectorizer: {str(e)}")
    st.stop()

# -------------------- CLEANING FUNCTION --------------------
stop_words = set(stopwords.words('english'))


def clean_text(text):
    if len(text) > 1000:  # Limit input length
        text = text[:1000]
    text = re.sub(r'<br\s*/><br\s*/>', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
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


# -------------------- LOG PREDICTIONS --------------------
def log_prediction(review, sentiment, confidence):
    with open("predictions.log", "a") as f:
        f.write(f"Review: {review}\nSentiment: {sentiment} (Confidence: {confidence:.2%})\n---\n")


# -------------------- SESSION STATE --------------------
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "input_key" not in st.session_state:
    st.session_state.input_key = 0  # unique key for text area

# -------------------- TEXT INPUT --------------------
user_input = st.text_area(
    "Enter your movie review:",
    height=200,
    key=f"review_input_{st.session_state.input_key}"  # dynamic key
)

# -------------------- PREDICT BUTTON --------------------
if st.button("Predict"):
    if user_input.strip():
        cleaned_input = clean_text(user_input)
        input_vec = vectorizer.transform([cleaned_input])
        y_pred_prob = model.predict_proba(input_vec)[:, 1]

        threshold = 0.4
        prediction = int(y_pred_prob >= threshold)
        probability = y_pred_prob[0] if prediction == 1 else 1 - y_pred_prob[0]
        sentiment = "Positive" if prediction == 1 else "Negative"

        log_prediction(user_input, sentiment, probability)
        st.session_state.last_prediction = f"{sentiment} (Confidence: {probability:.2%})"

        #  Clear input by incrementing key
        st.session_state.input_key += 1
        st.rerun()

# -------------------- DISPLAY RESULTS --------------------
col1, col2 = st.columns([1, 1])

with col1:
    if st.session_state.last_prediction:
        st.success(f"Last Prediction: {st.session_state.last_prediction}")

with col2:
    if os.path.exists("predictions.log"):
        with open("predictions.log", "r") as f:
            lines = f.read().split("---\n")
            predictions = [p.strip().split("\n") for p in lines if p.strip()]
            data = []
            for p in predictions[-10:]:  # Last 10 predictions
                review = next((l.replace("Review: ", "") for l in p if l.startswith("Review: ")), "")
                sentiment = next(
                    (l.replace("Sentiment: ", "").split(" (Confidence: ")[0] for l in p if l.startswith("Sentiment: ")),
                    "")
                data.append([review, sentiment])

            df_data = (
                pd.DataFrame(data, columns=["Review", "Sentiment"])
                if data else
                pd.DataFrame([["No predictions yet", ""]], columns=["Review", "Sentiment"])
            )
            st.table(df_data)

st.markdown("---")
st.write("Coded by Parsa.Khattat!")