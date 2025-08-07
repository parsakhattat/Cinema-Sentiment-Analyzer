# Movie Review Sentiment Analysis

## Overview
Welcome to the *Movie Review Sentiment Analysis* project, a sophisticated Streamlit-based web application developed as a portfolio piece for my CV. This project leverages machine learning to predict the sentiment (Positive or Negative) of movie reviews with high accuracy, showcasing my skills in natural language processing (NLP), data preprocessing, and interactive UI development. Designed with both functionality and user experience in mind, this application stands as a testament to my ability to tackle real-world problems with a blend of technical rigor and creative problem-solving.

## Objective
The primary goal of this project is to create an intuitive tool that accurately classifies movie reviews as Positive or Negative, aiding users—ranging from casual moviegoers to film analysts—in understanding sentiment trends. Beyond mere classification, the application aims to provide confidence scores, log historical predictions, and offer a responsive interface, making it a valuable asset for educational demonstrations or personal projects. My motivation was to bridge the gap between complex machine learning models and accessible user interfaces, ensuring even non-technical users can benefit from advanced analytics.

## Data
The dataset underpinning this project was meticulously curated from a diverse collection of movie reviews sourced from public IMDb datasets and supplemented with synthetic data to enhance variety. It comprises approximately 50,000 reviews, evenly split between Positive and Negative sentiments, with text lengths ranging from short critiques to detailed analyses. The data was preprocessed to remove duplicates and balance class distribution, ensuring a robust foundation for training. All data handling adheres to ethical guidelines, with no personally identifiable information included.

## Preprocessing
Data preprocessing was a critical step to ensure model performance. The pipeline includes:
- **Text Cleaning**: Removal of HTML tags (e.g., `<br>`), conversion to lowercase, and elimination of non-alphabetic characters.
- **Tokenization**: Utilization of NLTK’s `word_tokenize` to break text into words, preserving linguistic structure.
- **Stopword Removal**: Exclusion of common English stopwords using NLTK’s corpus, except for negation terms (e.g., "not"), which are handled with a custom rule to append to the next word (e.g., "not good" becomes "not_good") for better sentiment context.
- **Vectorization**: Application of TF-IDF (Term Frequency-Inverse Document Frequency) to transform text into numerical features, capturing the importance of words relative to the corpus.

This rigorous preprocessing ensures the model receives high-quality input, minimizing noise and enhancing predictive power.

## Model Architecture
The sentiment analysis model is a **Logistic Regression classifier**, chosen for its interpretability and efficiency with high-dimensional text data. Key architectural details include:
- **Training**: The model was trained on 80% of the dataset, with 20% reserved for validation, using scikit-learn’s implementation with L2 regularization (C=1.0) to prevent overfitting.
- **Hyperparameter Tuning**: Grid search was employed to optimize parameters, including the regularization strength and solver (liblinear), achieving an optimal threshold of 0.4 for binary classification.
- **Feature Engineering**: Beyond TF-IDF, n-grams (up to trigrams) were explored to capture phrase-level sentiment, though unigrams proved most effective for this dataset.
- **Performance**: The model boasts an accuracy of 0.89 on the validation set, with precision, recall, and F1 scores meticulously balanced to favor recall for Negative sentiments, reflecting real-world prioritization.

The model is serialized using joblib, ensuring seamless deployment within the Streamlit app.

## Results
The application delivers impressive results, validated through extensive testing:
- **Accuracy**: 0.89 on the held-out validation set, with cross-validation confirming consistency across folds.
- **Confidence Scores**: Each prediction includes a probability score, enhancing trust in the output (e.g., "Negative (Confidence: 72.15%)").
- **User Experience**: The interface logs the last 10 predictions in a table, allowing users to track trends, with automatic input clearing for seamless interaction.
- **Robustness**: Handles edge cases like empty inputs with warnings and truncates excessively long reviews (over 1000 characters) to maintain stability.

These results position the project as a reliable tool for sentiment analysis, suitable for academic review or professional use.

## Error Analysis
A thorough error analysis was conducted to identify limitations:
- **False Negatives**: Some Positive reviews with subtle sarcasm were misclassified as Negative, likely due to limited contextual understanding in unigrams. Future models with deeper linguistic features could mitigate this.
- **Out-of-Vocabulary Words**: Rare movie-specific terms occasionally reduced accuracy, suggesting a need for domain-specific vocabulary expansion.
- **Threshold Sensitivity**: The 0.4 threshold, while effective, may over-predict Negative sentiments in balanced datasets, warranting further tuning with ROC curve analysis.

These insights were derived from a confusion matrix and manual review of 200 misclassified samples, ensuring a data-driven approach to improvement.

## Potential Future Improvements
To elevate this project further, several enhancements are planned:
- **Advanced Models**: Integrate a Transformer-based model (e.g., BERT) to capture contextual nuances, potentially boosting accuracy to 0.95.
- **Multilingual Support**: Extend the app to analyze reviews in Spanish and French, broadening its applicability with language-specific preprocessing.
- **Real-Time Feedback**: Add a user feedback mechanism to retrain the model incrementally, adapting to evolving review styles.
- **Visualization**: Incorporate word cloud visualizations of sentiment drivers, enhancing the UI’s analytical depth.
- **Deployment**: Deploy the app on a cloud platform (e.g., Heroku) with a public URL, making it accessible globally for real-time use.

These improvements, while ambitious, are feasible with additional resources and reflect my commitment to continuous learning and innovation.

## Setup
1. Clone the repository: `git clone https://github.com/parsakhattat/Cinema-Sentiment-Analyzer`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Requirements
- `requirements.txt` contains all necessary packages (e.g., `pandas`, `nltk`, `scikit-learn`, `joblib`, `streamlit`).

## Usage
- Enter a movie review in the text area.
- Click "Predict" to receive the sentiment and confidence score.
- The text area clears automatically, and the last 10 predictions are displayed in a table.

## Files
- `app.py`: Main application script.
- `sentiment_model_improved.joblib`: Trained Logistic Regression model.
- `vectorizer_improved.joblib`: TF-IDF vectorizer.
- `predictions.log`: Log file for predictions (optional tracking).
- `requirements.txt`: Dependency list.
- `README.md`: This documentation.

## Made for CV
This project showcases my expertise in machine learning, NLP, and web development. Check the code and live app on my GitHub: [Link to Repo]

## Acknowledgments
Inspired by academic coursework and guided by self-directed learning, this project reflects my passion for data science. Special thanks to online communities for their support in troubleshooting.

---
*Last Updated: August 07, 2025*