import joblib

# Load trained model
model = joblib.load("models/classifier_model.pkl")  # or whatever path you're using
vectorizer = joblib.load("models/vectorizer.pkl")

def classify_text(text):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    
    # Get top non-zero words in the input
    word_list = vectorizer.get_feature_names_out()
    word_indices = X.nonzero()[1]
    top_words = [word_list[i] for i in word_indices]

    explanation = f"Prediction: {prediction.upper()} — based on words: {', '.join(top_words)}"
    return prediction, explanation
