import joblib

# Load the trained classification model and vectorizer
model = joblib.load("models/classifier_model.pkl")  # Load the saved classifier model
vectorizer = joblib.load("models/vectorizer.pkl")   # Load the saved text vectorizer

def classify_text(text):
    # Transform input text into vectorized form
    X = vectorizer.transform([text])
    
    # Predict the class (e.g., engaged or not)
    prediction = model.predict(X)[0]
    
    # Extract feature names and the indices of non-zero features in input
    word_list = vectorizer.get_feature_names_out()
    word_indices = X.nonzero()[1]
    
    # Get the top contributing words from the input
    top_words = [word_list[i] for i in word_indices]

    # Create an explanation based on the top words
    explanation = f"Prediction: {prediction.upper()} — based on words: {', '.join(top_words)}"
    return prediction, explanation
