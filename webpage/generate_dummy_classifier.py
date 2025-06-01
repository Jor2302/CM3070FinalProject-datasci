import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dummy training data
texts = [
    "I loved this course",          # engaged
    "Very helpful and clear",       # engaged
    "Terrible experience",          # not engaged
    "I was confused and bored",     # not engaged
]

labels = ["engaged", "engaged", "not engaged", "not engaged"]

# Vectorizer and model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# Save to 'models/' folder
import os
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/classifier_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Dummy classifier and vectorizer saved to models/")
