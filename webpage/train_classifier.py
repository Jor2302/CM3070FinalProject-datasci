import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Expanded training dataset
texts = [
    # Engaged
    "I loved this course",
    "Very helpful and clear",
    "Well explained content",
    "I was interested throughout",
    "Great instructor and examples",
    "Learned a lot from this module",
    "Super informative and engaging",
    "Clear explanations and relevant examples",
    "This course kept me focused",
    "The pace was just right",
    "Excellent teaching style",
    "Would recommend to others",
    "Fun and insightful",
    "This really helped me understand the topic",
    "Highly engaging course",

    # Not engaged
    "Terrible experience",
    "I was confused and bored",
    "Too fast and unclear",
    "Didn't enjoy the course",
    "Waste of time",
    "The content was all over the place",
    "Lecturer was monotone",
    "Not relevant to my needs",
    "Disappointing and poorly structured",
    "It was hard to follow",
    "Too theoretical and no examples",
    "Lost interest halfway through",
    "Would not recommend",
    "Slides were messy and outdated",
    "No real takeaways from this",
]

labels = ["engaged"] * 15 + ["not engaged"] * 15

# Train vectorizer and model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# Save model and vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/classifier_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Enhanced classifier and vectorizer saved to /models/")
