from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Function to analyze sentiment of a given text
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()  # Create a VADER sentiment analyzer
    scores = analyzer.polarity_scores(text)  # Get sentiment scores (pos, neu, neg, compound)
    compound = scores['compound']  # Use the compound score as overall sentiment

    # Classify sentiment based on compound score thresholds
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    # Provide an explanation with the compound score
    explanation = f"Compound score: {compound} → {label.upper()}"
    return label, explanation  # Return sentiment label and explanation
