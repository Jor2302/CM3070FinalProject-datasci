# Import necessary libraries
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to train Word2Vec model on course titles and subjects
def train_word2vec_model():
    df = pd.read_csv("data/udemy_course_data.csv")  # Load dataset
    df['text'] = df['course_title'].fillna('') + " " + df['subject'].fillna('')  # Combine title and subject
    tokenized = df['text'].str.lower().str.split()  # Tokenize text to lowercase word lists

    model = Word2Vec(sentences=tokenized, vector_size=50, window=5, min_count=1, workers=2, sg=1)  # Train Word2Vec
    return model, tokenized, df  # Return model, tokenized data, and original DataFrame

# Function to get top 5 similar courses based on input course title
def get_similar_courses(input_course_title):
    model, tokenized, df = train_word2vec_model()  # Train model and get data
    input_words = input_course_title.lower().split()  # Tokenize input course title

    # Compute average Word2Vec vector for input course title
    input_vec = np.mean([model.wv[w] for w in input_words if w in model.wv], axis=0).reshape(1, -1)

    # Compute average Word2Vec vector for each course
    df['vector'] = tokenized.apply(lambda words: np.mean([model.wv[w] for w in words if w in model.wv], axis=0))
    df = df.dropna(subset=['vector'])  # Remove rows where vector is NaN

    # Compute cosine similarity between input vector and each course vector
    df['similarity'] = df['vector'].apply(lambda vec: cosine_similarity([vec], input_vec)[0][0])

    top = df.sort_values(by='similarity', ascending=False).head(5)  # Select top 5 similar courses
    return top[['course_title', 'subject', 'similarity']].to_dict(orient="records")  # Return as dictionary
