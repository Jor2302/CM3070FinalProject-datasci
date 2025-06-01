# webpage/word2vec_similarity.py

import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def train_word2vec_model():
    df = pd.read_csv("data/udemy_course_data.csv")
    df['text'] = df['course_title'].fillna('') + " " + df['subject'].fillna('')
    tokenized = df['text'].str.lower().str.split()

    model = Word2Vec(sentences=tokenized, vector_size=50, window=5, min_count=1, workers=2, sg=1)
    return model, tokenized, df

def get_similar_courses(input_course_title):
    model, tokenized, df = train_word2vec_model()
    input_words = input_course_title.lower().split()

    # Average vector for input
    input_vec = np.mean([model.wv[w] for w in input_words if w in model.wv], axis=0).reshape(1, -1)

    # Average vector for each course
    df['vector'] = tokenized.apply(lambda words: np.mean([model.wv[w] for w in words if w in model.wv], axis=0))
    df = df.dropna(subset=['vector'])
    df['similarity'] = df['vector'].apply(lambda vec: cosine_similarity([vec], input_vec)[0][0])

    top = df.sort_values(by='similarity', ascending=False).head(5)
    return top[['course_title', 'subject', 'similarity']].to_dict(orient="records")
