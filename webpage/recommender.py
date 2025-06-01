import os
os.environ["USE_TF"] = "0"  # Disable TensorFlow in sentence-transformers

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
course_df = pd.read_csv("data/udemy_course_data.csv")
ratings_df = pd.read_csv("data/User-Item_Rating_Matrix.csv", index_col=0)
interactions_df = pd.read_csv("data/Synthetic_Interactions.csv")

# Preprocessing
course_df['course_id'] = course_df['course_id'].astype(int)
interactions_df['course_id'] = interactions_df['course_id'].astype(int)
interactions_df['user_id'] = interactions_df['user_id'].astype(str).str.strip()
course_df['text'] = course_df['course_title'] + " " + course_df['subject'].astype(str)

# TF-IDF model
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(course_df['text'])

# BERT embeddings
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
bert_embeddings = bert_model.encode(course_df['text'].tolist(), show_progress_bar=False)

# User clustering
user_vectors = ratings_df.fillna(0).values
kmeans = KMeans(n_clusters=5, random_state=42)
user_clusters = kmeans.fit_predict(user_vectors)
user_cluster_map = dict(zip(ratings_df.index, user_clusters))

def get_recommendations(user_id, top_n=5):
    user_id = user_id.strip()
    print(f"📥 Incoming user ID: {user_id}")

    # --- Feedback-based Reranking ---
    feedback_path = "data/user_feedback.csv"
    if os.path.exists(feedback_path):
        feedback_df = pd.read_csv(feedback_path)
        positive_courses = feedback_df[
            (feedback_df['user_id'] == user_id) &
            (feedback_df['feedback'] == 'thumbs_up')
        ]['course_title'].unique()

        if len(positive_courses) > 0:
            print(f"🔁 Feedback-aware reranking using {len(positive_courses)} liked course(s)")
            liked_text = course_df[course_df['course_title'].isin(positive_courses)]['text']
            liked_vec = tfidf.transform([" ".join(liked_text)])
            similarity = cosine_similarity(liked_vec, tfidf_matrix).flatten()
            boost_indices = similarity.argsort()[-top_n:][::-1]
            feedback_boosted = [{
                "course_title": course_df.iloc[i]['course_title'],
                "explanation": "Recommended based on your liked courses (feedback)"
            } for i in boost_indices]
            return feedback_boosted

    # --- Cold-start fallback ---
    if user_id not in ratings_df.index:
        print("🧊 Cold-start detected → BERT fallback")
        course_scores = cosine_similarity([bert_embeddings[0]], bert_embeddings).flatten()
        top_indices = course_scores.argsort()[-top_n:][::-1]
        return [{
            "course_title": course_df.iloc[i]['course_title'],
            "explanation": "Cold-start recommendation via BERT"
        } for i in top_indices]

    # --- Collaborative Filtering ---
    if user_id in interactions_df['user_id'].values:
        print("🤝 Found interaction data → Collaborative Filtering")
        user_data = interactions_df[interactions_df['user_id'] == user_id]
        top_courses = user_data.sort_values(by='rating', ascending=False).head(top_n)
        results = []
        for _, row in top_courses.iterrows():
            course_info = course_df[course_df['course_id'] == row['course_id']]
            if not course_info.empty:
                results.append({
                    "course_title": course_info.iloc[0]['course_title'],
                    "explanation": f"Based on your rating of {row['rating']}"
                })
        if results:
            return results

    # --- TF-IDF fallback ---
    print("🔁 Using TF-IDF fallback")
    user_ratings = ratings_df.loc[user_id].fillna(0)
    liked_courses = [
        cid for cid in ratings_df.columns
        if cid != 'cluster' and user_ratings.get(cid, 0) > 0
    ]

    if liked_courses:
        print(f"📚 TF-IDF based on liked courses: {liked_courses}")
        liked_texts = course_df[course_df['course_id'].isin([int(cid) for cid in liked_courses])]['text']
        user_profile = tfidf.transform([" ".join(liked_texts)])
        similarity = cosine_similarity(user_profile, tfidf_matrix).flatten()
        top_indices = similarity.argsort()[-top_n:][::-1]
        return [{
            "course_title": course_df.iloc[i]['course_title'],
            "explanation": "TF-IDF content similarity"
        } for i in top_indices]

    # --- Cluster-based fallback ---
    print("🧠 Using cluster-based fallback")
    user_cluster = user_cluster_map.get(user_id, None)
    if user_cluster is not None:
        similar_users = [u for u, c in user_cluster_map.items() if c == user_cluster and u != user_id]
        similar_user_data = interactions_df[interactions_df['user_id'].isin(similar_users)]
        top_courses = (
            similar_user_data
            .groupby('course_id')['rating'].mean()
            .sort_values(ascending=False)
            .head(top_n)
        )
        results = []
        for course_id in top_courses.index:
            title = course_df[course_df['course_id'] == course_id]['course_title'].values
            if len(title) > 0:
                results.append({
                    "course_title": title[0],
                    "explanation": "Popular among users like you (cluster-based)"
                })
        if results:
            return results

    print("❌ No useful data found for this user")
    return [{
        "course_title": "No data available",
        "explanation": "User not found in any dataset"
    }]
