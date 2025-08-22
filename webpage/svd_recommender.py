# svd_recommender.py
from __future__ import annotations

import os
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# Robust base dir so scripts run from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _resolve(path: str) -> str:
    """Join with BASE_DIR if a relative path is given."""
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)

# Function to train an SVD collaborative filtering model
def train_svd_model(interactions_path: str = "data/Synthetic_Interactions.csv"):
    """
    Trains an SVD model with CV (RMSE/MAE), then FITS on the FULL trainset for serving.
    Returns: (model, avg_rmse, avg_mae)
    """
    interactions_path = _resolve(interactions_path)

    df = pd.read_csv(interactions_path)[["user_id", "course_id", "rating"]].dropna()
    # Surprise expects string IDs
    df["user_id"] = df["user_id"].astype(str)
    df["course_id"] = df["course_id"].astype(str)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "course_id", "rating"]], reader)

    # CV first (model re-initialized internally per fold)
    model = SVD(random_state=42)
    cv_results = cross_validate(model, data, measures=["RMSE", "MAE"], cv=5, verbose=False)
    avg_rmse = round(float(cv_results["test_rmse"].mean()), 3)
    avg_mae  = round(float(cv_results["test_mae"].mean()), 3)

    # Fit on the FULL trainset for serving predictions
    full_trainset = data.build_full_trainset()
    model.fit(full_trainset)

    return model, avg_rmse, avg_mae

# Function to generate top-N course recommendations for a user using the trained SVD model
def get_svd_predictions(
    user_id,
    model: SVD,
    course_df: pd.DataFrame,
    interactions_path: str = "data/Synthetic_Interactions.csv",
    top_n: int = 5
):
    """
    Returns a list of {course_title, explanation} for unseen courses, ranked by predicted rating.
    Assumes course_df has columns: ['course_id', 'course_title'].
    """
    interactions_path = _resolve(interactions_path)
    top_n = max(1, min(int(top_n), 50))

    # Load interactions & normalize types
    df = pd.read_csv(interactions_path)
    df["user_id"] = df["user_id"].astype(str)
    df["course_id"] = df["course_id"].astype(str)

    # Build candidate set: all courses not yet rated by this user
    user_id_str = str(user_id)
    already_rated = set(df.loc[df["user_id"] == user_id_str, "course_id"].tolist())

    # Ensure course_df ids as strings for matching
    course_df = course_df.copy()
    course_df["course_id"] = course_df["course_id"].astype(str)

    all_courses = course_df["course_id"].tolist()
    unseen = [cid for cid in all_courses if cid not in already_rated]

    # Predict ratings for unseen courses
    preds = []
    for cid in unseen:
        est = model.predict(user_id_str, cid).est
        preds.append((cid, float(est)))

    # Rank and format results
    preds.sort(key=lambda x: x[1], reverse=True)
    top = preds[:top_n]

    results = []
    for cid, score in top:
        # Safe title lookup
        title_series = course_df.loc[course_df["course_id"] == cid, "course_title"]
        title = title_series.iloc[0] if not title_series.empty else f"Course {cid}"
        results.append({
            "course_title": str(title),
            "explanation": f"Predicted rating: {round(score, 2)}"
        })

    return results
