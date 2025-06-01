from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import pandas as pd

def train_svd_model(interactions_path="data/Synthetic_Interactions.csv"):
    df = pd.read_csv(interactions_path)
    df = df[['user_id', 'course_id', 'rating']].dropna()
    df['user_id'] = df['user_id'].astype(str)
    df['course_id'] = df['course_id'].astype(str)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'course_id', 'rating']], reader)

    model = SVD()
    cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
    avg_rmse = round(cv_results['test_rmse'].mean(), 3)
    avg_mae = round(cv_results['test_mae'].mean(), 3)
    return model, avg_rmse, avg_mae

def get_svd_predictions(user_id, model, course_df, interactions_path="data/Synthetic_Interactions.csv", top_n=5):
    df = pd.read_csv(interactions_path)
    df['user_id'] = df['user_id'].astype(str)
    df['course_id'] = df['course_id'].astype(str)

    rated_courses = df[df['user_id'] == user_id]['course_id'].tolist()
    all_courses = course_df['course_id'].astype(str).tolist()
    unseen = [cid for cid in all_courses if cid not in rated_courses]

    predictions = []
    for cid in unseen:
        est = model.predict(user_id, cid).est
        predictions.append((cid, est))

    top = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    results = []
    for cid, score in top:
        title = course_df[course_df['course_id'].astype(str) == cid]['course_title'].values[0]
        results.append({
            "course_title": title,
            "explanation": f"Predicted rating: {round(score, 2)}"
        })
    return results
