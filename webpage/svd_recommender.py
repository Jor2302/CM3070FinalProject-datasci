from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import pandas as pd

# Function to train an SVD collaborative filtering model
def train_svd_model(interactions_path="data/Synthetic_Interactions.csv"):
    df = pd.read_csv(interactions_path)  # Load interaction data
    df = df[['user_id', 'course_id', 'rating']].dropna()  # Keep only relevant columns and drop missing values
    df['user_id'] = df['user_id'].astype(str)  # Ensure user IDs are strings
    df['course_id'] = df['course_id'].astype(str)  # Ensure course IDs are strings

    reader = Reader(rating_scale=(1, 5))  # Define rating scale
    data = Dataset.load_from_df(df[['user_id', 'course_id', 'rating']], reader)  # Load data into Surprise format

    model = SVD()  # Instantiate the SVD model
    cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)  # Perform cross-validation
    avg_rmse = round(cv_results['test_rmse'].mean(), 3)  # Calculate average RMSE
    avg_mae = round(cv_results['test_mae'].mean(), 3)  # Calculate average MAE
    return model, avg_rmse, avg_mae  # Return the trained model and metrics


# Function to generate top-N course recommendations for a user using the trained SVD model
def get_svd_predictions(user_id, model, course_df, interactions_path="data/Synthetic_Interactions.csv", top_n=5):
    df = pd.read_csv(interactions_path)  # Load interaction data
    df['user_id'] = df['user_id'].astype(str)  # Convert user IDs to string
    df['course_id'] = df['course_id'].astype(str)  # Convert course IDs to string

    rated_courses = df[df['user_id'] == user_id]['course_id'].tolist()  # Get list of courses the user has rated
    all_courses = course_df['course_id'].astype(str).tolist()  # Get all course IDs
    unseen = [cid for cid in all_courses if cid not in rated_courses]  # Identify courses not yet rated by the user

    predictions = []
    for cid in unseen:
        est = model.predict(user_id, cid).est  # Predict rating for each unseen course
        predictions.append((cid, est))

    # Sort by predicted rating and select top N
    top = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for cid, score in top:
        title = course_df[course_df['course_id'].astype(str) == cid]['course_title'].values[0]  # Get course title
        results.append({
            "course_title": title,
            "explanation": f"Predicted rating: {round(score, 2)}"
        })

    return results  # Return recommendations with explanations
