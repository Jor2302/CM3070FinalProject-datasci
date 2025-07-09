import pandas as pd
from scipy.stats import ttest_ind
import hashlib

# Function to assign users into Group A or B using hashing
def assign_group(user_id):
    return "A" if int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 2 == 0 else "B"

def run_ab_test_real_feedback():
    try:
        # Load feedback CSV
        df = pd.read_csv("data/user_feedback.csv")

        # Rename 'comment' column to 'rating' for easier reference
        df = df.rename(columns={"comment": "rating"})

        # Convert rating column to numeric (invalid values become NaN)
        df["rating"] = pd.to_numeric(df["rating"], errors='coerce')

        # Remove rows without rating or user_id
        df = df.dropna(subset=["rating", "user_id"])

        # Assign each user to Group A or B
        df["group"] = df["user_id"].astype(str).apply(assign_group)

        # Separate ratings into two groups
        group_a = df[df["group"] == "A"]["rating"]
        group_b = df[df["group"] == "B"]["rating"]

        # Print debug info: group sizes and example ratings
        print("✅ Group A size:", len(group_a))
        print("✅ Group B size:", len(group_b))
        print("👉 Sample A ratings:", group_a.head(3).tolist())
        print("👉 Sample B ratings:", group_b.head(3).tolist())

        # If either group has no data, return error
        if group_a.empty or group_b.empty:
            return {"error": "Not enough data in either group A or B."}

        # Perform independent t-test
        t_stat, p_value = ttest_ind(group_b, group_a)

        # Return summary statistics and conclusion
        return {
            "group_a_avg": round(group_a.mean(), 3),
            "group_b_avg": round(group_b.mean(), 3),
            "t_stat": round(t_stat, 3),
            "p_value": round(p_value, 5),
            "conclusion": "Significant" if p_value < 0.05 else "Not Significant"
        }

    except Exception as e:
        # Catch and return any error message
        return {"error": str(e)}
