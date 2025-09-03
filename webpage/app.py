# app.py
from __future__ import annotations

import os
import csv
import re
from datetime import datetime
from collections import Counter
from typing import Any, Dict, List

from flask import Flask, request, render_template, jsonify
import pandas as pd

from recommender import get_recommendations
from sentiment_analysis import analyze_sentiment
from association_rules import explain_rules_for_course
from classifier import classify_text
from ab_testing import run_ab_test_real_feedback
from evaluate import run_full_evaluation_bundle
from chat_assistant import ChatAssistant

# Word2Vec utilities
from word2vec_similarity import (
    get_title_samples,
    get_similar_courses,
    highlight_tokens,
)

app = Flask(__name__)

# --- paths & files ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

feedback_file       = os.path.join(DATA_DIR, "user_feedback.csv")
user_profiles_file  = os.path.join(DATA_DIR, "user_profiles.csv")
courses_csv         = os.path.join(DATA_DIR, "udemy_course_data.csv")  # course_id, course_title, subject
real_users_csv      = os.path.join(DATA_DIR, "real_users.csv")

# --- chat assistant ---
CATALOGS = [
    os.path.join(DATA_DIR, "udemy_courses.csv"),      # title, subject
    os.path.join(DATA_DIR, "courses.csv"),            # title, subject
    os.path.join(DATA_DIR, "udemy_course_data.csv"),  # course_title, subject (mapped in ChatAssistant)
]
CHAT = ChatAssistant(data_paths=CATALOGS)

# ---------------------------
# Helper: map course_id -> title
# ---------------------------
def title_for(course_id: int) -> str:
    try:
        df = pd.read_csv(courses_csv, usecols=["course_id", "course_title"])
        df["course_id"] = pd.to_numeric(df["course_id"], errors="coerce")
        df = df.dropna(subset=["course_id"]).astype({"course_id": int})
        m = df.loc[df["course_id"] == int(course_id), "course_title"]
        return m.iloc[0] if not m.empty else ""
    except Exception:
        return ""

# Keep your fixed list for the rules demo (or load dynamically if preferred)
valid_course_ids = [200742, 574194, 647884, 755198, 869312, 1239206]

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

# Minimal /chat endpoint (always returns JSON 200)
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    try:
        top_k = int(data.get("top_k") or 3)
    except Exception:
        top_k = 3
    try:
        res = CHAT.reply(msg, top_k=top_k)
    except Exception as e:
        # never leak a 500 to the widget
        res = {"reply": f"Chat is not ready: {e}"}
    return jsonify(res), 200

@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = request.args.get("user_id", default="user_1", type=str)
    try:
        top_n = int(request.args.get("top_n", 5))
    except Exception:
        top_n = 5
    top_n = max(1, min(top_n, 50))

    results = get_recommendations(user_id, top_n)
    return render_template("recommend.html", user_id=user_id, recommendations=results)

@app.route("/recommender_form", methods=["GET", "POST"])
def recommender_form():
    recommendations = None
    user_id = ""
    top_n = 5

    if request.method == "POST":
        user_id = request.form.get("user_id", "user_1")
        try:
            top_n = int(request.form.get("top_n", 5))
        except Exception:
            top_n = 5
        top_n = max(1, min(top_n, 50))

        recommendations = get_recommendations(user_id, top_n)

    return render_template(
        "recommender_form.html",
        user_id=user_id,
        top_n=top_n,
        recommendations=recommendations,
    )

@app.route("/evaluate")
def evaluate_page():
    ctx = run_full_evaluation_bundle()

    # (optional) show feedback log if present
    feedback_header, feedback_rows = None, []
    if os.path.exists(feedback_file):
        with open(feedback_file, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
            if rows:
                feedback_header, feedback_rows = rows[0], rows[1:]
    ctx["feedback_header"] = feedback_header
    ctx["feedback_rows"] = feedback_rows

    return render_template("evaluate.html", **ctx)

@app.route("/rules", methods=["GET", "POST"])
def rules():
    course_id = None
    selected_title = ""
    explanations = []

    if request.method == "POST":
        raw = (request.form.get("course_id") or "").strip()
        try:
            cid = int(raw)
        except ValueError:
            cid = None

        if cid in valid_course_ids:
            course_id = cid
            selected_title = title_for(cid)
            explanations = explain_rules_for_course(course_id, top_k=10)

    return render_template(
        "rules.html",
        course_id=course_id,
        selected_title=selected_title,
        explanations=explanations,
        course_options=valid_course_ids,
    )

@app.route("/word2vec", methods=["GET", "POST"])
def word2vec_page():
    results: List[Dict[str, Any]] | None = None
    input_title = ""
    top_n = 5

    # up to 100 titles in the HTML datalist for autocomplete
    suggestions = get_title_samples(100)

    if request.method == "POST":
        input_title = (request.form.get("input_title") or "").strip()

        # clamp Top-N
        try:
            top_n = int(request.form.get("top_n", 5))
        except Exception:
            top_n = 5
        top_n = max(1, min(top_n, 20))

        try:
            rows = get_similar_courses(input_title, top_n=top_n) or []
        except Exception as e:
            # Avoid 500 in the UI; show no results and a console hint
            app.logger.warning("word2vec error for %r: %s", input_title, e)
            rows = []

        # highlight matching tokens in each result title
        tokens = re.findall(r"[a-z0-9]+", input_title.lower())
        for r in rows:
            r["title_highlight"] = highlight_tokens(r.get("course_title", ""), tokens)
        results = rows

    return render_template(
        "word2vec.html",
        results=results,
        input_title=input_title,
        top_n=top_n,
        suggestions=suggestions,
    )

# Optional JSON API
@app.route("/api/word2vec", methods=["GET"])
def api_word2vec():
    title = (request.args.get("q") or "").strip()
    try:
        k = int(request.args.get("k", 5))
    except Exception:
        k = 5
    k = max(1, min(k, 20))

    try:
        rows = get_similar_courses(title, top_n=k) or []
    except Exception as e:
        app.logger.warning("api/word2vec error for %r: %s", title, e)
        rows = []

    return jsonify({
        "query": title,
        "k": k,
        "results": rows,
        "count": len(rows),
    })

@app.route("/ab_test")
def ab_test():
    csv_path = feedback_file  # same feedback CSV
    results = run_ab_test_real_feedback(csv_path=csv_path)
    return render_template("ab_test.html", results=results)

@app.route("/user_testing")
def user_testing():
    test_data: List[List[str]] = []
    rating_counts = Counter()

    if os.path.exists(real_users_csv):
        with open(real_users_csv, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            _header = next(reader, None)  # skip header if present
            for row in reader:
                # Expected columns: Timestamp, User ID, Course ID, Feedback, Rating
                test_data.append(row)
                if len(row) >= 5 and row[4]:
                    try:
                        r = int(float(row[4]))
                        if 1 <= r <= 5:
                            rating_counts[r] += 1
                    except ValueError:
                        pass

    # Always pass arrays so the template never breaks
    rating_labels = [1, 2, 3, 4, 5]
    rating_values = [rating_counts.get(k, 0) for k in rating_labels]

    return render_template(
        "user_testing.html",
        test_data=test_data,
        rating_labels=rating_labels,
        rating_values=rating_values,
    )

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        user_id = request.form["user_id"]
        course_id = int(request.form["course_id"])
        feedback_text = request.form["feedback"]
        rating = float(request.form["rating"])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        sentiment, _ = analyze_sentiment(feedback_text)
        classification, _ = classify_text(feedback_text)

        # Append to feedback and user profile CSVs
        os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
        with open(feedback_file, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [timestamp, user_id, course_id, feedback_text, rating, sentiment, classification]
            )

        os.makedirs(os.path.dirname(user_profiles_file), exist_ok=True)
        with open(user_profiles_file, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([user_id, course_id, rating])

        return render_template(
            "feedback.html",
            show_result=True,
            sentiment=sentiment,
            classification=classification,
        )

    return render_template("feedback.html", show_result=False)

@app.route("/course_list")
def course_list():
    course_df = pd.read_csv(courses_csv)
    course_table = (
        course_df[["course_id", "course_title"]]
        .dropna()
        .drop_duplicates()
        .sort_values(by="course_id")
    )
    return render_template("course_list.html", course_table=course_table)

@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    user_id = request.form.get("user_id")
    course_title = request.form.get("course_title")
    feedback_val = request.form.get("feedback")
    feedback_comment = request.form.get("feedback_comment", "")

    sentiment, sentiment_expl = analyze_sentiment(feedback_comment)
    classification, class_expl = classify_text(feedback_comment)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
    with open(feedback_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                timestamp,
                user_id,
                course_title,
                feedback_val,
                feedback_comment,
                sentiment,
                classification,
            ]
        )

    # Refresh recommendations after feedback
    results = get_recommendations(user_id or "user_1", top_n=5)

    return render_template(
        "recommend.html",
        user_id=user_id,
        recommendations=results,
        show_feedback=True,
        course_title=course_title,
        comment=feedback_comment,
        sentiment=sentiment,
        classification=classification,
    )

if __name__ == "__main__":
    # Ensure relative paths resolve from project root
    os.chdir(BASE_DIR)
    app.run(host="0.0.0.0", port=5000, debug=True)
