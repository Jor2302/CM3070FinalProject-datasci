from recommender import get_recommendations
from evaluate import run_evaluation, run_cross_validation, run_baselines, run_svd_evaluation, generate_topic_insights

from sentiment_analysis import analyze_sentiment
from association_rules import explain_rules_for_course
from word2vec_similarity import get_similar_courses
from classifier import classify_text
from flask import flash, get_flashed_messages
from ab_testing import run_ab_test_real_feedback

from flask import Flask, request, jsonify, render_template, redirect, url_for
import csv
import os
from datetime import datetime
import pandas as pd
import random


app = Flask(__name__)
feedback_file = "data/user_feedback.csv"


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id', default="user_1", type=str)
    top_n = request.args.get('top_n', default=5, type=int)
    results = get_recommendations(user_id, top_n)
    return render_template('recommend.html', user_id=user_id, recommendations=results)

@app.route('/recommender_form', methods=['GET', 'POST'])
def recommender_form():
    recommendations = None
    user_id = ""
    top_n = 5

    if request.method == 'POST':
        user_id = request.form.get('user_id', 'user_1')
        top_n = int(request.form.get('top_n', 5))
        recommendations = get_recommendations(user_id, top_n)

    return render_template('recommender_form.html', user_id=user_id, top_n=top_n, recommendations=recommendations)

@app.route('/evaluate')
def evaluate_page():
    metrics = run_evaluation()
    cv = run_cross_validation()
    svd = run_svd_evaluation()
    generate_topic_insights()  # generates LDA chart

    # NEW: Load feedback data
    feedback_entries = []
    feedback_path = "data/user_feedback.csv"
    if os.path.exists(feedback_path):
        with open(feedback_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            feedback_entries = list(reader)

    return render_template('evaluate.html',
                           metrics=metrics,
                           cv=cv,
                           svd=svd,
                           feedback_entries=feedback_entries)


# Load valid course IDs for dropdown
valid_course_ids = [200742, 574194, 647884, 755198, 869312, 1239206]

@app.route('/rules', methods=['GET', 'POST'])
def rules():
    course_id = None
    explanations = []
    if request.method == 'POST':
        course_id = int(request.form.get('course_id'))
        explanations = explain_rules_for_course(course_id)
    return render_template("rules.html", course_id=course_id, explanations=explanations, course_options=valid_course_ids)


@app.route('/word2vec', methods=['GET', 'POST'])
def word2vec_page():
    results = None
    if request.method == 'POST':
        input_title = request.form.get('input_title', '')
        results = get_similar_courses(input_title)
    return render_template('word2vec.html', results=results)



@app.route('/ab_test')
def ab_test():
    results = run_ab_test_real_feedback()
    return render_template("ab_test.html", results=results)



@app.route('/user_testing')
def user_testing():
    test_data = []
    path = "data/real_users.csv"
    if os.path.exists(path):
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            test_data = list(reader)
    return render_template("user_testing.html", test_data=test_data)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        user_id = request.form['user_id']
        course_id = int(request.form['course_id'])
        feedback_text = request.form['feedback']
        rating = float(request.form['rating'])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        from sentiment_analysis import analyze_sentiment
        from classifier import classify_text

        sentiment, _ = analyze_sentiment(feedback_text)
        classification, _ = classify_text(feedback_text)

        with open('data/user_feedback.csv', 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, user_id, course_id, feedback_text, rating, sentiment, classification])

        with open('data/user_profiles.csv', 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([user_id, course_id, rating])

        return render_template(
            'feedback.html',
            show_result=True,
            sentiment=sentiment,
            classification=classification
        )


    return render_template('feedback.html', show_result=False)



@app.route('/course_list')
def course_list():
    course_df = pd.read_csv("data/udemy_course_data.csv")
    course_table = course_df[['course_id', 'course_title']].dropna().drop_duplicates().sort_values(by='course_id')
    return render_template("course_list.html", course_table=course_table)


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    user_id = request.form.get('user_id')
    course_title = request.form.get('course_title')
    feedback = request.form.get('feedback')
    feedback_comment = request.form.get('feedback_comment', '')

    from sentiment_analysis import analyze_sentiment
    from classifier import classify_text

    sentiment, sentiment_expl = analyze_sentiment(feedback_comment)
    classification, class_expl = classify_text(feedback_comment)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(feedback_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, user_id, course_title, feedback,
            feedback_comment, sentiment, classification
        ])

    # Get updated recommendations again
    results = get_recommendations(user_id, top_n=5)

    return render_template(
        'recommend.html',
        user_id=user_id,
        recommendations=results,
        show_feedback=True,
        course_title=course_title,
        comment=feedback_comment,
        sentiment=sentiment,
        classification=classification
    )

    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
