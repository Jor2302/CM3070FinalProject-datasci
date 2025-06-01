# 📚 Educational Content Recommender System

A hybrid, data-driven recommendation system for personalized learning, developed using machine learning, NLP, and deep learning modules.

## 🧠 Features

| Technique                   | Description                                        |
| --------------------------- | -------------------------------------------------- |
| SVD Collaborative Filtering | Learns latent user-item factors                    |
| TF-IDF & BERT               | Content-based similarity using text embeddings     |
| Clustering                  | Segments users for fallback logic                  |
| Sentiment Analysis (VADER)  | Detects review polarity                            |
| Association Rule Mining     | Extracts “users who took X also take Y” patterns   |
| Word2Vec                    | Learner input vs course title semantic matching    |
| Classification              | Predicts if user will rate a course highly         |
| A/B Testing                 | Compares CF vs Hybrid recommendation effectiveness |

## 📁 Project Structure

webpage/
│
├── app.py # Flask app backend
├── recommender.py # SVD, TF-IDF, BERT logic
├── sentiment.py # VADER analysis
├── association\_rules.py # Apriori mining
├── word2vec\_similarity.py
├── classifier.py # Logistic regression
├── ab\_testing.py # T-test comparison
├── templates/ # All HTML pages
└── data/ # CSV datasets

## 🚀 How to Run

1. Clone the repo
2. Create a virtual environment
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
python app.py
```

5. Open [http://localhost:5000](http://localhost:5000)

## 📊 Sample A/B Result

| Metric           | Group A (CF) | Group B (Hybrid) |
| ---------------- | ------------ | ---------------- |
| Avg Satisfaction | 0.55         | 0.67             |
| p-value          | 0.00123      | ✅ Significant    |

## 🎓 Author

CM3070 Data Science Final Project – Lee Wei Jie Jordy
