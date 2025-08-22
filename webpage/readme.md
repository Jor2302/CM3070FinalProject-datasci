# 📚 Educational Content Recommender System

A hybrid, data-driven recommendation system for personalised learning. Combines collaborative filtering (SVD), content similarity (TF-IDF + optional BERT), and association rules, with clear evaluation and A/B testing.

> CM3070 Data Science Final Project — **Lee Wei Jie Jordy** (Student ID: 230682469)

---

## 🧠 Key Features

- **Hybrid Recommender**: `0.6×SVD + 0.3×Content + 0.1×Rules` (min-max normalised), with popularity fallback for cold-start.
- **Content Signals**: TF-IDF over `title + subject`, optional sentence embeddings (all-MiniLM-L6-v2).
- **Association Rules**: Apriori (“users who took X also take Y”) with support, confidence, lift.
- **A/B Testing**: Welch’s t-test, 95% CI, Cohen’s d on real user feedback.
- **Evaluation**: RMSE/MAE (SVD CV), Precision@5 & Recall@5 for Popularity / SVD / Content / Hybrid, PR & ROC curves.
- **Explainability**: Rules page shows titles + likelihood; recommender shows “why” (SVD/content/rules).

---

## 📁 Project Structure

webpage/
├── app.py
├── recommender.py # Hybrid (SVD + content + rules) + cold-start fallback
├── svd_recommender.py # Surprise SVD: CV + fit-on-full for serving
├── association_rules.py # Apriori mining + human-readable explanations
├── word2vec_similarity.py # Course-title similarity page (optional helper)
├── classifier.py # Simple text classifier (used in feedback flow)
├── sentiment_analysis.py # VADER sentiment for feedback
├── lda_topics.py # Topic modelling visual (for /evaluate)
├── ab_testing.py # Welch t, CI, Cohen's d; robust to sparse data
├── evaluate.py # RMSE/MAE, P@5/R@5 tables, PR/ROC, CSV export
├── templates/ # Jinja2 templates (index, rules, word2vec, evaluate, ab_test, ...)
├── static/ # Generated figures (PR/ROC, precision bars, LDA) + assets
└── data/ # CSVs (see below)


**Required data files (place under `webpage/data/`):**
- `udemy_course_data.csv` — columns: `course_id`, `course_title`, `subject`
- `Synthetic_Interactions.csv` — columns: `user_id`, `course_id`, `rating`
- `User-Item_Rating_Matrix.csv` — (optional) wide matrix for clustering/analysis
- `user_feedback.csv` — (optional) appended by the app, used by A/B test and feedback reranking
- `real_users.csv` — (optional) demo table for `/user_testing`

> Embeddings cache is stored in `webpage/embeddings/` (created automatically).

---

## 🛠️ Installation

**Python:** 3.10+ recommended  
**OS:** Windows/macOS/Linux

```bash
# from project root (folder containing app.py)
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
