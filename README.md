cat > README.md <<'EOF'
# 📚 Educational Content Recommender System

A hybrid, data-driven recommendation system for personalised learning. Combines collaborative filtering (SVD), content similarity (TF-IDF with optional sentence embeddings), and association rules, with clear evaluation and A/B testing.

> CM3070 Data Science Final Project — **Lee Wei Jie Jordy** (Student ID: 230682469)

---

## 🧠 Key Features

- **Hybrid Recommender**  
  `0.6 × SVD + 0.3 × Content + 0.1 × Rules` (min-max normalised), with a popularity fallback for cold-start users.
- **Content Signals**  
  TF-IDF over `title + subject`. Optional SBERT (`all-MiniLM-L6-v2`) for stronger semantic matching.
- **Association Rules**  
  Apriori mining of “users who took X also take Y”, with **support**, **confidence**, and **lift**, plus human-readable explanations and likelihoods.
- **A/B Testing**  
  Welch’s t-test, 95% bootstrap CI for mean difference, and Cohen’s *d* effect size on real user feedback.
- **Evaluation**  
  RMSE & MAE (SVD CV), Precision@5 and Recall@5 for Popularity / SVD / Content / Hybrid, with PR & ROC curves and a CSV export.
- **Explainability**  
  Rules page shows titles and likelihoods; recommender surfaces “why” (SVD/content/rules).
- **Mini Chat Widget**  
  Bottom-right floating assistant that does instant keyword-to-course suggestions backed by TF-IDF.

---

## 📁 Project Structure

webpage/
├── app.py # Flask app (routes & wiring)
├── recommender.py # Hybrid (SVD + content + rules) + cold-start fallback
├── svd_recommender.py # Surprise SVD: CV + fit-on-full for serving
├── association_rules.py # Apriori mining + human-readable explanations
├── word2vec_similarity.py # Course-title similarity page (optional)
├── classifier.py # Simple text classifier (used in feedback flow)
├── sentiment_analysis.py # VADER sentiment for feedback
├── lda_topics.py # Topic modelling visual for /evaluate
├── ab_testing.py # Welch t, CI, Cohen's d; robust to sparse data
├── evaluate.py # RMSE/MAE, P@5/R@5, PR/ROC, CSV export
├── chat_assistant.py # TF-IDF mini chat assistant logic
├── templates/ # Jinja2 templates (index, rules, evaluate, ab_test, ...)
│ ├── _nav.html
│ ├── _chat_widget.html # Floating chat UI (included in pages)
│ ├── index.html
│ ├── ...
├── static/ # Generated figures (PR/ROC, precision bars, LDA) + assets
└── data/ # CSVs (see below)


**Required data files (place under `webpage/data/`):**
- `udemy_course_data.csv` — columns: `course_id`, `course_title`, `subject`  
  *The mini chat also accepts `udemy_courses.csv` or `courses.csv` with (`course_id`,`title`,`subject`).*
- `Synthetic_Interactions.csv` — columns: `user_id`, `course_id`, `rating`
- `User-Item_Rating_Matrix.csv` — *(optional)* wide matrix for clustering/analysis
- `user_feedback.csv` — appended by the app; used by A/B test and feedback re-ranking
- `real_users.csv` — table for `/user_testing`

> Embeddings cache is created on demand under `webpage/embeddings/`.

---

## 🛠️ Installation

**Python:** 3.10+ recommended  
**OS:** Windows / macOS / Linux

```bash
# from the project root (folder containing app.py)
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

# from the project root
python app.py

Main Routes

/ — Home dashboard with links to all tools and the floating chat widget

/recommender_form — Learner recommender interface

/word2vec — Course-title similarity explorer

/rules — Association rules browser with explanations & likelihoods

/evaluate — Runs evaluation bundle, saves charts & CSV, and shows metrics

/ab_test — A/B testing dashboard for real feedback

/feedback — Submit user feedback

/user_testing — Lists feedback rows and shows a Ratings Distribution bar chart

/chat (POST) — Mini chat assistant API: {message, top_k} → {reply, items, link?}
