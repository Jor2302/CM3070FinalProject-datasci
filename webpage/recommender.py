# recommender.py
from __future__ import annotations
import os
os.environ["USE_TF"] = "0"  # keep sentence-transformers light on Windows

import math
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Optional (content embeddings). If missing, we fall back to TF-IDF only.
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

# Optional (rules boost). If missing, rules_boost will be empty.
try:
    from association_rules import explain_rules_for_course
except Exception:  # pragma: no cover
    def explain_rules_for_course(*args, **kwargs):
        return []  # type: ignore

# SVD (collaborative filtering) – train-on-full after CV (see svd_recommender.py I sent)
try:
    from svd_recommender import train_svd_model, svd_predict_for_user
except Exception:  # pragma: no cover
    train_svd_model = svd_predict_for_user = None  # type: ignore

# ---------------------- paths & small utils ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def P(*parts): return os.path.join(BASE_DIR, *parts)

def _minmax(d: Dict[int, float]) -> Dict[int, float]:
    if not d: return {}
    vals = np.fromiter(d.values(), dtype=float)
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax <= vmin:  # constant or single value
        return {k: 0.0 for k in d}
    rng = vmax - vmin
    return {k: (float(v) - vmin) / rng for k, v in d.items()}

# ---------------------- data loading ----------------------
# Courses
_course_df = pd.read_csv(P("data", "udemy_course_data.csv"),
                         usecols=["course_id", "course_title", "subject"])
_course_df = _course_df.dropna(subset=["course_id"]).copy()
_course_df["course_id"] = pd.to_numeric(_course_df["course_id"], errors="coerce")
_course_df = _course_df.dropna(subset=["course_id"]).copy()
_course_df["course_id"] = _course_df["course_id"].astype(int)
_course_df["course_title"] = _course_df["course_title"].fillna("").astype(str)
_course_df["subject"] = _course_df["subject"].fillna("").astype(str)
_course_df["text"] = (_course_df["course_title"] + " " + _course_df["subject"]).str.strip()

COURSE_IDS = _course_df["course_id"].tolist()

# Interactions (user_id, course_id, rating)
try:
    _inter = pd.read_csv(P("data", "Synthetic_Interactions.csv"))
    _inter["user_id"] = _inter["user_id"].astype(str).str.strip()
    _inter["course_id"] = pd.to_numeric(_inter["course_id"], errors="coerce")
    _inter = _inter.dropna(subset=["course_id"]).copy()
    _inter["course_id"] = _inter["course_id"].astype(int)
    if "rating" not in _inter.columns:
        _inter["rating"] = 0.0
except Exception:
    _inter = pd.DataFrame(columns=["user_id", "course_id", "rating"])

# Wide rating matrix (optional)
try:
    _ratings = pd.read_csv(P("data", "User-Item_Rating_Matrix.csv"), index_col=0)
    _ratings.index = _ratings.index.astype(str)
except Exception:
    _ratings = pd.DataFrame()

# ---------------------- vectorizers / embeddings ----------------------
_tfidf = TfidfVectorizer(stop_words="english")
_tfidf_matrix = _tqidf_matrix = _tfidf.fit_transform(_course_df["text"])  # (N, V)

# BERT sentence embeddings (cached)
_EMB_DIR = P("embeddings")
os.makedirs(_EMB_DIR, exist_ok=True)
_BERT_PATH = P("embeddings", "courses_minilm.npy")

if SentenceTransformer is not None:
    try:
        _bert = np.load(_BERT_PATH)
        if _bert.shape[0] != len(_course_df):
            raise ValueError("BERT cache size mismatch")
    except Exception:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _bert = _model.encode(_course_df["text"].tolist(), show_progress_bar=False, convert_to_numpy=True)
        # normalize once (cosine = dot)
        norms = np.linalg.norm(_bert, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _bert = _bert / norms
        np.save(_BERT_PATH, _bert)
else:
    _bert = None  # no BERT available

# ---------------------- clustering (optional fallback) ----------------------
try:
    if not _ratings.empty:
        _user_vectors = _ratings.fillna(0).values
        _kmeans = KMeans(n_clusters=5, random_state=42)
        _clusters = _kmeans.fit_predict(_user_vectors)
        _user_cluster_map = dict(zip(_ratings.index.tolist(), _clusters.tolist()))
    else:
        _user_cluster_map = {}
except Exception:
    _user_cluster_map = {}

# ---------------------- SVD model (lazy) ----------------------
_SVD_MODEL = None
def _svd_scores(user_id: str, candidate_ids: List[int]) -> Dict[int, float]:
    """Return {cid: est_rating}; empty if SVD unavailable."""
    global _SVD_MODEL
    if train_svd_model is None or svd_predict_for_user is None:
        return {}
    if _SVD_MODEL is None:
        _SVD_MODEL, _, _ = train_svd_model(P("data", "Synthetic_Interactions.csv"))
    try:
        # our helper returns {cid:int -> est:float}
        return svd_predict_for_user(_SVD_MODEL, str(user_id), candidate_ids)  # type: ignore
    except Exception:
        # fallback: predict directly
        out = {}
        for cid in candidate_ids:
            try:
                out[cid] = float(_SVD_MODEL.predict(str(user_id), str(cid)).est)  # type: ignore
            except Exception:
                pass
        return out

# ---------------------- components for the hybrid ----------------------
def _seen_courses(user_id: str) -> set[int]:
    seen = set()
    if not _inter.empty:
        seen |= set(_inter.loc[_inter["user_id"] == user_id, "course_id"].astype(int).tolist())
    if not _ratings.empty and user_id in _ratings.index:
        # consider any positive value as "seen" (adjust if you have explicit threshold)
        row = _ratings.loc[user_id]
        seen |= set(pd.to_numeric(row.index[row.fillna(0) > 0], errors="coerce").dropna().astype(int).tolist())
    return seen

def _popularity_scores(exclude: set[int]) -> Dict[int, float]:
    """Mean rating * log(1+count) to balance quality & volume."""
    if _inter.empty:
        return {}
    g = _inter.groupby("course_id")["rating"].agg(["mean", "count"])
    g["pop_score"] = g["mean"].fillna(0) * np.log1p(g["count"].fillna(0))
    pop = g["pop_score"].to_dict()
    for cid in list(pop.keys()):
        if cid in exclude:
            pop.pop(cid, None)
    return pop

def _feedback_liked_titles(user_id: str) -> List[str]:
    path = P("data", "user_feedback.csv")
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        # If there's a 'feedback' column with 'thumbs_up'
        if "feedback" in df.columns and "course_title" in df.columns:
            liked = df[(df["user_id"].astype(str) == user_id) & (df["feedback"] == "thumbs_up")]["course_title"]
            return liked.dropna().astype(str).unique().tolist()
        # Or use rating column if present
        if "rating" in df.columns and "course_title" in df.columns:
            liked = df[(df["user_id"].astype(str) == user_id) & (pd.to_numeric(df["rating"], errors="coerce") >= 4)]["course_title"]
            return liked.dropna().astype(str).unique().tolist()
    except Exception:
        pass
    return []

def _content_scores(user_id: str, candidate_ids: List[int]) -> Dict[int, float]:
    """TF-IDF (+ BERT if available) based similarity from user's 'liked' items."""
    # liked via feedback -> titles -> text
    liked_titles = _feedback_liked_titles(user_id)

    liked_ids: List[int] = []
    if liked_titles:
        # map liked titles to ids
        liked_ids = _course_df.loc[_course_df["course_title"].isin(liked_titles), "course_id"].astype(int).tolist()

    # if still empty, use items with positive interactions/ratings
    if not liked_ids and not _ratings.empty and user_id in _ratings.index:
        row = _ratings.loc[user_id].fillna(0)
        liked_ids = [int(c) for c in row.index if c != "cluster" and row.get(c, 0) > 0 and str(c).isdigit()]

    if not liked_ids:
        return {}

    liked_texts = _course_df.loc[_course_df["course_id"].isin(liked_ids), "text"].astype(str).tolist()
    if not liked_texts:
        return {}

    # TF-IDF similarity
    tfidf_vec = _tfidf.transform([" ".join(liked_texts)])
    tfidf_sims = cosine_similarity(tfidf_vec, _tfidf_matrix).flatten()  # len = N courses

    # Optional BERT similarity
    if _bert is not None:
        # embed the combined liked text
        from numpy.linalg import norm
        v = SentenceTransformer("all-MiniLM-L6-v2").encode([" ".join(liked_texts)], show_progress_bar=False, convert_to_numpy=True)[0]
        n = norm(v)
        if n > 0:
            v = v / n
            bert_sims = (_bert @ v)  # cosine via dot
            sims = 0.5 * tfidf_sims + 0.5 * bert_sims
        else:
            sims = tfidf_sims
    else:
        sims = tfidf_sims

    # keep only candidate ids
    idx_by_cid = {int(cid): i for i, cid in enumerate(_course_df["course_id"].tolist())}
    out = {cid: float(sims[idx_by_cid[cid]]) for cid in candidate_ids if cid in idx_by_cid}
    return out

def _rules_boost(prior_course_ids: List[int], candidate_ids: List[int]) -> Dict[int, float]:
    """Use association-rule lift as a small boost signal."""
    if not callable(explain_rules_for_course) or not prior_course_ids:
        return {}
    boost: Dict[int, float] = {}
    for cid in prior_course_ids[:5]:  # limit API calls
        try:
            rows = explain_rules_for_course(cid, top_k=50)
        except Exception:
            rows = []
        for r in rows:
            cons = r.get("consequent")
            lift = float(r.get("lift", 0.0))
            if cons is None: 
                continue
            try:
                cons = int(cons)
            except Exception:
                continue
            if cons not in candidate_ids:
                continue
            boost[cons] = max(boost.get(cons, 0.0), lift)  # take best lift for that consequent
    return boost

# ---------------------- HYBRID weights ----------------------
ALPHA, BETA, GAMMA = 0.6, 0.3, 0.1  # SVD, content, rules

def _combine(candidate_ids: List[int], svd_s: Dict[int, float], cont_s: Dict[int, float], rule_s: Dict[int, float]) -> Dict[int, float]:
    a = _minmax({k: svd_s.get(k, 0.0) for k in candidate_ids})
    b = _minmax({k: cont_s.get(k, 0.0) for k in candidate_ids})
    c = _minmax({k: rule_s.get(k, 0.0) for k in candidate_ids})
    return {k: ALPHA * a.get(k, 0.0) + BETA * b.get(k, 0.0) + GAMMA * c.get(k, 0.0) for k in candidate_ids}

# ---------------------- PUBLIC API ----------------------
def get_recommendations(user_id: str, top_n: int = 5) -> List[Dict]:
    """
    Hybrid recommender:
      - If truly new user -> popularity fallback (cold-start).
      - Else combine SVD (collab), content (TF-IDF/BERT), and rules boost.
    Returns: list of {"course_title": str, "explanation": str}
    """
    user_id = str(user_id).strip()
    top_n = max(1, min(int(top_n), 50))

    seen = _seen_courses(user_id)
    all_ids = set(COURSE_IDS)
    candidate_ids = sorted(list(all_ids - seen))
    if not candidate_ids:
        candidate_ids = COURSE_IDS  # extreme case: show something

    # True cold-start: user not in ratings nor interactions -> POPULARITY
    no_ratings = _ratings.empty or user_id not in _ratings.index
    no_inter = _inter.empty or user_id not in set(_inter["user_id"])
    if no_ratings and no_inter:
        pop = _popularity_scores(exclude=seen)
        if not pop:
            # dataset missing? Show top by course_id to avoid empty UI
            fallback = sorted(candidate_ids)[:top_n]
            return [{"course_title": _course_df.loc[_course_df["course_id"] == cid, "course_title"].iloc[0],
                     "explanation": "Cold-start fallback"} for cid in fallback]
        top = sorted(pop.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"course_title": _course_df.loc[_course_df["course_id"] == cid, "course_title"].iloc[0],
                 "explanation": "Popular with learners (cold-start)"} for cid, _ in top]

    # Build component scores
    svd_scores = _svd_scores(user_id, candidate_ids)  # may be {}
    content_scores = _content_scores(user_id, candidate_ids)  # may be {}

    # prior courses for rules: use seen (cap to a few)
    prior_for_rules = sorted(list(seen))[:5]
    rules_scores = _rules_boost(prior_for_rules, candidate_ids)  # may be {}

    final = _combine(candidate_ids, svd_scores, content_scores, rules_scores)
    ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)

    # If hybrid ended up empty (e.g., all components missing), fall back to popularity
    if not ranked:
        pop = _popularity_scores(exclude=seen)
        ranked = sorted(pop.items(), key=lambda x: x[1], reverse=True)

    top = ranked[:top_n]
    out: List[Dict] = []
    for cid, score in top:
        title = _course_df.loc[_course_df["course_id"] == int(cid), "course_title"]
        title = title.iloc[0] if not title.empty else f"Course {cid}"

        parts = []
        if cid in svd_scores: parts.append("SVD")
        if cid in content_scores: parts.append("content similarity")
        if cid in rules_scores: parts.append("co-enrollment rules")
        why = " + ".join(parts) if parts else "hybrid"
        out.append({
            "course_title": str(title),
            "explanation": f"{why}; hybrid score={round(float(score), 3)}"
        })

    return out
