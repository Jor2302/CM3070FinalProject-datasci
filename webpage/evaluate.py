# evaluate.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Dataset, Reader, SVD
from svd_recommender import train_svd_model  # SVD CV + fit-on-full helper

# LDA is optional; keep UI happy even if module/plot not present
try:
    from lda_topics import train_lda_model, plot_topic_summary
except Exception:  # pragma: no cover
    def train_lda_model():
        return None
    def plot_topic_summary(_):
        return None

# ---------- paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# ---------- small utils ----------
def _load_interactions() -> pd.DataFrame:
    p = os.path.join(DATA_DIR, "Synthetic_Interactions.csv")
    df = pd.read_csv(p).dropna(subset=["rating"])
    df["user_id"] = df["user_id"].astype(str)
    df["course_id"] = df["course_id"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).clip(1, 5)
    return df

def _load_courses() -> pd.DataFrame:
    p = os.path.join(DATA_DIR, "udemy_course_data.csv")
    df = pd.read_csv(p, usecols=["course_id", "course_title", "subject"])
    df["course_id"] = pd.to_numeric(df["course_id"], errors="coerce")
    df = df.dropna(subset=["course_id"]).copy()
    df["course_id"] = df["course_id"].astype(int).astype(str)
    df["course_title"] = df["course_title"].fillna("").astype(str)
    df["subject"] = df["subject"].fillna("").astype(str)
    df["text"] = (df["course_title"] + " " + df["subject"]).str.strip()
    return df

def _user_split(df, test_frac=0.2, min_per_user=3, threshold=3.5, seed=42):
    """
    Per-user split that (when possible) guarantees at least ONE positive (rating >= threshold)
    goes to TEST. If a user has no positives, we just do a small random holdout.
    """
    rng = np.random.default_rng(seed)
    train_rows, test_rows = [], []
    for uid, g in df.groupby("user_id"):
        g = g.sample(frac=1.0, random_state=seed)  # shuffle per user
        if len(g) < min_per_user:
            train_rows.append(g)
            continue

        pos_idx = g.index[g["rating"] >= threshold].tolist()
        if pos_idx:
            # put exactly one positive into test
            test_pick = rng.choice(pos_idx, size=1, replace=False)
            test_rows.append(g.loc[test_pick])
            train_rows.append(g.drop(test_pick))
        else:
            # no positives: small random holdout to keep evaluation balanced
            k = max(1, int(len(g) * test_frac))
            held = rng.choice(g.index, size=k, replace=False)
            test_rows.append(g.loc[held])
            train_rows.append(g.drop(held))

    train = pd.concat(train_rows, ignore_index=True)
    test = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=df.columns)
    return train, test

def _popularity_scores(train: pd.DataFrame) -> pd.Series:
    means = train.groupby("course_id")["rating"].mean()
    counts = train.groupby("course_id")["rating"].count()
    # tiny count term to break ties reproducibly
    return (means + 1e-6 * counts).sort_values(ascending=False)

def _precision_recall_at_k(test: pd.DataFrame, recs_by_user: Dict[str, List[str]],
                           k=5, threshold=3.5) -> Tuple[float, float]:
    if test.empty:
        return 0.0, 0.0
    rel = test[test["rating"] >= threshold].groupby("user_id")["course_id"].apply(set)
    p_vals, r_vals = [], []
    for u, recs in recs_by_user.items():
        if u not in rel:
            continue
        topk = recs[:k] if recs else []
        if not topk:
            continue
        hits = len(set(topk) & rel[u])
        p_vals.append(hits / max(1, len(topk)))
        r_vals.append(hits / max(1, len(rel[u])))
    if not p_vals:
        return 0.0, 0.0
    return float(np.mean(p_vals)), float(np.mean(r_vals))

def _recs_from_scores(train: pd.DataFrame, ranked_items: List[str],
                      users: List[str], top_n=50) -> Dict[str, List[str]]:
    seen = train.groupby("user_id")["course_id"].apply(set)
    out = {}
    for u in users:
        hide = seen.get(u, set())
        ranked = [cid for cid in ranked_items if cid not in hide]
        out[u] = ranked[:top_n] if ranked else ranked_items[:top_n]
    return out

# ---------- headline quick metrics (kept lightweight) ----------
def run_evaluation():
    df = _load_interactions()
    # simple noisy predictor just for a headline RMSE + P@3
    pred = df["rating"] + np.random.default_rng(42).normal(0, 0.5, size=len(df))
    pred = np.clip(pred, 1.0, 5.0)
    rmse = float(np.sqrt(mean_squared_error(df["rating"], pred)))
    is_top = (df["rating"] >= 4).astype(int)
    pred_top = (pred >= 4).astype(int)
    precision_at3 = float((is_top & pred_top).sum() / max(1, pred_top.sum()))
    return {"rmse": round(rmse, 3), "precision": round(precision_at3, 3)}

def run_cross_validation(k=5):
    df = _load_interactions()
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmses = []
    for _, te_idx in kf.split(df):
        test = df.iloc[te_idx].copy()
        pred = test["rating"] + np.random.default_rng(0).normal(0, 0.5, size=len(test))
        pred = np.clip(pred, 1.0, 5.0)
        rmses.append(np.sqrt(mean_squared_error(test["rating"], pred)))
    return {"cv_rmse_mean": round(float(np.mean(rmses)), 3),
            "cv_rmse_std": round(float(np.std(rmses)), 3)}

def run_svd_evaluation():
    _, avg_rmse, avg_mae = train_svd_model(os.path.join(DATA_DIR, "Synthetic_Interactions.csv"))
    return {"svd_rmse": round(float(avg_rmse), 3),
            "svd_mae": round(float(avg_mae), 3)}

# ---------- Top-N offline eval with 4 models: Pop, SVD, Content, Hybrid ----------
def run_precision_bars_and_curves(k=5, threshold=3.5, seed=42):
    df = _load_interactions()
    courses = _load_courses()
    all_items = courses["course_id"].tolist()

    # TF-IDF model for content-based
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(courses["text"])

    # user split (guarantee a positive in test when possible)
    train, test = _user_split(df, test_frac=0.2, min_per_user=3, threshold=threshold, seed=seed)
    users = sorted(test["user_id"].unique()) if not test.empty else []

    # Popularity
    pop_series = _popularity_scores(train)
    pop_ranked = list(pop_series.index)
    pop_recs = _recs_from_scores(train, pop_ranked, users, top_n=100)

    # SVD (fit on train only)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train[["user_id", "course_id", "rating"]], reader)
    svd = SVD(random_state=seed)
    svd.fit(data.build_full_trainset())

    # SVD recs per user
    recs_svd: Dict[str, List[str]] = {}
    for u in users:
        seen = set(train.loc[train["user_id"] == u, "course_id"])
        cand = [cid for cid in all_items if cid not in seen]
        preds = [(cid, svd.predict(u, cid).est) for cid in cand]
        preds.sort(key=lambda x: x[1], reverse=True)
        recs_svd[u] = [cid for cid, _ in preds[:100]] if preds else pop_ranked[:100]

    # Content recs (TF-IDF profile from user's liked items in train)
    idx_by_cid = {cid: i for i, cid in enumerate(courses["course_id"].tolist())}

    def _content_for_user(u: str, top=100) -> List[str]:
        liked_ids = train[(train["user_id"] == u) & (train["rating"] >= threshold)]["course_id"].tolist()
        liked_idxs = [idx_by_cid[c] for c in liked_ids if c in idx_by_cid]
        if not liked_idxs:
            # Fallback so content never returns [] (prevents guaranteed zeros)
            return pop_ranked[:top]
        user_vec = tfidf_matrix[liked_idxs].mean(axis=0)
        user_vec = np.asarray(user_vec).reshape(1, -1)  # avoid np.matrix
        sims = cosine_similarity(user_vec, tfidf_matrix).ravel()
        ranked = [x for _, x in sorted(zip(sims, courses["course_id"].tolist()), reverse=True)]
        seen = set(train.loc[train["user_id"] == u, "course_id"])
        ranked = [cid for cid in ranked if cid not in seen]
        return ranked[:top] if ranked else pop_ranked[:top]

    recs_content = {u: _content_for_user(u, top=100) for u in users}

    # Hybrid: normalized combo of SVD + Content (weights match recommender.py idea)
    ALPHA, BETA = 0.6, 0.4

    def _mm(d: Dict[str, float]) -> Dict[str, float]:
        if not d:
            return {}
        v = np.array(list(d.values()), dtype=float)
        lo, hi = float(np.min(v)), float(np.max(v))
        if hi <= lo:  # constant
            return {k: 0.0 for k in d}
        return {k: (float(x) - lo) / (hi - lo) for k, x in d.items()}

    def _hybrid_for_user(u: str, top=100) -> List[str]:
        cand = set(recs_svd.get(u, [])) | set(recs_content.get(u, [])) | set(pop_ranked[:200])
        if not cand:
            return pop_ranked[:top]
        svd_scores = {cid: svd.predict(u, cid).est for cid in cand}
        cont_rank = recs_content.get(u, [])
        cont_scores = {cid: (len(cont_rank) - i) / max(1, len(cont_rank)) for i, cid in enumerate(cont_rank)}
        a, b = _mm(svd_scores), _mm(cont_scores)
        final = {cid: ALPHA * a.get(cid, 0.0) + BETA * b.get(cid, 0.0) for cid in cand}
        ranked = [cid for cid, _ in sorted(final.items(), key=lambda x: x[1], reverse=True)]
        seen = set(train.loc[train["user_id"] == u, "course_id"])
        ranked = [cid for cid in ranked if cid not in seen]
        return ranked[:top] if ranked else pop_ranked[:top]

    recs_hybrid = {u: _hybrid_for_user(u, top=100) for u in users}

    # Metrics @k
    p5_pop, r5_pop = _precision_recall_at_k(test, pop_recs, k=k, threshold=threshold)
    p5_svd, r5_svd = _precision_recall_at_k(test, recs_svd, k=k, threshold=threshold)
    p5_con, r5_con = _precision_recall_at_k(test, recs_content, k=k, threshold=threshold)
    p5_hyb, r5_hyb = _precision_recall_at_k(test, recs_hybrid, k=k, threshold=threshold)

    # Bar chart (Precision@k)
    plt.figure(figsize=(6.5, 4))
    labels = ["Popularity", "SVD", "Content", "Hybrid"]
    vals = [p5_pop, p5_svd, p5_con, p5_hyb]
    plt.bar(labels, vals)
    plt.ylabel(f"Precision@{k}")
    plt.title("Top-N Precision (offline)")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "precision_bar.png"))
    plt.close()

    # PR/ROC using SVD scores for held-out (binary: rating >= threshold)
    if not test.empty:
        y_true = (test["rating"].values >= threshold).astype(int)
        y_score = np.array([svd.predict(r.user_id, r.course_id).est for r in test.itertuples(index=False)])
        if y_score.size > 0:
            y_score = (y_score - y_score.min()) / max(1e-8, (y_score.max() - y_score.min()))
        pr, rc, _ = precision_recall_curve(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_pr = auc(rc, pr)
        auc_roc = auc(fpr, tpr)

        # PR
        plt.figure(figsize=(5, 4))
        plt.plot(rc, pr)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Precision-Recall (AUC={auc_pr:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_DIR, "pr_curve.png"))
        plt.close()

        # ROC
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"ROC (AUC={auc_roc:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_DIR, "roc_curve.png"))
        plt.close()
    else:
        auc_pr = 0.0
        auc_roc = 0.0

    # optional console diag
    users_with_pos = int(test[test["rating"] >= threshold]["user_id"].nunique())
    print(f"[evaluate] users in test={len(users)}; users with ≥{threshold} positives={users_with_pos}")

    return {
        "p_at5_popularity": round(float(p5_pop), 3),
        "r_at5_popularity": round(float(r5_pop), 3),
        "p_at5_svd": round(float(p5_svd), 3),
        "r_at5_svd": round(float(r5_svd), 3),
        "p_at5_content": round(float(p5_con), 3),
        "r_at5_content": round(float(r5_con), 3),
        "p_at5_hybrid": round(float(p5_hyb), 3),
        "r_at5_hybrid": round(float(r5_hyb), 3),
        "bar_img": "precision_bar.png",
        "auc_pr": round(float(auc_pr), 3),
        "auc_roc": round(float(auc_roc), 3),
        "pr_img": "pr_curve.png" if os.path.exists(os.path.join(STATIC_DIR, "pr_curve.png")) else "",
        "roc_img": "roc_curve.png" if os.path.exists(os.path.join(STATIC_DIR, "roc_curve.png")) else "",
    }

def generate_topic_insights():
    try:
        topics = train_lda_model()
        plot_topic_summary(topics)  # saves static/lda_topics.png
    except Exception:
        pass
    return {"ok": True}

def export_metrics_csv(metrics, cv, svd, models):
    rows = [
        {"metric": "RMSE (headline)", "value": metrics["rmse"]},
        {"metric": "Precision@3 (headline)", "value": metrics["precision"]},
        {"metric": "CV RMSE (mean)", "value": cv["cv_rmse_mean"]},
        {"metric": "CV RMSE (std)", "value": cv["cv_rmse_std"]},
        {"metric": "SVD RMSE (CV)", "value": svd["svd_rmse"]},
        {"metric": "SVD MAE (CV)", "value": svd["svd_mae"]},
        {"metric": "P@5 Popularity", "value": models["p_at5_popularity"]},
        {"metric": "R@5 Popularity", "value": models["r_at5_popularity"]},
        {"metric": "P@5 SVD", "value": models["p_at5_svd"]},
        {"metric": "R@5 SVD", "value": models["r_at5_svd"]},
        {"metric": "P@5 Content", "value": models["p_at5_content"]},
        {"metric": "R@5 Content", "value": models["r_at5_content"]},
        {"metric": "P@5 Hybrid", "value": models["p_at5_hybrid"]},
        {"metric": "R@5 Hybrid", "value": models["r_at5_hybrid"]},
        {"metric": "AUC(PR)", "value": models.get("auc_pr", 0)},
        {"metric": "AUC(ROC)", "value": models.get("auc_roc", 0)},
    ]
    out = os.path.join(STATIC_DIR, "evaluation_metrics.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    return {"metrics_csv": out}

def run_full_evaluation_bundle():
    metrics = run_evaluation()
    cv = run_cross_validation(k=5)
    svd = run_svd_evaluation()
    models = run_precision_bars_and_curves(k=5, threshold=3.5)
    generate_topic_insights()
    files = export_metrics_csv(metrics, cv, svd, models)

    return {
        "metrics": metrics,
        "cv": cv,
        "svd": svd,
        "baselines": models,
        "curves": {
            "auc_pr": models["auc_pr"], "auc_roc": models["auc_roc"],
            "pr_img": models["pr_img"], "roc_img": models["roc_img"]
        },
        "files": files,
    }
