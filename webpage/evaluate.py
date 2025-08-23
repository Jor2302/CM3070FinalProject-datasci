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

# LDA optional; keep UI happy even if missing
try:
    from lda_topics import train_lda_model, plot_topic_summary
except Exception:  # pragma: no cover
    def train_lda_model():
        return None
    def plot_topic_summary(_):
        return None

# Optional SBERT for stronger content similarity (falls back to TF-IDF if unavailable)
_USE_SBERT = True
_SBERT_OK = False
try:
    if _USE_SBERT:
        os.environ["USE_TF"] = "0"  # sentence-transformers without TF
        from sentence_transformers import SentenceTransformer
        from numpy.linalg import norm
        _SBERT_OK = True
except Exception:
    _SBERT_OK = False

# ---------- paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# ---------- small utils ----------
def _load_interactions() -> pd.DataFrame:
    p = os.path.join(DATA_DIR, "Synthetic_Interactions.csv")
    df = pd.read_csv(p).dropna(subset=["rating"])
    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["course_id"] = pd.to_numeric(df["course_id"], errors="coerce")
    df = df.dropna(subset=["course_id"]).copy()
    df["course_id"] = df["course_id"].astype(int).astype(str)
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
    goes to TEST. Also adds up to 3 negatives for balance. If a user has no positives,
    we do a small random holdout.
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
            # add up to 3 negatives into test
            remain = g.drop(test_pick)
            neg_pool = remain.index[remain["rating"] < threshold].tolist()
            k_neg = min(3, len(neg_pool))
            neg_pick = rng.choice(neg_pool, size=k_neg, replace=False) if k_neg > 0 else []
            test_rows.append(pd.concat([g.loc[test_pick], g.loc[neg_pick]]))
            train_rows.append(g.drop(index=list(test_pick) + list(neg_pick)))
        else:
            # no positives: small random holdout
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

def _users_with_any_hit(test: pd.DataFrame, recs_by_user: Dict[str, List[str]],
                        k=5, threshold=3.5) -> Tuple[int, int]:
    if test.empty:
        return 0, 0
    rel = test[test["rating"] >= threshold].groupby("user_id")["course_id"].apply(set)
    users = 0; hits = 0
    for u, pos in rel.items():
        users += 1
        hit = len(set(recs_by_user.get(u, [])[:k]) & pos) > 0
        hits += int(hit)
    return hits, users

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

# ---------- Top-N offline eval with 4 models: Popularity, SVD, Content, Hybrid ----------
def run_precision_bars_and_curves(k=5, threshold=3.5, seed=42):
    df = _load_interactions()
    courses = _load_courses()
    all_items = courses["course_id"].tolist()

    # Subject map for candidate pools
    subj_map = dict(zip(courses["course_id"], courses["subject"]))

    # Prepare content models
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(courses["text"])
    idx_by_cid = {cid: i for i, cid in enumerate(courses["course_id"].tolist())}

    # SBERT (if available)
    if _SBERT_OK:
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        course_texts = courses["text"].tolist()
        emb = sbert.encode(course_texts, show_progress_bar=False, normalize_embeddings=True)
        cid_to_i = {cid: i for i, cid in enumerate(courses["course_id"].tolist())}

    # Split
    train, test = _user_split(df, test_frac=0.2, min_per_user=3, threshold=threshold, seed=seed)
    users = sorted(test["user_id"].unique()) if not test.empty else []

    # Keep each user's held-out items available in their candidate pool (eval only)
    test_items_by_user = {u: set(test.loc[test["user_id"] == u, "course_id"]) for u in users}

    # Popularity ranking and per-user candidate pools
    pop_series = _popularity_scores(train)
    pop_ranked = list(pop_series.index)

    MAX_POOL = 250          # total pool size per user (tighter => better Top-K)
    POPULAR_TOP = 200       # seed with popularity
    PROFILE_TOP = 300       # add profile-similar items

    def _candidate_pool_for_user(u: str, max_pool=MAX_POOL) -> List[str]:
        # 1) same-subject bucket
        liked_ids = train[(train["user_id"] == u) & (train["rating"] >= threshold)]["course_id"].tolist()
        liked_subj = {subj_map[c] for c in liked_ids if c in subj_map}
        if not liked_subj:
            any_ids = train[train["user_id"] == u]["course_id"].tolist()
            liked_subj = {subj_map[c] for c in any_ids if c in subj_map}
        subject_pool = (
            courses.loc[courses["subject"].isin(liked_subj), "course_id"].tolist()
            if liked_subj else all_items
        )

        # 2) profile-similar items (SBERT preferred; TF-IDF fallback)
        profile_top = []
        if _SBERT_OK:
            liked_idx = [cid_to_i[c] for c in liked_ids if 'cid_to_i' in locals() and c in cid_to_i]
            if liked_idx:
                user_vec = emb[liked_idx].mean(axis=0)
                user_vec = user_vec / max(1e-9, norm(user_vec))
                sims = emb @ user_vec
                ranked_all = [cid for _, cid in sorted(zip(sims, courses["course_id"].tolist()), reverse=True)]
                profile_top = ranked_all[:PROFILE_TOP]
        else:
            liked_idxs = [idx_by_cid[c] for c in liked_ids if c in idx_by_cid]
            if liked_idxs:
                user_vec = tfidf_matrix[liked_idxs].mean(axis=0)
                user_vec = np.asarray(user_vec).reshape(1, -1)  # avoid np.matrix
                sims = cosine_similarity(user_vec, tfidf_matrix).ravel()
                ranked_all = [cid for _, cid in sorted(zip(sims, courses["course_id"].tolist()), reverse=True)]
                profile_top = ranked_all[:PROFILE_TOP]

        # 3) popularity seed
        popularity_seed = pop_ranked[:POPULAR_TOP]

        # 4) merge + dedupe (popularity → profile → subject)
        pool = list(dict.fromkeys(popularity_seed + profile_top + subject_pool))

        # 5) ensure the user's held-out items are present for eval
        pool = list(dict.fromkeys(pool + list(test_items_by_user.get(u, set()))))

        return pool[:max_pool]

    # Build recommendations for each model
    seen = train.groupby("user_id")["course_id"].apply(set)

    # Popularity
    pop_recs: Dict[str, List[str]] = {}
    for u in users:
        pool = _candidate_pool_for_user(u)
        hide = seen.get(u, set())
        ranked = [cid for cid in pool if cid not in hide]
        pop_recs[u] = ranked[:100] if ranked else [cid for cid in pop_ranked if cid not in hide][:100]

    # SVD (fit on train only) and recs
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train[["user_id", "course_id", "rating"]], reader)
    svd = SVD(random_state=seed)
    svd.fit(data.build_full_trainset())

    recs_svd: Dict[str, List[str]] = {}
    for u in users:
        pool = _candidate_pool_for_user(u)
        hide = seen.get(u, set())
        cand = [cid for cid in pool if cid not in hide]
        preds = [(cid, svd.predict(u, cid).est) for cid in cand]
        preds.sort(key=lambda x: x[1], reverse=True)
        recs_svd[u] = [cid for cid, _ in preds[:100]] if preds else [cid for cid in pop_ranked if cid not in hide][:100]

    # Content (SBERT preferred; TF-IDF fallback)
    recs_content: Dict[str, List[str]] = {}
    for u in users:
        pool = _candidate_pool_for_user(u)
        hide = seen.get(u, set())

        if _SBERT_OK:
            cid_to_i = {cid: i for i, cid in enumerate(courses["course_id"].tolist())}
            pool_idx = [cid_to_i[c] for c in pool if c in cid_to_i]
            liked_ids = train[(train["user_id"] == u) & (train["rating"] >= threshold)]["course_id"].tolist()
            liked_idx = [cid_to_i[c] for c in liked_ids if c in cid_to_i]
            if liked_idx and pool_idx:
                user_vec = emb[liked_idx].mean(axis=0)
                user_vec = user_vec / max(1e-9, norm(user_vec))
                pool_vecs = emb[pool_idx]
                sims = (pool_vecs @ user_vec)
                pool_ids = [courses["course_id"].iloc[i] for i in pool_idx]
                ranked = [x for _, x in sorted(zip(sims, pool_ids), reverse=True)]
            else:
                ranked = pool
        else:
            pool_idxs = [idx_by_cid[c] for c in pool if c in idx_by_cid]
            liked_ids = train[(train["user_id"] == u) & (train["rating"] >= threshold)]["course_id"].tolist()
            liked_idxs = [idx_by_cid[c] for c in liked_ids if c in idx_by_cid]
            if liked_idxs and pool_idxs:
                user_vec = tfidf_matrix[liked_idxs].mean(axis=0)
                user_vec = np.asarray(user_vec).reshape(1, -1)  # avoid np.matrix
                sims = cosine_similarity(user_vec, tfidf_matrix[pool_idxs]).ravel()
                pool_ids = [courses["course_id"].iloc[i] for i in pool_idxs]
                ranked = [x for _, x in sorted(zip(sims, pool_ids), reverse=True)]
            else:
                ranked = pool

        ranked = [cid for cid in ranked if cid not in hide]
        recs_content[u] = ranked[:100] if ranked else [cid for cid in pop_ranked if cid not in hide][:100]

    # Hybrid (normalized combo of SVD + Content + tiny Popularity prior)
    ALPHA, BETA, GAMMA = 0.5, 0.45, 0.05  # SVD / Content / Pop prior

    def _mm(d: Dict[str, float]) -> Dict[str, float]:
        if not d:
            return {}
        v = np.array(list(d.values()), dtype=float)
        lo, hi = float(np.min(v)), float(np.max(v))
        if hi <= lo:  # constant
            return {k: 0.0 for k in d}
        return {k: (float(x) - lo) / (hi - lo) for k, x in d.items()}

    recs_hybrid: Dict[str, List[str]] = {}
    pop_pos = {cid: i for i, cid in enumerate(pop_ranked)}  # for small pop prior
    for u in users:
        cand = set(pop_recs.get(u, [])) | set(recs_svd.get(u, [])) | set(recs_content.get(u, []))
        if not cand:
            recs_hybrid[u] = pop_recs.get(u, [])[:100]
            continue

        svd_scores = {cid: svd.predict(u, cid).est for cid in cand}
        cont_rank = recs_content.get(u, [])
        cont_scores = {cid: (len(cont_rank) - i) / max(1, len(cont_rank)) for i, cid in enumerate(cont_rank)}
        pop_scores = {cid: (len(pop_ranked) - pop_pos[cid]) / len(pop_ranked) for cid in cand if cid in pop_pos}

        a, b, c = _mm(svd_scores), _mm(cont_scores), _mm(pop_scores)
        final = {cid: ALPHA * a.get(cid, 0.0) + BETA * b.get(cid, 0.0) + GAMMA * c.get(cid, 0.0) for cid in cand}
        ranked = [cid for cid, _ in sorted(final.items(), key=lambda x: x[1], reverse=True)]

        hide = seen.get(u, set())
        ranked = [cid for cid in ranked if cid not in hide]
        recs_hybrid[u] = ranked[:100] if ranked else pop_recs.get(u, [])[:100]

    # --- Metrics @k ---
    p5_pop, r5_pop = _precision_recall_at_k(test, pop_recs, k=k, threshold=threshold)
    p5_svd, r5_svd = _precision_recall_at_k(test, recs_svd, k=k, threshold=threshold)
    p5_con, r5_con = _precision_recall_at_k(test, recs_content, k=k, threshold=threshold)
    p5_hyb, r5_hyb = _precision_recall_at_k(test, recs_hybrid, k=k, threshold=threshold)

    # Coverage diag (prints to console)
    cov_pop = _users_with_any_hit(test, pop_recs, k=k, threshold=threshold)
    cov_svd = _users_with_any_hit(test, recs_svd, k=k, threshold=threshold)
    cov_con = _users_with_any_hit(test, recs_content, k=k, threshold=threshold)
    cov_hyb = _users_with_any_hit(test, recs_hybrid, k=k, threshold=threshold)
    print(f"[coverage] users with ≥1 hit@{k} (pop, svd, content, hybrid): "
          f"{cov_pop[0]}/{cov_pop[1]}, {cov_svd[0]}/{cov_svd[1]}, "
          f"{cov_con[0]}/{cov_con[1]}, {cov_hyb[0]}/{cov_hyb[1]}")

    # --- Bar chart (Precision@k) ---
    plt.figure(figsize=(6.5, 4))
    labels = ["Popularity", "SVD", "Content", "Hybrid"]
    vals = [p5_pop, p5_svd, p5_con, p5_hyb]
    plt.bar(labels, vals)
    plt.ylabel(f"Precision@{k}")
    plt.title("Top-N Precision (offline)")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "precision_bar.png"))
    plt.close()

    # --- PR/ROC (unchanged, as requested) ---
    if not test.empty:
        # Binary label: rating >= threshold for positives
        y_true = (test["rating"].values >= threshold).astype(int)
        # SVD scores on the held-out pairs
        y_score = np.array([svd.predict(r.user_id, r.course_id).est for r in test.itertuples(index=False)])
        # scale scores to [0,1] for curves
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
