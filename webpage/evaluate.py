import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import KFold
from surprise import Dataset, Reader, SVD

from svd_recommender import train_svd_model  # your existing CV helper
from lda_topics import train_lda_model, plot_topic_summary

DATA_DIR = "data"
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# -----------------------
# Basic helpers
# -----------------------
def _load_interactions():
    df = pd.read_csv(os.path.join(DATA_DIR, "Synthetic_Interactions.csv")).dropna(subset=["rating"])
    df["user_id"] = df["user_id"].astype(str)
    df["course_id"] = df["course_id"].astype(str)
    df["rating"] = df["rating"].astype(float).clip(1, 5)
    return df

def _user_split(df, test_frac=0.2, min_per_user=5, seed=42):
    rng = np.random.default_rng(seed)
    train_rows, test_rows = [], []
    for uid, g in df.groupby("user_id"):
        g = g.sample(frac=1.0, random_state=seed)
        if len(g) >= min_per_user:
            k = max(1, int(len(g) * test_frac))
            idx = rng.choice(len(g), size=k, replace=False)
            test_rows.append(g.iloc[idx])
            mask = np.ones(len(g), dtype=bool); mask[idx] = False
            train_rows.append(g.iloc[mask])
        else:
            train_rows.append(g)
    train = pd.concat(train_rows, ignore_index=True)
    test = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=df.columns)
    return train, test

def _popularity_scores(train):
    means = train.groupby("course_id")["rating"].mean()
    counts = train.groupby("course_id")["rating"].count()
    return (means + 1e-6 * counts).sort_values(ascending=False)

def _precision_at_k(test, recs_by_user, k=5):
    if test.empty:
        return 0.0
    rel = test[test["rating"] >= 4].groupby("user_id")["course_id"].apply(set)
    vals = []
    for u, recs in recs_by_user.items():
        if u not in rel: 
            continue
        topk = recs[:k]
        hits = len(set(topk) & rel[u])
        vals.append(hits / max(1, len(topk)))
    return float(np.mean(vals)) if vals else 0.0

def _recs_from_scores(train, scores, users, top_n=50):
    seen = train.groupby("user_id")["course_id"].apply(set)
    all_ranked = list(scores.index)
    out = {}
    for u in users:
        hide = seen.get(u, set())
        ranked = [cid for cid in all_ranked if cid not in hide]
        out[u] = ranked[:top_n]
    return out

# -----------------------
# Public functions used by app.py
# -----------------------
def run_evaluation():
    """Main simple metrics (kept from your original behaviour)."""
    df = _load_interactions()
    # simulate prediction noise for a quick headline RMSE + P@3
    pred = df["rating"] + np.random.normal(0, 0.5, size=len(df))
    pred = np.clip(pred, 1.0, 5.0)
    rmse = float(np.sqrt(mean_squared_error(df["rating"], pred)))

    # Precision@3 on binarized top/not-top
    is_top = (df["rating"] >= 4).astype(int)
    pred_top = (pred >= 4).astype(int)
    precision_at3 = float((is_top & pred_top).sum() / max(1, pred_top.sum()))

    return {"rmse": round(rmse, 3), "precision": round(precision_at3, 3)}

def run_cross_validation(k=5):
    """Keep your KFold simulation for display."""
    df = _load_interactions()
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmses = []
    for tr_idx, te_idx in kf.split(df):
        test = df.iloc[te_idx].copy()
        pred = test["rating"] + np.random.normal(0, 0.5, size=len(test))
        pred = np.clip(pred, 1.0, 5.0)
        rmses.append(np.sqrt(mean_squared_error(test["rating"], pred)))
    return {"cv_rmse_mean": round(float(np.mean(rmses)), 3),
            "cv_rmse_std": round(float(np.std(rmses)), 3)}

def run_svd_evaluation():
    """Use your Surprise SVD CV to report avg RMSE/MAE."""
    _, avg_rmse, avg_mae = train_svd_model()
    return {"svd_rmse": round(float(avg_rmse), 3),
            "svd_mae": round(float(avg_mae), 3)}

def run_precision_bar_and_curves(k=5, threshold=4.0, seed=42):
    """Real offline split: make P@5 bar (Popularity vs SVD) + PR/ROC curves."""
    df = _load_interactions()
    train, test = _user_split(df, test_frac=0.2, min_per_user=5, seed=seed)

    # popularity recommendations
    pop = _popularity_scores(train)
    users = sorted(test["user_id"].unique()) if not test.empty else []
    pop_recs = _recs_from_scores(train, pop, users, top_n=50)
    p_at5_pop = _precision_at_k(test, pop_recs, k=k)

    # SVD fit on train
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train[["user_id", "course_id", "rating"]], reader)
    svd = SVD(random_state=seed)
    trainset = data.build_full_trainset()
    svd.fit(trainset)

    # P@5 with SVD ranking
    all_items = set(df["course_id"].unique())
    recs_svd = {}
    for u in users:
        seen = set(train.loc[train["user_id"] == u, "course_id"])
        cand = list(all_items - seen)
        preds = [(cid, svd.predict(u, cid).est) for cid in cand]
        preds.sort(key=lambda x: x[1], reverse=True)
        recs_svd[u] = [cid for cid, _ in preds[:50]]
    p_at5_svd = _precision_at_k(test, recs_svd, k=k)

    # error bars via bootstrap over users
    def _boot_sd(recs):
        if not users: return 0.0
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(200):
            sample = list(rng.choice(users, size=len(users), replace=True))
            ps = []
            for u in sample:
                rel = set(test[(test["user_id"] == u) & (test["rating"] >= 4)]["course_id"])
                topk = recs[u][:k] if u in recs else []
                hits = len(set(topk) & rel)
                ps.append(hits / max(1, len(topk)) if topk else 0.0)
            vals.append(np.mean(ps) if ps else 0.0)
        return float(np.std(vals, ddof=1))

    sd_pop = _boot_sd(pop_recs)
    sd_svd = _boot_sd(recs_svd)

    # bar chart
    plt.figure(figsize=(6,4))
    plt.bar(["Popularity","SVD"], [p_at5_pop, p_at5_svd], yerr=[sd_pop, sd_svd], capsize=4)
    plt.ylabel(f"Precision@{k}")
    plt.title("Top-N Precision (± SD)")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "precision_bar.png"))
    plt.close()

    # PR/ROC using SVD scores on test (binary: rating>=threshold)
    if not test.empty:
        y_true = (test["rating"].values >= threshold).astype(int)
        y_score = np.array([svd.predict(r.user_id, r.course_id).est for r in test.itertuples(index=False)])
        # scale to [0,1]
        y_score = (y_score - y_score.min()) / max(1e-8, (y_score.max() - y_score.min()))

        pr, rc, _ = precision_recall_curve(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_pr = auc(rc, pr)
        auc_roc = auc(fpr, tpr)

        # PR
        plt.figure(figsize=(5,4))
        plt.plot(rc, pr)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Precision-Recall (AUC={auc_pr:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_DIR, "pr_curve.png"))
        plt.close()

        # ROC
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"ROC (AUC={auc_roc:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_DIR, "roc_curve.png"))
        plt.close()
    else:
        auc_pr = 0.0
        auc_roc = 0.0

    return {
        "p_at5_popularity": round(p_at5_pop, 3),
        "p_at5_svd": round(p_at5_svd, 3),
        "bar_img": "precision_bar.png",
        "auc_pr": round(float(auc_pr), 3),
        "auc_roc": round(float(auc_roc), 3),
        "pr_img": "pr_curve.png" if os.path.exists(os.path.join(STATIC_DIR, "pr_curve.png")) else "",
        "roc_img": "roc_curve.png" if os.path.exists(os.path.join(STATIC_DIR, "roc_curve.png")) else "",
    }

def generate_topic_insights():
    topics = train_lda_model()
    plot_topic_summary(topics)  # should save static/lda_topics.png
    return {"ok": True}

def export_metrics_csv(metrics, cv, svd, baselines):
    rows = [
        {"metric": "RMSE (headline)", "value": metrics["rmse"]},
        {"metric": "Precision@3 (headline)", "value": metrics["precision"]},
        {"metric": "CV RMSE (mean)", "value": cv["cv_rmse_mean"]},
        {"metric": "CV RMSE (std)", "value": cv["cv_rmse_std"]},
        {"metric": "SVD RMSE (CV)", "value": svd["svd_rmse"]},
        {"metric": "SVD MAE (CV)", "value": svd["svd_mae"]},
        {"metric": "P@5 Popularity", "value": baselines["p_at5_popularity"]},
        {"metric": "P@5 SVD", "value": baselines["p_at5_svd"]},
        {"metric": "AUC(PR)", "value": baselines.get("auc_pr", 0)},
        {"metric": "AUC(ROC)", "value": baselines.get("auc_roc", 0)},
    ]
    out = os.path.join(STATIC_DIR, "evaluation_metrics.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    return {"metrics_csv": out}

def run_full_evaluation_bundle():
    metrics = run_evaluation()
    cv = run_cross_validation(k=5)
    svd = run_svd_evaluation()
    curves_and_bar = run_precision_bar_and_curves(k=5, threshold=4.0)
    generate_topic_insights()
    files = export_metrics_csv(metrics, cv, svd, curves_and_bar)

    return {
        "metrics": metrics,
        "cv": cv,
        "svd": svd,
        "baselines": curves_and_bar,
        "curves": {"auc_pr": curves_and_bar["auc_pr"], "auc_roc": curves_and_bar["auc_roc"],
                   "pr_img": curves_and_bar["pr_img"], "roc_img": curves_and_bar["roc_img"]},
        "files": files,
    }
