# association_rules.py
from __future__ import annotations

import os
from typing import List, Dict

import joblib
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


# -------- paths --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _load_transactions(csv_path: str = os.path.join(DATA_DIR, "Synthetic_Interactions.csv")) -> list[list[str]]:
    """Load user->course transactions with safe dtypes (user_id=str, course_id=str)."""
    df = pd.read_csv(csv_path, usecols=["user_id", "course_id"])
    df["user_id"] = df["user_id"].astype(str)

    # course_id -> int -> str to avoid '1.0' style ids after float coercion
    df["course_id"] = pd.to_numeric(df["course_id"], errors="coerce")
    df = df.dropna(subset=["course_id"]).copy()
    df["course_id"] = df["course_id"].astype(int).astype(str)

    # group to list of course_ids
    transactions = df.groupby("user_id")["course_id"].apply(list).tolist()
    return transactions


def _build_rules(
    min_support: float = 0.01,
    min_confidence: float = 0.10,
    *,
    csv_path: str = os.path.join(DATA_DIR, "Synthetic_Interactions.csv"),
) -> pd.DataFrame:
    """Build association rules with Apriori; return a DataFrame sorted by lift/confidence."""
    transactions = _load_transactions(csv_path)

    if not transactions:
        return pd.DataFrame()

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    onehot = pd.DataFrame(te_ary, columns=te.columns_)

    freq = apriori(onehot, min_support=min_support, use_colnames=True)
    if freq.empty:
        return pd.DataFrame()

    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return rules

    # stable ordering
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)
    return rules


def load_rules(
    min_support: float = 0.01,
    min_confidence: float = 0.10,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load (or build) association rules. Caches to cache/assoc_rules_ms{..}_mc{..}.pkl.
    """
    cache_file = os.path.join(CACHE_DIR, f"assoc_rules_ms{min_support}_mc{min_confidence}.pkl")

    if use_cache and os.path.exists(cache_file):
        try:
            return joblib.load(cache_file)
        except Exception:
            pass  # fall through to rebuild

    rules = _build_rules(min_support=min_support, min_confidence=min_confidence)
    if use_cache and not rules.empty:
        joblib.dump(rules, cache_file)
    return rules

# association_rules.py (only the function below needs replacing)

def explain_rules_for_course(
    course_id: int | str,
    top_k: int = 10,
    *,
    min_support: float = 0.01,
    min_confidence: float = 0.10,
    course_catalog_csv: str = os.path.join(DATA_DIR, "udemy_course_data.csv"),
) -> List[Dict]:
    """Return top-k recs + friendly fields: likelihood %, sample sizes, baseline %."""

    # normalise id to str
    try:
        course_key = str(int(course_id))
    except Exception:
        course_key = str(course_id)

    # --- load rules (cached) ---
    rules = load_rules(min_support=min_support, min_confidence=min_confidence, use_cache=True)
    if rules.empty:
        return []

    # filter rules that use selected course in antecedents
    mask = rules["antecedents"].apply(lambda s: course_key in s)
    filtered = rules.loc[mask].copy()
    if filtered.empty:
        return []

    # pick single consequent
    def _first_item(s):
        try:
            return next(iter(s))
        except Exception:
            return None

    filtered["consequent"] = filtered["consequents"].apply(_first_item)
    filtered = filtered.dropna(subset=["consequent"]).copy()

    # aggregate duplicates
    agg = (
        filtered.groupby("consequent", as_index=False)
        .agg({"support": "max", "confidence": "max", "lift": "max"})
        .sort_values(["lift", "confidence", "support"], ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    # --- friendly counts from transactions ---
    tx = _load_transactions()                     # list[list[str]]
    total_users = len(tx)
    users_selected = sum(course_key in t for t in tx)

    # baseline freq for each consequent (how common overall)
    from collections import Counter
    flat = [c for t in tx for c in set(t)]
    overall_counts = Counter(flat)

    # titles map
    try:
        catalog = pd.read_csv(course_catalog_csv, usecols=["course_id", "course_title"])
        catalog["course_id"] = pd.to_numeric(catalog["course_id"], errors="coerce").dropna().astype(int).astype(str)
        id2title = dict(zip(catalog["course_id"], catalog["course_title"]))
    except Exception:
        id2title = {}

    rows = []
    for _, r in agg.iterrows():
        cid = str(r["consequent"])

        users_both = sum((course_key in t) and (cid in t) for t in tx)
        baseline = overall_counts.get(cid, 0) / total_users if total_users else 0.0
        likelihood = (users_both / users_selected) if users_selected else 0.0

        rows.append({
            "consequent": int(cid) if cid.isdigit() else cid,
            "title": id2title.get(cid, "(title not found)"),
            # friendly fields
            "likelihood_pct": round(likelihood * 100, 1),   # e.g., 66.7
            "n_both": int(users_both),
            "n_selected": int(users_selected),
            "baseline_pct": round(baseline * 100, 2),       # e.g., 1.05
            # advanced (still available)
            "support": round(float(r["support"]), 3),
            "confidence": round(float(r["confidence"]), 3),
            "lift": round(float(r["lift"]), 3),
        })
    return rows
