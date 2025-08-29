# association_rules.py
from __future__ import annotations

import os
from typing import List, Dict, Iterable, Union
from collections import Counter

import joblib
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# -------- paths --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ----------------------------
# Helpers: transactions & rules
# ----------------------------
def _load_transactions(csv_path: str = os.path.join(DATA_DIR, "Synthetic_Interactions.csv")) -> list[list[str]]:
    """Load user->course transactions with safe dtypes (user_id=str, course_id=str)."""
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path, usecols=["user_id", "course_id"])
    df["user_id"] = df["user_id"].astype(str)

    # course_id -> numeric -> int -> str to avoid '1.0' / NaN artifacts
    df["course_id"] = pd.to_numeric(df["course_id"], errors="coerce")
    df = df.dropna(subset=["course_id"]).copy()
    df["course_id"] = df["course_id"].astype(int).astype(str)

    # group to list of course_ids
    return df.groupby("user_id")["course_id"].apply(list).tolist()


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


# ----------------------------
# Catalogue helpers
# ----------------------------
def _read_catalog(catalog_path: str) -> pd.DataFrame:
    """
    Read a course catalogue; tolerate different column names.
    Returns DataFrame with columns: course_id(str), title(str).
    """
    if not os.path.exists(catalog_path):
        return pd.DataFrame(columns=["course_id", "title"])

    df = pd.read_csv(catalog_path)
    # Normalize course_id
    id_col = None
    for cand in ["course_id", "id", "courseID", "courseId"]:
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        return pd.DataFrame(columns=["course_id", "title"])

    # Normalize title/name column
    title_col = None
    for cand in ["title", "course_title", "name", "courseName"]:
        if cand in df.columns:
            title_col = cand
            break
    if title_col is None:
        # Still return id only so function does not crash
        title_col = id_col  # fall back to id as "title" text

    # Clean and coerce to strings
    df = df[[id_col, title_col]].rename(columns={id_col: "course_id", title_col: "title"}).copy()
    df["course_id"] = pd.to_numeric(df["course_id"], errors="coerce")
    df = df.dropna(subset=["course_id"])
    df["course_id"] = df["course_id"].astype(int).astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    return df


def _course_key(x: Union[int, str]) -> str:
    """Normalize a course id to string of int if possible, else plain str."""
    try:
        return str(int(x))
    except Exception:
        return str(x)


# ----------------------------------------
# Public: explain_rules_for_course (updated)
# ----------------------------------------
def explain_rules_for_course(
    course_id: int | str,
    top_k: int = 10,
    *,
    min_support: float = 0.01,
    min_confidence: float = 0.10,
    # default matches your project; accepts other schemas too via _read_catalog
    course_catalog_csv: str = os.path.join(DATA_DIR, "udemy_courses.csv"),
) -> List[Dict]:
    """
    Return top-k related-course rows with friendly fields:
      - title, likelihood_pct, n_both, n_selected, baseline_pct
      - support, confidence, lift (for technical detail)
    Robust to mixed ID types and catalogue schemas.
    """
    # normalize id
    course_key = _course_key(course_id)

    # --- load rules (cached) ---
    rules = load_rules(min_support=min_support, min_confidence=min_confidence, use_cache=True)
    if rules.empty or "antecedents" not in rules or "consequents" not in rules:
        return []

    # Ensure itemsets behave like sets of strings
    def _as_str_set(s: Iterable) -> set[str]:
        try:
            return set(map(_course_key, s))  # s should be frozenset from mlxtend
        except Exception:
            return set()

    # Note: rules from mlxtend already store frozensets; guard anyway
    ants = rules["antecedents"].apply(_as_str_set)
    cons = rules["consequents"].apply(_as_str_set)

    # keep rules where our course appears on either side
    mask = ants.apply(lambda s: course_key in s) | cons.apply(lambda s: course_key in s)
    if not mask.any():
        return []

    selected = rules.loc[mask].copy()
    selected = selected.assign(
        # choose "the other side" to recommend
        consequent=cons
    )

    # Flatten to one target per rule (pick any single id from consequents; if that equals the input, use one antecedent)
    def _pick_target(a_set: set[str], c_set: set[str]) -> str | None:
        # Prefer recommending from consequents
        for cid in c_set:
            if cid != course_key:
                return cid
        # Fall back to an antecedent different from the input
        for cid in a_set:
            if cid != course_key:
                return cid
        return None

    selected = selected.assign(
        _target=[_pick_target(a, c) for a, c in zip(ants, cons)]
    ).dropna(subset=["_target"])

    if selected.empty:
        return []

    # Aggregate per target to avoid duplicates from multiple rules
    agg = (
        selected.groupby("_target", as_index=False)
        .agg({"support": "max", "confidence": "max", "lift": "max"})
        .sort_values(["lift", "confidence", "support"], ascending=False)
        .head(max(1, min(int(top_k or 10), 50)))
        .reset_index(drop=True)
    )

    # --- friendly counts from transactions ---
    tx = _load_transactions()  # list[list[str]]
    total_users = len(tx)
    users_selected = sum(course_key in t for t in tx)

    # Count per-course presence across users (unique per user)
    overall_counts = Counter(c for t in tx for c in set(t))

    # Titles map
    catalog_df = _read_catalog(course_catalog_csv)
    id2title = dict(zip(catalog_df["course_id"], catalog_df["title"])) if not catalog_df.empty else {}

    rows: List[Dict] = []
    for _, r in agg.iterrows():
        cid = _course_key(r["_target"])
        users_both = sum((course_key in t) and (cid in t) for t in tx)
        baseline = (overall_counts.get(cid, 0) / total_users) if total_users else 0.0
        likelihood = (users_both / users_selected) if users_selected else 0.0

        rows.append({
            "consequent": int(cid) if cid.isdigit() else cid,
            "title": id2title.get(cid, "(title not found)"),
            # friendly fields
            "likelihood_pct": round(likelihood * 100, 1),
            "n_both": int(users_both),
            "n_selected": int(users_selected),
            "baseline_pct": round(baseline * 100, 2),
            # technical metrics
            "support": round(float(r.get("support", 0.0)), 3),
            "confidence": round(float(r.get("confidence", 0.0)), 3),
            "lift": round(float(r.get("lift", 0.0)), 3),
        })
    return rows
