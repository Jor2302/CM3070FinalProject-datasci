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
# Keep all paths project-relative so it runs from anywhere.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ----------------------------
# Helpers: transactions & rules
# ----------------------------
def _load_transactions(csv_path: str = os.path.join(DATA_DIR, "Synthetic_Interactions.csv")) -> list[list[str]]:
    """
    Load user→course transactions from CSV and convert to a list of lists.
    Each inner list is the set of course_ids taken by one user.
    This is the input format expected by Apriori (via TransactionEncoder).
    """
    if not os.path.exists(csv_path):
        return []
    # Read only the columns we need.
    df = pd.read_csv(csv_path, usecols=["user_id", "course_id"])
    df["user_id"] = df["user_id"].astype(str)

    # Normalize course_id to clean string IDs like "123" (avoid "123.0", NaN, etc.).
    df["course_id"] = pd.to_numeric(df["course_id"], errors="coerce")
    df = df.dropna(subset=["course_id"]).copy()
    df["course_id"] = df["course_id"].astype(int).astype(str)

    # Group by user into lists of course IDs.
    return df.groupby("user_id")["course_id"].apply(list).tolist()


def _build_rules(
    min_support: float = 0.01,
    min_confidence: float = 0.10,
    *,
    csv_path: str = os.path.join(DATA_DIR, "Synthetic_Interactions.csv"),
) -> pd.DataFrame:
    """
    Build association rules using Apriori:
      1) One-hot encode transactions.
      2) Mine frequent itemsets with min_support.
      3) Generate rules with min_confidence.
      4) Sort by lift/confidence/support for stability.
    """
    transactions = _load_transactions(csv_path)
    if not transactions:
        return pd.DataFrame()

    # One-hot encode transactions (rows = users, cols = course IDs).
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    onehot = pd.DataFrame(te_ary, columns=te.columns_)

    # Find frequent itemsets (e.g., {A,B} appears ≥ min_support of users).
    freq = apriori(onehot, min_support=min_support, use_colnames=True)
    if freq.empty:
        return pd.DataFrame()

    # Turn frequent itemsets into rules (A ⇒ B) with confidence ≥ threshold.
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return rules

    # Sort for consistent display.
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)
    return rules


def load_rules(
    min_support: float = 0.01,
    min_confidence: float = 0.10,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load rules from cache if available; otherwise build and cache them.
    Caching saves time during demos and repeated requests.
    """
    cache_file = os.path.join(CACHE_DIR, f"assoc_rules_ms{min_support}_mc{min_confidence}.pkl")

    if use_cache and os.path.exists(cache_file):
        try:
            return joblib.load(cache_file)
        except Exception:
            # If cache is corrupt or incompatible, fall back to rebuild.
            pass

    rules = _build_rules(min_support=min_support, min_confidence=min_confidence)
    if use_cache and not rules.empty:
        joblib.dump(rules, cache_file)
    return rules


# ----------------------------
# Catalogue helpers (id → title)
# ----------------------------
def _read_catalog(catalog_path: str) -> pd.DataFrame:
    """
    Read a course catalog and normalize columns to:
      - course_id (str)
      - title (str)
    This is schema-tolerant: it accepts several common header names.
    """
    if not os.path.exists(catalog_path):
        return pd.DataFrame(columns=["course_id", "title"])

    df = pd.read_csv(catalog_path)

    # Normalize course_id column name.
    id_col = None
    for cand in ["course_id", "id", "courseID", "courseId"]:
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        return pd.DataFrame(columns=["course_id", "title"])

    # Normalize title/name column name.
    title_col = None
    for cand in ["title", "course_title", "name", "courseName"]:
        if cand in df.columns:
            title_col = cand
            break
    if title_col is None:
        # If missing, fall back to using the ID as a string title (never crash).
        title_col = id_col

    # Keep only the two columns and coerce types.
    df = df[[id_col, title_col]].rename(columns={id_col: "course_id", title_col: "title"}).copy()
    df["course_id"] = pd.to_numeric(df["course_id"], errors="coerce")
    df = df.dropna(subset=["course_id"])
    df["course_id"] = df["course_id"].astype(int).astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    return df


def _load_titles_map() -> Dict[str, str]:
    """
    Merge titles from multiple possible catalogs so we rarely show '(title not found)'.
    Later files in the list can overwrite earlier ones (priority order).
    """
    paths = [
        os.path.join(DATA_DIR, "udemy_courses.csv"),      # title/name/course_title
        os.path.join(DATA_DIR, "udemy_course_data.csv"),  # course_title
        os.path.join(DATA_DIR, "courses.csv"),            # title
    ]
    id2title: Dict[str, str] = {}
    for p in paths:
        df = _read_catalog(p)
        if not df.empty:
            id2title.update(dict(zip(df["course_id"], df["title"])))
    return id2title


def _course_key(x: Union[int, str]) -> str:
    """Normalize a course ID to a clean string (e.g., 123 → '123')."""
    try:
        return str(int(x))
    except Exception:
        return str(x)


def _as_str_set(s: Iterable) -> set[str]:
    """Convert an iterable of items to a set of normalized string course IDs."""
    try:
        return set(map(_course_key, s))
    except Exception:
        return set()


# ----------------------------------------
# Public API: explain_rules_for_course
# ----------------------------------------
def explain_rules_for_course(
    course_id: int | str,
    top_k: int = 10,
    *,
    min_support: float = 0.01,
    min_confidence: float = 0.10,
    # kept for compatibility; we now merge multiple catalogs automatically
    course_catalog_csv: str | None = None,
) -> List[Dict]:
    """
    Given a selected course ID, return up to top_k *related* courses with:
      - human-friendly fields (title, estimated likelihood, counts, baseline)
      - technical metrics (support, confidence, lift)

    This powers the /rules page and keeps the output explainable for users.
    """
    # Normalize input ID once.
    course_key = _course_key(course_id)

    # Load (or build) association rules.
    rules = load_rules(min_support=min_support, min_confidence=min_confidence, use_cache=True)
    if rules.empty or "antecedents" not in rules or "consequents" not in rules:
        return []

    # Keep rules where our course appears on either side, then reset index
    mask = rules["antecedents"].apply(lambda s: course_key in _as_str_set(s)) | \
           rules["consequents"].apply(lambda s: course_key in _as_str_set(s))
    if not mask.any():
        return []

    selected = rules.loc[mask].reset_index(drop=True).copy()

    # Compute sets on the filtered subset (ensures lengths align)
    ants_sel = selected["antecedents"].apply(_as_str_set)
    cons_sel = selected["consequents"].apply(_as_str_set)

    # Pick one recommended target course per rule:
    #   Prefer something from the consequents; if that's the same as the selected course,
    #   fall back to an antecedent that is different.
    def _pick_target(a_set: set[str], c_set: set[str]) -> str | None:
        for cid in c_set:
            if cid != course_key:
                return cid
        for cid in a_set:
            if cid != course_key:
                return cid
        return None

    selected["_target"] = [ _pick_target(a, c) for a, c in zip(ants_sel, cons_sel) ]
    selected = selected.dropna(subset=["_target"])
    if selected.empty:
        return []

    # Aggregate duplicate targets coming from multiple rules; keep max of each metric.
    agg = (
        selected.groupby("_target", as_index=False)
        .agg({"support": "max", "confidence": "max", "lift": "max"})
        .sort_values(["lift", "confidence", "support"], ascending=False)
        .head(max(1, min(int(top_k or 10), 50)))
        .reset_index(drop=True)
    )

    # --- Friendly statistics from the raw transactions ---
    tx = _load_transactions()  # list[list[str]]
    total_users = len(tx)
    users_selected = sum(course_key in t for t in tx)  # how many users took the selected course
    overall_counts = Counter(c for t in tx for c in set(t))  # how common each course is overall

    # Titles map: merge of multiple catalogs (id → title)
    id2title = _load_titles_map()

    # Build the final rows, mixing friendly and technical fields.
    rows: List[Dict] = []
    for _, r in agg.iterrows():
        cid = _course_key(r["_target"])
        users_both = sum((course_key in t) and (cid in t) for t in tx)  # users who took both courses
        baseline = (overall_counts.get(cid, 0) / total_users) if total_users else 0.0
        likelihood = (users_both / users_selected) if users_selected else 0.0

        rows.append({
            "consequent": int(cid) if cid.isdigit() else cid,
            "title": id2title.get(cid, "(title not found)"),
            # Friendly, explainable fields for the UI:
            "likelihood_pct": round(likelihood * 100, 1),   # “Estimated likelihood”
            "n_both": int(users_both),                      # “2/10 users took both”, etc.
            "n_selected": int(users_selected),
            "baseline_pct": round(baseline * 100, 2),       # “vs typical”
            # Technical metrics from Apriori (for transparency):
            "support": round(float(r.get("support", 0.0)), 3),
            "confidence": round(float(r.get("confidence", 0.0)), 3),
            "lift": round(float(r.get("lift", 0.0)), 3),
        })
    return rows
