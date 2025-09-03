from __future__ import annotations

import os
import re
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from markupsafe import Markup, escape  # used to safely render highlighted HTML in Jinja templates
import csv as _csv  # reserved if you later switch to python CSV engine for messy files

# ---------- logging ----------
# Basic logger so we can see when the model is trained or cache is loaded.
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("w2v")

# ---------- paths ----------
# Project-relative paths so it works no matter where you run from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "data", "udemy_course_data.csv")

# Where we cache the trained Word2Vec model and precomputed course vectors.
EMB_DIR = os.path.join(BASE_DIR, "embeddings")
MODEL_PATH = os.path.join(EMB_DIR, "word2vec.model")
DOCVEC_NPY = os.path.join(EMB_DIR, "course_vectors.npy")   # matrix of course vectors
IDX_NPY = os.path.join(EMB_DIR, "course_index.npy")        # mapping back into the dataframe rows
os.makedirs(EMB_DIR, exist_ok=True)

# ---------- globals (lazy) ----------
# We only load/train once per process. Subsequent calls reuse these.
_W2V: Optional[Word2Vec] = None            # the trained gensim Word2Vec model
_DOCVECS: Optional[np.ndarray] = None      # (N, D) normalized course vectors
_INDEX: Optional[np.ndarray] = None        # N indices mapping vectors back to _DF rows
_DF: Optional[pd.DataFrame] = None         # courses dataframe with id, title, subject

# ---------- helpers ----------
# Tokenization regex: keep only letters/numbers, lowercase, drop punctuation.
_ALNUM = re.compile(r"[a-z0-9]+")

def _tokenize(s: str) -> list[str]:
    """Turn a title into lowercase alphanumeric tokens (e.g., 'Python 101! → ['python','101'])."""
    if not isinstance(s, str):
        return []
    return _ALNUM.findall(s.lower())

def _load_df() -> pd.DataFrame:
    """
    Load the course catalogue. We use only three columns:
      - course_id (int)
      - course_title (str)
      - subject (str, can be empty)
    We also clean titles a bit to avoid URL tails or double spaces.
    """
    df = pd.read_csv(DATA_CSV, usecols=["course_id", "course_title", "subject"])
    df["course_id"] = pd.to_numeric(df["course_id"], errors="coerce")
    df = df.dropna(subset=["course_id"]).copy()
    df["course_id"] = df["course_id"].astype(int)
    df["course_title"] = df["course_title"].fillna("").astype(str)
    df["subject"] = df["subject"].fillna("").astype(str)

    # Clean accidental URL tails and collapse double spaces
    df["course_title"] = df["course_title"].str.replace(
        r"\s+https?://\S+.*$", "", regex=True
    ).str.replace(r"\s{2,}", " ", regex=True).str.strip()

    return df

def _train_and_cache() -> Tuple[Word2Vec, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Train Word2Vec on the tokenized course titles + subject.
    Then build a single vector per course (mean of token vectors), L2-normalize it,
    and cache both the model and the vectors on disk for speed.
    """
    df = _load_df()
    if df.empty:
        raise RuntimeError("udemy_course_data.csv has no rows after cleaning")

    # Text to train on = "title + subject" per course
    text = (df["course_title"] + " " + df["subject"]).astype(str)
    tokenized = [_tokenize(t) for t in text]

    # Train a small Word2Vec model (skip-gram). Vector size 64 is fast and good enough.
    model = Word2Vec(
        sentences=tokenized,
        vector_size=64,
        window=5,
        min_count=1,     # keep even rare tokens (small dataset)
        workers=1,       # deterministic across OS
        sg=1,            # 1 = skip-gram, better for small data
        seed=42,
    )

    # Build one vector per course by averaging its token vectors (simple but effective).
    docvecs, keep_idx = [], []
    for i, toks in enumerate(tokenized):
        vecs = [model.wv[w] for w in toks if w in model.wv]
        if not vecs:
            continue  # skip rows with no valid tokens
        v = np.mean(vecs, axis=0, dtype=np.float32)
        docvecs.append(v)
        keep_idx.append(i)

    # Stack vectors into matrix, and keep index mapping back to df rows.
    if not docvecs:
        docarr = np.empty((0, model.vector_size), dtype=np.float32)
        keep = np.empty((0,), dtype=np.int32)
    else:
        docarr = np.vstack(docvecs).astype(np.float32)
        keep = np.array(keep_idx, dtype=np.int32)

    # Normalize so cosine similarity = dot product (fast).
    norms = np.linalg.norm(docarr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    docarr = docarr / norms

    # Cache to disk (so next run doesn’t retrain).
    os.makedirs(EMB_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    np.save(DOCVEC_NPY, docarr)
    np.save(IDX_NPY, keep)

    log.info("Trained W2V: kept=%d, dim=%d", docarr.shape[0], model.vector_size)
    return model, docarr, keep, df

def _ensure_ready() -> None:
    """
    Ensure globals are initialized:
      - If cache files exist, load them.
      - If not, train and cache.
      - If loading fails (version mismatch/corruption), retrain safely.
    """
    global _W2V, _DOCVECS, _INDEX, _DF
    if _W2V is not None and _DOCVECS is not None and _INDEX is not None and _DF is not None:
        return
    try:
        _DF = _load_df()
        if not (os.path.exists(MODEL_PATH) and os.path.exists(DOCVEC_NPY) and os.path.exists(IDX_NPY)):
            log.info("Cache missing → training...")
            _W2V, _DOCVECS, _INDEX, _DF = _train_and_cache()
        else:
            log.info("Loading cache...")
            _W2V = Word2Vec.load(MODEL_PATH)
            _DOCVECS = np.load(DOCVEC_NPY)
            _INDEX = np.load(IDX_NPY)

            # Quick sanity checks to catch stale or corrupted cache
            if _DOCVECS.ndim != 2 or _INDEX.ndim != 1:
                raise ValueError("Corrupted cache shapes")
            if _DOCVECS.shape[0] != _INDEX.shape[0]:
                raise ValueError("Vectors and index length mismatch")
            if _INDEX.size and int(_INDEX.max()) >= len(_DF):
                raise ValueError("Index points beyond dataframe rows")
    except Exception as e:
        # If anything goes wrong, retrain to self-heal.
        log.warning("Cache load failed (%s) → retraining...", e)
        _W2V, _DOCVECS, _INDEX, _DF = _train_and_cache()

def get_title_samples(k: int = 100) -> list[str]:
    """
    Return up to k course titles for the HTML <datalist> (autocomplete).
    If something fails (e.g., file missing), return an empty list to keep the UI usable.
    """
    try:
        df = pd.read_csv(DATA_CSV, usecols=["course_title"])
        titles = df["course_title"].dropna().astype(str).unique().tolist()
        return titles[:k]
    except Exception:
        return []

def highlight_tokens(title: str, tokens: list[str]) -> Markup:
    """
    Highlight query tokens inside a course title for the UI using <mark>.
    We escape first (to prevent HTML injection), then apply a case-insensitive regex.
    """
    if not title or not tokens:
        return Markup(escape(title))
    text = escape(title)
    for t in sorted(set(tokens), key=len, reverse=True):  # longer tokens first
        if not t:
            continue
        pattern = re.compile(rf"({re.escape(t)})", re.IGNORECASE)
        # Use raw string so \1 is the regex backreference group
        text = Markup(pattern.sub(r"<mark>\1</mark>", str(text)))
    return Markup(text)

def _fuzzy_overlap_fallback(query_tokens: list[str], top_n: int) -> List[Dict]:
    """
    Fallback path when all tokens are OOV (out-of-vocabulary) for Word2Vec,
    e.g., non-English or very rare words. We rank by simple token overlap
    between the query tokens and each course's tokens (title + subject).
    """
    if _DF is None or not query_tokens:
        return []
    df = _DF.copy()

    # Pre-tokenize titles/subjects in the dataframe
    df["__tok"] = (df["course_title"] + " " + df["subject"]).str.lower().str.findall(_ALNUM)

    # Score by overlap count
    q = set(query_tokens)
    df["__score"] = df["__tok"].apply(lambda ts: len(q & set(ts)))

    # Take the best matches, build output rows (treat score as a crude "similarity")
    df = df.sort_values("__score", ascending=False).head(max(top_n, 10))
    out: List[Dict] = []
    for _, row in df.iterrows():
        score = int(row["__score"])
        if score <= 0:
            continue
        out.append({
            "course_id": int(row["course_id"]),
            "course_title": str(row["course_title"]),
            "subject": str(row.get("subject") or "Unknown"),
            "similarity": float(score),
        })
        if len(out) >= top_n:
            break
    return out

def get_similar_courses(input_course_title: str, top_n: int = 5) -> List[Dict]:
    """
    Main entry point used by the Flask route:
      - Accept a free-text course title from the user.
      - Vectorize it with Word2Vec (mean of token vectors).
      - Compute cosine similarity against all course vectors.
      - Return the top-N matches, each with id/title/subject/similarity.

    Design choices:
      • If the query tokens are all OOV, we fall back to token-overlap ranking (never crash).
      • We L2-normalize vectors so cosine = dot product for speed.
      • We take a slightly larger top slice, then filter out duplicates/self-matches.
    """
    _ensure_ready()
    assert _W2V is not None and _DOCVECS is not None and _INDEX is not None and _DF is not None

    title = (input_course_title or "").strip()
    if not title:
        return []

    # Tokenize the query and keep only tokens present in the Word2Vec vocabulary
    toks_all = _tokenize(title)
    toks_iv = [w for w in toks_all if w in _W2V.wv]

    # If nothing is in-vocab (e.g., different language), use the fuzzy fallback.
    if not toks_iv:
        return _fuzzy_overlap_fallback(toks_all, top_n)

    # Build a single query vector as the mean of word vectors, then normalize.
    v = np.mean([_W2V.wv[w] for w in toks_iv], axis=0).astype(np.float32)
    nv = np.linalg.norm(v)
    if nv == 0 or _DOCVECS.size == 0:
        return []
    v /= nv

    # Cosine similarity via a fast dot product with all course vectors.
    sims = _DOCVECS @ v
    if sims.size == 0:
        return []

    # Take a generous slice (3x), then filter duplicates and exact title matches.
    top_idx = np.argsort(-sims)[: max(top_n * 3, 15)]

    rows: List[Dict] = []
    seen_titles = set()
    for j in top_idx:
        i = int(_INDEX[j])
        if i < 0 or i >= len(_DF):
            continue
        row = _DF.iloc[i]
        title_j = str(row["course_title"])

        # Skip exact self-match and duplicates in case of near-identical rows
        if title_j.lower() == title.lower():
            continue
        if title_j in seen_titles:
            continue
        seen_titles.add(title_j)

        rows.append({
            "course_id": int(row["course_id"]),
            "course_title": title_j,
            "subject": str(row.get("subject") or "Unknown"),
            "similarity": float(sims[j]),
        })
        if len(rows) >= top_n:
            break

    return rows
