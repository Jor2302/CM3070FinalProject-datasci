# word2vec_similarity.py
from __future__ import annotations

import os
import re
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from markupsafe import Markup, escape  # installed with Flask/Jinja

# ---------- paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "data", "udemy_course_data.csv")

EMB_DIR = os.path.join(BASE_DIR, "embeddings")
MODEL_PATH = os.path.join(EMB_DIR, "word2vec.model")
DOCVEC_NPY = os.path.join(EMB_DIR, "course_vectors.npy")
IDX_NPY = os.path.join(EMB_DIR, "course_index.npy")
os.makedirs(EMB_DIR, exist_ok=True)

# ---------- globals (lazy) ----------
_W2V: Optional[Word2Vec] = None
_DOCVECS: Optional[np.ndarray] = None   # (N, D) normalized vectors
_INDEX: Optional[np.ndarray] = None     # indices into df
_DF: Optional[pd.DataFrame] = None      # cached dataframe (id, title, subject)


def _tokenize(s: str) -> list[str]:
    """Lowercase alnum tokenization."""
    if not isinstance(s, str):
        return []
    return re.findall(r"[a-z0-9]+", s.lower())


def _load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, usecols=["course_id", "course_title", "subject"])
    df["course_id"] = pd.to_numeric(df["course_id"], errors="coerce")
    df = df.dropna(subset=["course_id"]).copy()
    df["course_id"] = df["course_id"].astype(int)
    df["course_title"] = df["course_title"].fillna("").astype(str)
    df["subject"] = df["subject"].fillna("").astype(str)
    return df


def _train_and_cache() -> tuple[Word2Vec, np.ndarray, np.ndarray, pd.DataFrame]:
    df = _load_df()
    text = (df["course_title"] + " " + df["subject"]).astype(str)
    tokenized = [_tokenize(t) for t in text]

    model = Word2Vec(
        sentences=tokenized,
        vector_size=64,
        window=5,
        min_count=1,
        workers=1,  # stable on Windows
        sg=1,
        seed=42,
    )

    docvecs = []
    keep_idx = []
    for i, toks in enumerate(tokenized):
        vecs = [model.wv[w] for w in toks if w in model.wv]
        if not vecs:
            continue
        v = np.mean(vecs, axis=0)
        docvecs.append(v)
        keep_idx.append(i)

    if not docvecs:
        docarr = np.empty((0, model.vector_size), dtype=np.float32)
        keep = np.empty((0,), dtype=np.int32)
    else:
        docarr = np.vstack(docvecs).astype(np.float32)
        keep = np.array(keep_idx, dtype=np.int32)

    # L2-normalize (cosine = dot)
    norms = np.linalg.norm(docarr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    docarr = docarr / norms

    model.save(MODEL_PATH)
    np.save(DOCVEC_NPY, docarr)
    np.save(IDX_NPY, keep)

    return model, docarr, keep, df


def _ensure_ready() -> None:
    """Lazy-load or train the model + vectors once."""
    global _W2V, _DOCVECS, _INDEX, _DF
    if _W2V is not None and _DOCVECS is not None and _INDEX is not None and _DF is not None:
        return

    try:
        _DF = _load_df()
        if os.path.exists(MODEL_PATH) and os.path.exists(DOCVEC_NPY) and os.path.exists(IDX_NPY):
            _W2V = Word2Vec.load(MODEL_PATH)
            _DOCVECS = np.load(DOCVEC_NPY)
            _INDEX = np.load(IDX_NPY)
        else:
            _W2V, _DOCVECS, _INDEX, _DF = _train_and_cache()
    except Exception:
        _W2V, _DOCVECS, _INDEX, _DF = _train_and_cache()


def get_title_samples(k: int = 100) -> list[str]:
    """Return up to k sample course titles for the autocomplete."""
    try:
        df = pd.read_csv(DATA_CSV, usecols=["course_title"])
        titles = df["course_title"].dropna().astype(str).unique().tolist()
        return titles[:k]
    except Exception:
        return []


def highlight_tokens(title: str, tokens: list[str]) -> Markup:
    """Wrap query tokens in <mark> for display (safe for Jinja)."""
    if not title or not tokens:
        return Markup(escape(title))
    text = escape(title)
    for t in sorted(set(tokens), key=len, reverse=True):
        if not t:
            continue
        pattern = re.compile(rf"({re.escape(t)})", re.IGNORECASE)
        text = Markup(pattern.sub(r"<mark>\1</mark>", str(text)))
    return Markup(text)


def get_similar_courses(input_course_title: str, top_n: int = 5) -> List[Dict]:
    """
    Return top-N similar courses for a free-text title.
    Each row: {course_id, course_title, subject, similarity}
    """
    _ensure_ready()
    assert _W2V is not None and _DOCVECS is not None and _INDEX is not None and _DF is not None

    title = (input_course_title or "").strip()
    if not title:
        return []

    toks = _tokenize(title)
    toks = [w for w in toks if w in _W2V.wv]
    if not toks:
        return []  # OOV or empty

    v = np.mean([_W2V.wv[w] for w in toks], axis=0).astype(np.float32)
    nv = np.linalg.norm(v)
    if nv == 0:
        return []
    v = v / nv

    sims = _DOCVECS @ v  # cosine similarity via dot
    if sims.size == 0:
        return []

    top_idx = np.argsort(-sims)[: max(top_n * 2, 10)]
    rows: List[Dict] = []
    seen_titles = set()
    for j in top_idx:
        i = int(_INDEX[j])
        row = _DF.iloc[i]
        title_j = row["course_title"]
        if title_j.lower() == title.lower():
            continue
        if title_j in seen_titles:
            continue
        seen_titles.add(title_j)
        rows.append({
            "course_id": int(row["course_id"]),
            "course_title": title_j,
            "subject": row["subject"] or "Unknown",
            "similarity": float(sims[j]),
        })
        if len(rows) >= top_n:
            break

    return rows
