# -----------------------------------------------------------------------------
# Minimal keyword→course helper for the floating chat widget.
# - Loads a course catalogue CSV (several possible paths)
# - Builds a TF-IDF index over title + subject
# - Given a free-text query, returns the top-k matching courses
# - Handles greetings and a tiny "user 123" deep-link intent
# -----------------------------------------------------------------------------

import os
import re
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ChatAssistant:
    """Free-text → top-k matching courses by TF-IDF on title+subject."""

    def __init__(self, data_paths: Optional[List[str]] = None):
        self.data_paths = data_paths or [
            os.path.join("webpage", "data", "udemy_courses.csv"),
            os.path.join("webpage", "data", "courses.csv"),
            os.path.join("data", "udemy_courses.csv"),
            os.path.join("data", "courses.csv"),
            os.path.join("data", "udemy_course_data.csv"),  # accepts course_title -> title
        ]
        self.enabled = True
        self.df = self._load_courses()

        if self.df.empty:
            self.enabled = False
            self._why_disabled = "No course catalogue CSV found. Expected: course_id, title/course_title, subject."
            self.vectorizer = None
            self.tfidf = None
            return

        # clean text fields
        self.df["title"] = self.df["title"].fillna("").astype(str)
        if "subject" not in self.df.columns:
            self.df["subject"] = ""
        self.df["subject"] = self.df["subject"].fillna("").astype(str)
        self.df["__text__"] = (self.df["title"] + " " + self.df["subject"]).str.strip()

        # drop empty rows
        self.df = self.df[self.df["__text__"].str.len() > 0].reset_index(drop=True)
        if self.df.empty:
            self.enabled = False
            self._why_disabled = "Course catalogue has no non-empty titles/subjects."
            self.vectorizer = None
            self.tfidf = None
            return

        # robust vectorizer (no stopwords so short titles survive)
        self.vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words=None)
        self.tfidf = self.vectorizer.fit_transform(self.df["__text__"].tolist())

    def _load_courses(self) -> pd.DataFrame:
        """
        Try several known paths for a catalogue CSV and normalize schema:
        - Accepts `course_id` or common aliases (id, courseID, courseId)
        - Accepts `title` or `course_title` (mapped to title)
        - Ensures `subject` exists (empty string if missing)
        Returns only: course_id, title, subject
        """
        for p in self.data_paths:
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p)

                    # normalize id -> course_id
                    if "course_id" not in df.columns:
                        for alt in ["id", "courseID", "courseId"]:
                            if alt in df.columns:
                                df = df.rename(columns={alt: "course_id"})
                                break

                    # normalize title (accept course_title)
                    if "title" not in df.columns:
                        if "course_title" in df.columns:
                            df = df.rename(columns={"course_title": "title"})
                        else:
                            continue  # no usable title field

                    if "subject" not in df.columns:
                        df["subject"] = ""

                    # minimal view
                    keep = df[["course_id", "title", "subject"]].dropna(subset=["title"])
                    # enforce str IDs (avoid 1.0 etc.)
                    keep["course_id"] = keep["course_id"].astype(str)
                    return keep
                except Exception:
                    continue
        return pd.DataFrame()

    def help_text(self) -> str:
        return (
            "Tell me what you want to learn.\n"
            "Try: python for data analysis · statistics basics · machine learning for beginners\n"
            "Tip: type 'user 123' to open recommendations for that user."
        )

    def _is_greeting(self, msg: str) -> bool:
        m = msg.lower().strip()
        greetings = ("hi", "hello", "hey", "help", "?")
        return any(m == g or m.startswith(g + " ") or m.endswith(" " + g) for g in greetings)

    def reply(self, message: str, top_k: int = 3) -> Dict[str, Any]:
        msg = (message or "").strip()
        if not msg:
            return {"reply": "Tell me what you want to learn and I will suggest courses."}

        if not self.enabled or self.tfidf is None or self.tfidf.shape[0] == 0:
            why = getattr(self, "_why_disabled", "Catalogue not loaded.")
            return {"reply": f"I am not ready yet: {why}"}

        if self._is_greeting(msg):
            return {"reply": self.help_text()}

        m = re.match(r"user\s+(\w+)", msg.lower())
        if m:
            user_id = m.group(1)
            return {"reply": f"Opening recommendations for user {user_id}.", "link": f"/recommend?user_id={user_id}"}

        q = self.vectorizer.transform([msg])
        sims = linear_kernel(q, self.tfidf).ravel()

        if sims.max(initial=0) <= 0:
            return {"reply": "I didn’t find a close match. Try different keywords or add 'beginner/advanced'."}

        k = max(1, min(int(top_k or 3), 10))
        top_idx = sims.argsort()[::-1][:k]

        items = []
        for i in top_idx:
            row = self.df.iloc[i]
            items.append({
                "course_id": str(row["course_id"]),
                "title": str(row["title"]),
                "subject": str(row.get("subject", "")),
                "link": f"/recommend?course_id={row['course_id']}"
            })

        reply = f"Top {len(items)} matches. Closest: “{items[0]['title']}”."
        return {"reply": reply, "items": items}
