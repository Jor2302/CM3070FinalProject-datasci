# webpage/chat_assistant.py
import os, re, pandas as pd
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
        ]
        self.enabled = True
        self.df = self._load_courses()

        if self.df.empty:
            self.enabled = False
            self._why_disabled = "No course catalogue CSV found. Expected: course_id, title, subject."
            self.vectorizer = None
            self.tfidf = None
            return

        # ensure required columns and clean
        self.df["title"] = self.df["title"].fillna("").astype(str)
        if "subject" not in self.df.columns:
            self.df["subject"] = ""
        self.df["subject"] = self.df["subject"].fillna("").astype(str)
        self.df["__text__"] = (self.df["title"] + " " + self.df["subject"]).str.strip()

        # drop rows with no text
        self.df = self.df[self.df["__text__"].str.len() > 0].reset_index(drop=True)
        if self.df.empty:
            self.enabled = False
            self._why_disabled = "Course catalogue has no non-empty titles/subjects."
            self.vectorizer = None
            self.tfidf = None
            return

        # Keep it simple and robust: no stopwords (avoids empty vocab with short titles)
        self.vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words=None)
        self.tfidf = self.vectorizer.fit_transform(self.df["__text__"].tolist())

    def _load_courses(self) -> pd.DataFrame:
        for p in self.data_paths:
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    if "course_id" not in df.columns:
                        for alt in ["id", "courseID", "courseId"]:
                            if alt in df.columns:
                                df = df.rename(columns={alt: "course_id"})
                                break
                    if "title" not in df.columns:
                        continue
                    if "subject" not in df.columns:
                        df["subject"] = ""
                    return df[["course_id", "title", "subject"]].dropna(subset=["title"])
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

        # keyword search
        q = self.vectorizer.transform([msg])
        sims = linear_kernel(q, self.tfidf).ravel()
        if sims.max(initial=0) <= 0:
            return {"reply": "I didn’t find a close match. Try different keywords or add 'beginner/advanced'."}

        k = max(1, min(int(top_k or 3), 10))  # cap to 10
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
