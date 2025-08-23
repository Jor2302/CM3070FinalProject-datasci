# ab_testing.py
from __future__ import annotations

import os
import hashlib
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# ---------------------------
# Helpers
# ---------------------------

def assign_group(user_id: str) -> str:
    """Deterministic assignment: same user_id -> same group (A or B)."""
    h = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return "A" if (h % 2 == 0) else "B"

def _pick_rating_column(df: pd.DataFrame) -> pd.Series:
    """
    Try common column names and coerce to numeric.
    Priority: 'rating' > 'comment' > 'feedback' > 'score'
    """
    for col in ["rating", "comment", "feedback", "score"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                return s
    return pd.Series([np.nan] * len(df), index=df.index)

def _cohens_d(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Cohen's d for independent samples using pooled SD."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return None
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled_var = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / max(1, (n1 + n2 - 2))
    if pooled_var <= 0:
        return None
    return float((np.mean(b) - np.mean(a)) / np.sqrt(pooled_var))

def _effect_label(d: Optional[float]) -> Optional[str]:
    if d is None:
        return None
    ad = abs(d)
    if ad < 0.2:   return "negligible"
    if ad < 0.5:   return "small"
    if ad < 0.8:   return "medium"
    return "large"

def _bootstrap_ci_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[Optional[float], Optional[float]]:
    """Bootstrap CI for mean(b) - mean(a)."""
    if len(a) < 2 or len(b) < 2:
        return (None, None)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        diffs[i] = np.mean(b_s) - np.mean(a_s)
    low = float(np.percentile(diffs, 100 * (alpha / 2)))
    high = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return (low, high)

# ---------------------------
# Public API
# ---------------------------

def run_ab_test_real_feedback(
    csv_path: str = os.path.join("data", "user_feedback.csv"),
    *,
    rating_col: str | None = None,
    alpha: float = 0.05,
    bootstrap_iters: int = 10000
) -> Dict[str, object]:
    """
    Run an A/B test on real user feedback from CSV.

    - If `variant` column exists, it is respected (A/B); otherwise we deterministically assign by user_id.
    - Rating column is auto-detected unless `rating_col` is provided.
    - Reports Welch's t-test p-value, Cohen's d, and bootstrap 95% CI for mean difference.
    """
    try:
        if not os.path.exists(csv_path):
            return {"error": f"File not found: {csv_path}"}

        df = pd.read_csv(csv_path)

        if "user_id" not in df.columns:
            return {"error": "user_id column is missing in user_feedback.csv"}

        # Choose rating column
        if rating_col and rating_col in df.columns:
            ratings = pd.to_numeric(df[rating_col], errors="coerce")
        else:
            ratings = _pick_rating_column(df)

        df = df.assign(rating=ratings)
        df = df.dropna(subset=["rating", "user_id"]).copy()
        df["rating"] = df["rating"].astype(float)
        df["user_id"] = df["user_id"].astype(str)

        # Respect provided variant if present; otherwise deterministic split
        if "variant" in df.columns:
            df["group"] = df["variant"].astype(str).str.upper().map({"A": "A", "B": "B"})
            df["group"] = df["group"].fillna(df["user_id"].apply(assign_group))
        else:
            df["group"] = df["user_id"].apply(assign_group)

        # Labels for the page (A=CF, B=Hybrid)
        group_a_label = "CF"
        group_b_label = "Hybrid"

        group_a = df.loc[df["group"] == "A", "rating"].to_numpy()
        group_b = df.loc[df["group"] == "B", "rating"].to_numpy()

        n_a, n_b = int(len(group_a)), int(len(group_b))
        mean_a = float(np.mean(group_a)) if n_a else np.nan
        mean_b = float(np.mean(group_b)) if n_b else np.nan
        mean_diff = float(mean_b - mean_a) if (n_a and n_b) else np.nan

        # basic checks
        if n_a < 2 or n_b < 2:
            return {
                "n_a": n_a, "n_b": n_b,
                "group_a_avg": round(mean_a, 3) if n_a else None,
                "group_b_avg": round(mean_b, 3) if n_b else None,
                "mean_diff": None,
                "t_stat": None, "p_value": None,
                "ci95": None,
                "cohens_d": None,
                "alpha": alpha,
                "conclusion": "Not enough data to compute statistical significance.",
                "group_a_label": group_a_label,
                "group_b_label": group_b_label,
            }

        if (np.allclose(group_a, group_a[0]) and np.allclose(group_b, group_b[0])):
            return {
                "n_a": n_a, "n_b": n_b,
                "group_a_avg": round(mean_a, 3),
                "group_b_avg": round(mean_b, 3),
                "mean_diff": round(mean_diff, 3),
                "t_stat": None, "p_value": None,
                "ci95": None,
                "cohens_d": None,
                "alpha": alpha,
                "conclusion": "No rating variation—cannot test significance.",
                "group_a_label": group_a_label,
                "group_b_label": group_b_label,
            }

        # Welch's t-test + effect size + CI
        t_stat, p_value = ttest_ind(group_b, group_a, equal_var=False)
        ci_low, ci_high = _bootstrap_ci_mean_diff(group_a, group_b, n_boot=bootstrap_iters, alpha=alpha)
        d = _cohens_d(group_a, group_b)
        effect_lbl = _effect_label(d)
        ci_includes_zero = (ci_low is not None and ci_high is not None and ci_low <= 0.0 <= ci_high)

        # small, helpful notes
        total_n = n_a + n_b
        power_note = None
        if total_n < 50:
            power_note = "Small sample size — limited power; consider collecting more ratings."
        elif min(n_a, n_b) < 20:
            power_note = "Uneven or small groups — estimates may be noisy."
        imbalance_note = None
        if min(n_a, n_b) / max(n_a, n_b) < 0.5:
            imbalance_note = "Groups are imbalanced; Welch’s test is used, but balance would help."

        return {
            "n_a": n_a,
            "n_b": n_b,
            "group_a_avg": round(mean_a, 3),
            "group_b_avg": round(mean_b, 3),
            "mean_diff": round(mean_diff, 3),
            "t_stat": round(float(t_stat), 3),
            "p_value": round(float(p_value), 5),
            "ci95": (round(ci_low, 3), round(ci_high, 3)) if ci_low is not None else None,
            "ci_includes_zero": ci_includes_zero,
            "cohens_d": round(d, 3) if d is not None else None,
            "effect_label": effect_lbl,
            "alpha": alpha,
            "conclusion": "Significant" if p_value < alpha else "Not Significant",
            "group_a_label": group_a_label,
            "group_b_label": group_b_label,
            "power_note": power_note,
            "imbalance_note": imbalance_note,
        }

    except Exception as e:
        return {"error": str(e)}

# Optional CLI
def _pretty_print(result: Dict[str, object]) -> None:
    if "error" in result:
        print(f"[ERROR] {result['error']}")
        return
    print("--- A/B Test (Real Feedback) ---")
    print(f"n_A={result['n_a']}  n_B={result['n_b']}")
    print(f"avg_A={result['group_a_avg']}  avg_B={result['group_b_avg']}")
    print(f"mean_diff (B - A) = {result['mean_diff']}")
    print(f"Welch t={result['t_stat']}, p={result['p_value']}  (alpha={result['alpha']})")
    print(f"95% CI for diff = {result['ci95']}")
    print(f"Cohen's d = {result['cohens_d']} ({result.get('effect_label')})")
    print(f"Conclusion: {result['conclusion']}")
    if result.get("power_note"): print("Note:", result["power_note"])
    if result.get("imbalance_note"): print("Note:", result["imbalance_note"])

if __name__ == "__main__":
    res = run_ab_test_real_feedback()
    _pretty_print(res)
