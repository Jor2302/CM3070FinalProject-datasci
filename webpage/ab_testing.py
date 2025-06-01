# webpage/ab_testing.py

import numpy as np
from scipy.stats import ttest_ind
import random

def simulate_ab_test(n_users=100):
    # Simulated click/satisfaction scores (scale 0–1)
    group_a = [random.gauss(0.55, 0.1) for _ in range(n_users)]  # CF users
    group_b = [random.gauss(0.67, 0.1) for _ in range(n_users)]  # Hybrid users

    # Clip scores between 0 and 1
    group_a = np.clip(group_a, 0, 1)
    group_b = np.clip(group_b, 0, 1)

    t_stat, p_value = ttest_ind(group_b, group_a)

    return {
        "group_a_avg": round(np.mean(group_a), 3),
        "group_b_avg": round(np.mean(group_b), 3),
        "t_stat": round(t_stat, 3),
        "p_value": round(p_value, 5),
        "conclusion": "Significant" if p_value < 0.05 else "Not Significant"
    }
