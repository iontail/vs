# Kendall's Tau (τ) for Video Summarization

## Overview

Kendall's Tau is a rank correlation metric that measures **pairwise order consistency** between predicted importance scores and human annotations in video summarization tasks.

## Purpose

In video summarization, we need to evaluate whether a model's predicted importance scores align with human-annotated importance rankings. Kendall's Tau focuses on **relative ordering of pairs** rather than absolute score values.

---

## Problem Formulation

Given a video with N segments/shots/frames:

- Human annotation scores: **r** = {r₁, r₂, ..., rₙ}
- Model predicted scores: **ŝ** = {ŝ₁, ŝ₂, ..., ŝₙ}

Both are converted to rankings for comparison.

---

## Definition

### Concordant and Discordant Pairs

For all pairs (i, j) where i < j:

- **Concordant pair (C)**: The number of pairs where both rankings agree on order
  ```
  (rᵢ − rⱼ)(ŝᵢ − ŝⱼ) > 0
  ```
  
- **Discordant pair (D)**: The number of pairs where Rankings disagree on order
  ```
  (rᵢ − rⱼ)(ŝᵢ − ŝⱼ) < 0
  ```

### Kendall's Tau Formula

Total number of pairs:
```
T = N(N − 1) / 2
```

Kendall's Tau:
```
τ = (C − D) / T
```

---

## Interpretation

| τ Value | Meaning |
|---------|---------|
| +1.0 | Perfect agreement in ordering |
| 0.0 | No correlation |
| -1.0 | Perfect disagreement (reverse order) |

**In video summarization context:**
> τ measures whether the model correctly identifies the **relative importance ordering** between video segments

---

## Computation in TVSum and SumMe

### TVSum Dataset

1. **Human annotations**: 20 users rate each shot (1-5 scale)
2. **Evaluation protocol**:
   ```python
   # Compute Kendall's Tau for EACH annotator
   taus = []
   for user_scores in all_user_scores:  # 20 annotators
       tau, p_value = kendalltau(user_scores, predicted_scores)
       taus.append(tau)
   
   # Average across all annotators
   final_tau = np.mean(taus)
   ```

### SumMe Dataset

1. **Human annotations**: 15-18 users create binary summaries (0/1 per frame)
2. **Evaluation protocol**:
   ```python
   # Method 1: Compute tau for EACH annotator (Standard)
   taus = []
   for user_summary in all_user_summaries:  # 15-18 annotators
       # user_summary: binary [0,1] indicating frame selection
       tau = kendalltau(user_summary, predicted_scores)[0]
       taus.append(tau)
   
   final_tau = np.mean(taus)
   
   # Method 2: Use aggregated importance scores (Alternative)
   # importance_scores = all_user_summaries.mean(axis=0)
   # tau = kendalltau(importance_scores, predicted_scores)[0]
   ```

3. **Note**: Most standard benchmarks use **Method 1** (per-annotator evaluation)
4. **Cross-validation**: Often computed per-fold and averaged

---

## Implementation Example

```python
import numpy as np
from scipy.stats import kendalltau

def evaluate_kendall_tau_tvsum(all_user_scores, predicted_scores):
    """
    Compute Kendall's Tau for TVSum dataset
    
    Args:
        all_user_scores: Array of shape [num_annotators, N]
        predicted_scores: Array of predicted scores [N]
    
    Returns:
        mean_tau: Average Kendall's Tau across all annotators
    """
    taus = []
    for user_scores in all_user_scores:
        tau, _ = kendalltau(user_scores, predicted_scores)
        taus.append(tau)
    
    return np.mean(taus)


def evaluate_kendall_tau_summe(all_user_summaries, predicted_scores):
    """
    Compute Kendall's Tau for SumMe dataset
    
    Args:
        all_user_summaries: Binary array [num_annotators, N]
        predicted_scores: Array of predicted scores [N]
    
    Returns:
        mean_tau: Average Kendall's Tau across all annotators
    """
    taus = []
    for user_summary in all_user_summaries:
        tau, _ = kendalltau(user_summary, predicted_scores)
        taus.append(tau)
    
    return np.mean(taus)


# Example usage - TVSum
all_user_scores = np.array([
    [3.2, 4.5, 2.1, 4.8, 3.7],  # User 1
    [3.0, 4.2, 2.5, 4.9, 3.5],  # User 2
    # ... 20 users total
])
predicted_scores = np.array([0.45, 0.78, 0.23, 0.89, 0.56])

mean_tau = evaluate_kendall_tau_tvsum(all_user_scores, predicted_scores)
print(f"TVSum Kendall's Tau: {mean_tau:.3f}")


# Example usage - SumMe
all_user_summaries = np.array([
    [0, 1, 0, 1, 1],  # User 1 binary summary
    [0, 1, 0, 1, 0],  # User 2 binary summary
    # ... 15-18 users total
])

mean_tau = evaluate_kendall_tau_summe(all_user_summaries, predicted_scores)
print(f"SumMe Kendall's Tau: {mean_tau:.3f}")
```

---

## Key Takeaway

Kendall's Tau answers: **"Did the model correctly rank which segments are more important?"**

It evaluates global ordering consistency through pairwise comparisons, making it ideal for importance-based video summarization evaluation.

**Important**: Standard evaluation computes tau for **each annotator individually** and then averages, rather than aggregating annotations first.

---

## References

- Kendall, M. G. (1938). "A new measure of rank correlation"
- Video summarization benchmarks: TVSum (CVPR 2015), SumMe (ECCV 2014)