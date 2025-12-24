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

- **Concordant pair (C)**: Both rankings agree on order
  ```
  (rᵢ − rⱼ)(ŝᵢ − ŝⱼ) > 0
  ```
  
- **Discordant pair (D)**: Rankings disagree on order
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
2. **Ground truth**: Average scores across users
3. **Evaluation protocol**:
   ```python
   # Convert scores to ranks
   human_ranks = rankdata(human_scores)
   pred_ranks = rankdata(predicted_scores)
   
   # Compute Kendall's Tau
   tau, p_value = kendalltau(human_ranks, pred_ranks)
   ```

4. **Typical usage**: Reported alongside F-score metrics

### SumMe Dataset

1. **Human annotations**: 15-18 users create binary summaries
2. **Importance scores**: Derived from selection frequency
3. **Evaluation protocol**:
   ```python
   # Aggregate multiple user annotations
   importance_scores = user_selections.mean(axis=0)
   
   # Rank correlation with predicted scores
   tau = kendalltau(importance_scores, predicted_scores)[0]
   ```

4. **Cross-validation**: Often computed per-fold and averaged

---

## Advantages for Video Summarization

1. **Scale-invariant**: Robust to different scoring ranges
2. **Outlier-resistant**: Focuses on ordering, not magnitude
3. **Interpretable**: Direct measure of ranking agreement
4. **Annotation-friendly**: Works well with noisy human labels

---

## Implementation Example

```python
import numpy as np
from scipy.stats import kendalltau

def evaluate_kendall_tau(human_scores, predicted_scores):
    """
    Compute Kendall's Tau for video summarization
    
    Args:
        human_scores: Array of human importance scores [N]
        predicted_scores: Array of predicted scores [N]
    
    Returns:
        tau: Kendall's Tau coefficient
    """
    tau, p_value = kendalltau(human_scores, predicted_scores)
    return tau

# Example usage
human_scores = np.array([3.2, 4.5, 2.1, 4.8, 3.7])
predicted_scores = np.array([0.45, 0.78, 0.23, 0.89, 0.56])

tau = evaluate_kendall_tau(human_scores, predicted_scores)
print(f"Kendall's Tau: {tau:.3f}")
```

---

## Key Takeaway

Kendall's Tau answers: **"Did the model correctly rank which segments are more important?"**

It evaluates global ordering consistency through pairwise comparisons, making it ideal for importance-based video summarization evaluation.

---

## References

- Kendall, M. G. (1938). "A new measure of rank correlation"
- Video summarization benchmarks: TVSum (CVPR 2015), SumMe (ECCV 2014)
- Standard evaluation protocols in video summarization literature