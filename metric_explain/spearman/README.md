# Spearman's Rank Correlation (ρ) for Video Summarization

## Overview

Spearman's rank correlation coefficient measures the **monotonic relationship** between two ranked variables. In video summarization, it evaluates how well predicted importance scores follow the overall distribution pattern of human annotations.

## Purpose

While Kendall's Tau focuses on pairwise ordering, Spearman's ρ assesses the **global similarity of rank distributions**, capturing whether the model's predictions maintain the same relative trend as human judgments.

---

## Problem Formulation

Given a video with N segments:

- Human annotation scores → ranks: **r** = {r₁, r₂, ..., rₙ}
- Model predicted scores → ranks: **s** = {s₁, s₂, ..., sₙ}

---

## Definition

### Rank Difference

For each segment i:
```
dᵢ = rᵢ − sᵢ
```

### Spearman's Coefficient Formula

When no ties exist:
```
ρ = 1 − (6 Σdᵢ²) / (N(N² − 1))
```

General formula (with ties):
```
ρ = cov(r, s) / (σᵣ · σₛ)
```

where cov is covariance and σ is standard deviation of ranks.

---

## Interpretation

| ρ Value | Meaning |
|---------|---------|
| +1.0 | Perfect monotonic positive relationship |
| 0.0 | No monotonic correlation |
| -1.0 | Perfect monotonic negative relationship |

**In video summarization context:**
> ρ measures whether the model's predictions follow the **same importance distribution pattern** as human annotations

---

## Computation in TVSum and SumMe

### TVSum Dataset

1. **Annotation structure**: 20 users × K shots (1-5 Likert scale)
2. **Ground truth generation**:
   ```python
   # Average across users
   human_scores = user_annotations.mean(axis=0)  # Shape: [K]
   ```

3. **Evaluation protocol**:
   ```python
   from scipy.stats import spearmanr
   
   # Compute Spearman correlation
   rho, p_value = spearmanr(human_scores, predicted_scores)
   ```

4. **Typical range**: ρ ∈ [0.3, 0.7] for state-of-the-art methods

### SumMe Dataset

1. **Annotation structure**: 15-18 binary summaries per video
2. **Importance derivation**:
   ```python
   # Count selection frequency
   importance = binary_summaries.sum(axis=0) / n_users
   # Values range from 0 (never selected) to 1 (always selected)
   ```

3. **Evaluation protocol**:
   ```python
   # Per-user evaluation (common approach)
   rho_scores = []
   for user_summary in user_summaries:
       user_importance = derive_importance(user_summary)
       rho = spearmanr(user_importance, predicted_scores)[0]
       rho_scores.append(rho)
   
   # Report average
   mean_rho = np.mean(rho_scores)
   ```

4. **Cross-validation**: 5-fold split, averaged results

---

## Comparison with Kendall's Tau

| Aspect | Spearman (ρ) | Kendall (τ) |
|--------|--------------|-------------|
| **Comparison unit** | Individual rank differences | Pairwise order agreements |
| **Sensitivity** | Sensitive to large rank deviations | Sensitive to order inversions |
| **Interpretation** | Global distribution similarity | Local ordering consistency |
| **Computation** | O(N log N) with sorting | O(N²) naive, O(N log N) optimized |
| **Typical values** | Often higher magnitude | More conservative |

---

## Advantages for Video Summarization

1. **Distribution-aware**: Captures overall score patterns
2. **Continuous scores**: Works naturally with real-valued predictions
3. **Computationally efficient**: Faster than Kendall for large N
4. **Standard metric**: Widely adopted in ranking evaluation

---

## Implementation Example

```python
import numpy as np
from scipy.stats import spearmanr

def evaluate_spearman(human_scores, predicted_scores):
    """
    Compute Spearman correlation for video summarization
    
    Args:
        human_scores: Array of human importance scores [N]
        predicted_scores: Array of predicted scores [N]
    
    Returns:
        rho: Spearman correlation coefficient
        p_value: Statistical significance
    """
    rho, p_value = spearmanr(human_scores, predicted_scores)
    return rho, p_value

# Example: TVSum-style evaluation
human_scores = np.array([3.2, 4.5, 2.1, 4.8, 3.7])
predicted_scores = np.array([0.45, 0.78, 0.23, 0.89, 0.56])

rho, p = evaluate_spearman(human_scores, predicted_scores)
print(f"Spearman's ρ: {rho:.3f} (p={p:.4f})")

# Example: SumMe-style with multiple users
def evaluate_summe_spearman(user_summaries, predicted_scores):
    """Evaluate with multiple user annotations"""
    rho_scores = []
    for user_summary in user_summaries:
        # Convert binary summary to importance scores
        importance = user_summary.astype(float)
        rho = spearmanr(importance, predicted_scores)[0]
        rho_scores.append(rho)
    return np.mean(rho_scores), np.std(rho_scores)
```

---

## Practical Considerations

### Handling Ties

When multiple segments have identical scores:
- Use `scipy.stats.spearmanr` which handles ties automatically
- Ties are assigned average ranks

### Statistical Significance

Always report p-values when possible:
```python
rho, p_value = spearmanr(human_scores, predicted_scores)
if p_value < 0.05:
    print(f"Correlation is statistically significant: ρ={rho:.3f}")
```

### Dataset-Specific Notes

**TVSum**: Higher agreement between users → higher ρ values expected

**SumMe**: More diverse annotations → moderate ρ values are normal

---

## Key Takeaway

Spearman's ρ answers: **"Does the model's importance distribution follow the same trend as human annotations?"**

It evaluates global ranking similarity, making it ideal for assessing whether the model captures the overall importance pattern across the video.

---

## References

- Spearman, C. (1904). "The proof and measurement of association between two things"
- TVSum: Song et al. (CVPR 2015)
- SumMe: Gygli et al. (ECCV 2014)
- Standard evaluation protocols in video summarization benchmarks