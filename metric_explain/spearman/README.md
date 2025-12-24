# Spearman's Rank Correlation (ρ) for Video Summarization

## Overview

Spearman's rank correlation coefficient (ρ) measures the **monotonic relationship** between two ranked variables.  
In video summarization, it is used to evaluate whether a model preserves the **same relative importance ordering of video segments** as human annotations, regardless of the absolute score scale.

Importantly, Spearman's ρ does **not** compare score distributions or summary quality directly; it only evaluates agreement in **ranking trends**.

---

## Purpose

While **Kendall's Tau (τ)** focuses on pairwise ordering consistency, **Spearman's ρ** assesses **global monotonic agreement** between two rankings.

In the context of video summarization, ρ answers the question:
> "Does the model rank video segments in a way that is monotonically consistent with human judgments of importance?"

---

## Problem Formulation

Given a video divided into N segments:

- **Human annotation scores → ranks:** **r** = {r₁, r₂, ..., rₙ}
- **Model-predicted scores → ranks:** **s** = {s₁, s₂, ..., sₙ}

Ranks are obtained by sorting scores; ties are allowed and handled by assigning average ranks.

---

## Definition

### Spearman's Rank Correlation Coefficient

The **general and standard definition** of Spearman's ρ is:

```
ρ = cov(r, s) / (σᵣ · σₛ)
```

where:
- cov(·,·) is the covariance between rank vectors
- σᵣ, σₛ are the standard deviations of the ranks

This definition is equivalent to **computing Pearson correlation on ranks** and is valid **with or without ties**.

> **Note:** Closed-form formulas based on rank differences (e.g., involving Σdᵢ²) apply only when there are no ties. In practice, video summarization data almost always contains ties, so the covariance-based definition is used.

---

## Interpretation

| ρ value | Interpretation |
|:-------:|----------------|
| +1.0 | Perfect monotonic agreement |
| 0.0 | No monotonic relationship |
| -1.0 | Perfect inverse monotonic relationship |

**In video summarization:**  
A higher ρ indicates that segments judged as more important by humans are also ranked as more important by the model, even if absolute scores differ.

---

## Usage in Popular Datasets

### TVSum Dataset

**Annotation structure:**
- 20 annotators per video
- Segment-level importance scores (Likert scale: 1–5)

**Common evaluation practice:**
1. Compute Spearman's ρ **separately for each annotator**
2. Average ρ across annotators

```python
from scipy.stats import spearmanr
import numpy as np

rho_scores = []
for user_scores in all_user_scores:  # shape: [20, N]
    rho, _ = spearmanr(user_scores, predicted_scores)
    rho_scores.append(rho)

mean_rho = np.mean(rho_scores)
std_rho = np.std(rho_scores)
```

This protocol evaluates how consistently the model's ranking aligns with each individual human judgment.

---

### SumMe Dataset (Non-standard Usage)

**Important note:**  
Spearman's ρ is **not** part of the official SumMe benchmark.

- **Standard SumMe evaluation:** F-score between predicted summaries and binary human summaries
- **Spearman's ρ** may be reported only in alternative settings where the task is framed as importance score regression

**If used (non-standard):**
- Compute ρ per annotator using binary vectors (0/1) vs predicted scores
- Average across annotators

```python
from scipy.stats import spearmanr
import numpy as np

rho_scores = []
for user_summary in all_user_summaries:  # shape: [15–18, N]
    rho, _ = spearmanr(user_summary, predicted_scores)
    rho_scores.append(rho)

mean_rho = np.mean(rho_scores)
```

*Results obtained this way should not be directly compared to official SumMe leaderboard results.*

---

## Comparison with Kendall's Tau

| Aspect | Spearman (ρ) | Kendall (τ) |
|--------|--------------|-------------|
| **Comparison unit** | Rank vectors | Rank pairs |
| **Sensitivity** | Large rank deviations | Local order inversions |
| **Interpretation** | Global monotonic agreement | Local ordering consistency |
| **Ties** | Naturally supported | Requires tie-aware variants |
| **Typical magnitude** | Often higher | More conservative |

---

## Advantages in Video Summarization Research

1. **Order-aware:** Captures global monotonic relationships
2. **Scale-invariant:** Independent of absolute score magnitudes
3. **Tie-robust:** Suitable for averaged or discrete annotations
4. **Efficient:** Computable in O(N log N) time

---

## Implementation Example

```python
import numpy as np
from scipy.stats import spearmanr

def evaluate_spearman_tvsum(all_user_scores, predicted_scores):
    """
    Compute Spearman correlation for TVSum dataset
    
    Args:
        all_user_scores: Array of shape [num_annotators, N]
        predicted_scores: Array of predicted scores [N]
    
    Returns:
        mean_rho: Average Spearman coefficient across all annotators
        std_rho: Standard deviation of Spearman coefficients
    """
    rho_scores = []
    for user_scores in all_user_scores:
        rho, _ = spearmanr(user_scores, predicted_scores)
        rho_scores.append(rho)
    
    return np.mean(rho_scores), np.std(rho_scores)


def evaluate_spearman_summe(all_user_summaries, predicted_scores):
    """
    Compute Spearman correlation for SumMe dataset (non-standard)
    
    Args:
        all_user_summaries: Binary array [num_annotators, N]
        predicted_scores: Array of predicted scores [N]
    
    Returns:
        mean_rho: Average Spearman coefficient across all annotators
        std_rho: Standard deviation of Spearman coefficients
    """
    rho_scores = []
    for user_summary in all_user_summaries:
        rho, _ = spearmanr(user_summary, predicted_scores)
        rho_scores.append(rho)
    
    return np.mean(rho_scores), np.std(rho_scores)


# Example: TVSum-style evaluation
all_user_scores = np.array([
    [3.2, 4.5, 2.1, 4.8, 3.7],  # User 1
    [3.0, 4.2, 2.5, 4.9, 3.5],  # User 2
    [3.5, 4.3, 2.0, 4.7, 3.8],  # User 3
    # ... 20 users total
])
predicted_scores = np.array([0.45, 0.78, 0.23, 0.89, 0.56])

mean_rho, std_rho = evaluate_spearman_tvsum(all_user_scores, predicted_scores)
print(f"TVSum Spearman's ρ: {mean_rho:.3f} ± {std_rho:.3f}")


# Example: SumMe-style evaluation (non-standard)
all_user_summaries = np.array([
    [0, 1, 0, 1, 1],  # User 1 binary summary
    [0, 1, 0, 1, 0],  # User 2 binary summary
    [1, 1, 0, 1, 1],  # User 3 binary summary
    # ... 15-18 users total
])

mean_rho, std_rho = evaluate_spearman_summe(all_user_summaries, predicted_scores)
print(f"SumMe Spearman's ρ: {mean_rho:.3f} ± {std_rho:.3f}")
```

---

## Practical Considerations

### Handling Ties

- Use implementations that handle ties correctly (e.g., `scipy.stats.spearmanr`)
- Tied scores are assigned average ranks automatically

### Statistical Significance

```python
rho, p_value = spearmanr(user_scores, predicted_scores)
if p_value < 0.05:
    print(f"Statistically significant correlation: ρ={rho:.3f}")
```

### Reporting Results

- Always specify: Per-annotator vs aggregated evaluation
- Specify the dataset and split protocol
- Avoid comparing ρ values across different evaluation setups

---

## Key Takeaway

Spearman's ρ evaluates global monotonic agreement in importance rankings. In video summarization, it answers: 

> **"Does the model preserve the same relative importance ordering of video segments as human annotators, independent of absolute score scale?"**

It is standard for importance ranking evaluation (e.g., TVSum), but not a replacement for F-score–based summary evaluation (e.g., SumMe).

---

## References

- Spearman, C. (1904). *The proof and measurement of association between two things.*
- Song et al. (2015). *TVSum: Summarizing Web Videos using Title-based Image Search.* CVPR.
- Gygli et al. (2014). *Creating Summaries from User Videos.* ECCV.