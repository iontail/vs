# Spearman’s Rank Correlation (ρ) for Video Summarization

## Overview

Spearman’s rank correlation coefficient (ρ) measures the **monotonic relationship**
between two ranked variables.  
In video summarization, it is used to evaluate whether a model preserves the **same
relative importance ordering of video segments** as human annotations, regardless
of absolute score scale.

Importantly, Spearman’s ρ does **not** compare score distributions or summary quality
directly; it only evaluates agreement in **ranking trends**.

---

## Purpose

While **Kendall’s Tau (τ)** focuses on pairwise ordering consistency,
**Spearman’s ρ** assesses **global monotonic agreement** between two rankings.

In the context of video summarization, ρ answers the question:

> *Does the model rank video segments in a way that is monotonically consistent
> with human judgments of importance?*

---

## Problem Formulation

Given a video divided into \(N\) segments:

- Human annotation scores → ranks:  
  \[
  \mathbf{r} = \{r_1, r_2, \dots, r_N\}
  \]

- Model-predicted scores → ranks:  
  \[
  \mathbf{s} = \{s_1, s_2, \dots, s_N\}
  \]

Ranks are obtained by sorting scores; ties are allowed and handled by assigning
average ranks.

---

## Definition

### Spearman’s Rank Correlation Coefficient

The **general and standard definition** of Spearman’s ρ is:

\[
\rho = \frac{\operatorname{cov}(\mathbf{r}, \mathbf{s})}
             {\sigma_{\mathbf{r}} \, \sigma_{\mathbf{s}}}
\]

where:
- \(\operatorname{cov}(\cdot,\cdot)\) is the covariance between rank vectors,
- \(\sigma_{\mathbf{r}}, \sigma_{\mathbf{s}}\) are the standard deviations of the ranks.

This definition is equivalent to **computing Pearson correlation on ranks** and
is valid **with or without ties**.

> Note: Closed-form formulas based on rank differences (e.g., involving \(\sum d_i^2\))
> apply only when there are no ties. In practice, video summarization data almost
> always contains ties, so the covariance-based definition is used.

---

## Interpretation

| ρ value | Interpretation |
|--------:|----------------|
| +1.0 | Perfect monotonic agreement |
|  0.0 | No monotonic relationship |
| −1.0 | Perfect inverse monotonic relationship |

**In video summarization**:  
A higher ρ indicates that segments judged as more important by humans
are also ranked as more important by the model, even if absolute scores differ.

---

## Usage in Popular Datasets

### TVSum Dataset

**Annotation structure**
- 20 annotators per video
- Segment-level importance scores (Likert scale: 1–5)

**Common evaluation practice**
1. Compute Spearman’s ρ **separately for each annotator**
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

This protocol evaluates how consistently the model’s ranking aligns
with each individual human judgment.

⸻

SumMe Dataset (Non-standard Usage)

Important note
Spearman’s ρ is not part of the official SumMe benchmark.
	•	Standard SumMe evaluation:
F-score between predicted summaries and binary human summaries.
	•	Spearman’s ρ may be reported only in alternative settings where the task
is framed as importance score regression, not summary selection.

If used (non-standard):
	•	Compute ρ per annotator using binary vectors (0/1) vs predicted scores
	•	Average across annotators

from scipy.stats import spearmanr
import numpy as np

rho_scores = []
for user_summary in all_user_summaries:  # shape: [15–18, N]
    rho, _ = spearmanr(user_summary, predicted_scores)
    rho_scores.append(rho)

mean_rho = np.mean(rho_scores)

Results obtained this way should not be directly compared
to official SumMe leaderboard results.

⸻

Comparison with Kendall’s Tau

Aspect	Spearman (ρ)	Kendall (τ)
Comparison unit	Rank vectors	Rank pairs
Sensitivity	Large rank deviations	Local order inversions
Interpretation	Global monotonic agreement	Local ordering consistency
Ties	Naturally supported	Requires tie-aware variants
Typical magnitude	Often higher	More conservative


⸻

Advantages in Video Summarization Research
	1.	Order-aware: Captures global monotonic relationships
	2.	Scale-invariant: Independent of absolute score magnitudes
	3.	Tie-robust: Suitable for averaged or discrete annotations
	4.	Efficient: Computable in (O(N \log N)) time

⸻

Practical Considerations

Handling Ties
	•	Use implementations that handle ties correctly (e.g., scipy.stats.spearmanr)
	•	Tied scores are assigned average ranks automatically

Statistical Significance

rho, p_value = spearmanr(user_scores, predicted_scores)
if p_value < 0.05:
    print(f"Statistically significant correlation: ρ={rho:.3f}")

Reporting Results
	•	Always specify:
	•	Per-annotator vs aggregated evaluation
	•	Dataset and split protocol
	•	Avoid comparing ρ values across different evaluation setups

⸻

Key Takeaway

Spearman’s ρ evaluates global monotonic agreement in importance rankings.

In video summarization, it answers:

Does the model preserve the same relative importance ordering of video segments
as human annotators, independent of absolute score scale?

It is standard for importance ranking evaluation (e.g., TVSum),
but not a replacement for F-score–based summary evaluation (e.g., SumMe).

⸻

References
	•	Spearman, C. (1904). The proof and measurement of association between two things
	•	Song et al., TVSum: Summarizing Web Videos, CVPR 2015
	•	Gygli et al., Creating Summaries from User Videos, ECCV 2014

