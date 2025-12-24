# Knapsack Algorithm for Video Summarization

## Overview

The 0/1 Knapsack algorithm selects an optimal subset of video shots that maximizes importance scores while respecting length constraints.

**Source**: [CSTA Repository](https://github.com/thswodnjs3/CSTA/blob/master/knapsack_implementation.py)

---

## Problem Definition

**Given**:
- **n** video shots with importance scores `val[i]` and durations `wt[i]`
- **W**: Maximum summary length (budget constraint)

**Goal**: Select shots maximizing total importance while keeping total duration ≤ W

---

## Why Knapsack?

After predicting importance scores, we must select which shots to include in the final summary. This is the 0/1 Knapsack Problem:
- Each shot is either included (1) or excluded (0)
- Must respect length budget (e.g., 15% of original video)
- Maximize total importance of selected shots

---

## Algorithm

### Dynamic Programming Recurrence

```
K[i][w] = maximum importance using first i shots with budget w

K[i][w] = {
    0                                              if i=0 or w=0
    max(val[i-1] + K[i-1][w-wt[i-1]], K[i-1][w])  if wt[i-1] ≤ w
    K[i-1][w]                                      if wt[i-1] > w
}
```

### Implementation

```python
def knapSack(W, wt, val, n):
    """
    Args:
        W: Maximum summary length
        wt: Shot durations [n]
        val: Importance scores [n]
        n: Number of shots
    Returns:
        selected: Indices of selected shots
    """
    # Build DP table
    K = [[0 for _ in range(W + 1)] for _ in range(n + 1)]
    
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]
    
    # Backtrack to find selected shots
    selected = []
    w = W
    for i in range(n, 0, -1):
        if K[i][w] != K[i - 1][w]:
            selected.insert(0, i - 1)
            w -= wt[i - 1]
    
    return selected
```

**Complexity**: O(n × W) time and space

---

## Usage Example

```python
# Shot information
shot_durations = [30, 45, 25, 60, 35, 50]  # frames
importance_scores = [0.8, 0.6, 0.9, 0.5, 0.7, 0.4]  # predicted scores

# Define 15% length budget
max_length = int(0.15 * sum(shot_durations))

# Select shots
selected = knapSack(max_length, shot_durations, importance_scores, len(shot_durations))

# Create binary summary
binary_summary = [1 if i in selected else 0 for i in range(len(shot_durations))]
```

---

## Integration with TVSum/SumMe Evaluation

```python
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import f1_score

# Generate summary using knapsack
def generate_summary(predicted_scores, shot_durations, budget_ratio=0.15):
    n = len(predicted_scores)
    max_length = int(budget_ratio * sum(shot_durations))
    selected = knapSack(max_length, shot_durations, predicted_scores, n)
    
    binary_summary = [0] * n
    for idx in selected:
        binary_summary[idx] = 1
    return binary_summary

# Evaluate
pred_summary = generate_summary(predicted_scores, durations)
f1 = f1_score(ground_truth_summary, pred_summary)
rho = spearmanr(ground_truth_scores, predicted_scores)[0]
tau = kendalltau(ground_truth_scores, predicted_scores)[0]
```

---

## Advantages

- **Optimal**: Guarantees maximum importance within budget
- **Standard**: Widely used in video summarization research
- **Flexible**: Works with any importance scoring function
- **Fast**: Practical for typical videos (50-200 shots)

---

## Key Takeaway

Knapsack provides the optimal solution for shot selection, bridging importance prediction and final summary generation while strictly respecting length constraints.

---

## References

- [CSTA Implementation](https://github.com/thswodnjs3/CSTA/blob/master/knapsack_implementation.py)
- TVSum (CVPR 2015), SumMe (ECCV 2014)