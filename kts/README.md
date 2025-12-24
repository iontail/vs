# Kernel Temporal Segmentation (KTS)

## Overview

Kernel Temporal Segmentation (KTS) detects change points in videos to automatically segment them into temporal shots using kernel-based dynamic programming.

---

## Source Attribution

**Original Code**: 
- http://lear.inrialpes.fr/software
- http://pascal.inrialpes.fr/data2/potapov/med_summaries/kts_ver1.1.tar.gz

**Modified Version**: [DS-Net Repository](https://github.com/li-plus/DSNet.git)
- Modification: Removed weave dependency

**Related Paper**: Potapov et al., "Category-specific video summarization" (ECCV 2014)

**License**: Follow original LEAR/INRIA license

---

## Purpose in Video Summarization

KTS segments videos into shots by detecting change points where visual content changes. These segments are then scored and selected for the final summary.

**Typical Pipeline**:
```
Video → Features → KTS → Segments → Importance Scoring → Knapsack Selection → Summary
```

---

## Files

### `cpd_nonlin.py`

Core dynamic programming algorithm for change point detection with **fixed** number of segments.

#### `calc_scatters(K)`
Computes scatter matrix measuring variance within segments.
```python
scatters[i,j] = variance of segment [i, j]
```

#### `cpd_nonlin(K, ncp, lmin=1, lmax=100000, backtrack=True)`

**Parameters**:
- `K`: Kernel matrix (n×n) measuring frame similarity
- `ncp`: Number of change points to detect
- `lmin/lmax`: Minimum/maximum segment length
- `backtrack`: Return change points (True) or only scores (False)

**Returns**: `(cps, scores)` - change point indices and objective values

**Algorithm**: Finds optimal segmentation minimizing total within-segment variance using DP:
```
I[k, l] = min cost for k change points in first l frames
I[k, l] = min over t { scatter[t, l-1] + I[k-1, t] }
```

---

### `cpd_auto.py`

Automatically selects optimal number of change points.

#### `cpd_auto(K, ncp, vmax, desc_rate=1)`

**Parameters**:
- `K`: Kernel matrix (n×n)
- `ncp`: Maximum number of change points to consider
- `vmax`: Penalty weight (higher → fewer segments)
- `desc_rate`: Descriptor sampling rate

**Algorithm**:
1. Run `cpd_nonlin` for 0 to m change points
2. Apply penalty: `penalty[m] = (vmax * m / 2N) * (log(N/m) + 1)`
3. Select m minimizing: `cost[m] = score[m]/N + penalty[m]`

**Returns**: `(cps, scores)` - optimal change points and scores

---

### `demo.py`

Demonstration with synthetic data showing:
- 1D and multi-dimensional signals
- Automatic change point detection
- Visualization of detected vs. ground truth change points

---

## Usage

### Basic Example

```python
import numpy as np
from kts.cpd_auto import cpd_auto

# Extract features and compute kernel
features = extract_features(video)  # [n_frames, dim]
K = np.dot(features, features.T)

# Detect change points
change_points, scores = cpd_auto(K, ncp=20, vmax=1)

# Create segments
cps = [0] + list(change_points) + [len(features)]
segments = [(cps[i], cps[i+1]) for i in range(len(cps)-1)]
```

### Integration with Video Summarization

```python
# Segment video
K = np.dot(features, features.T)
cps, _ = cpd_auto(K, ncp=30, vmax=1.0)

# Convert to shots
shots = []
cps_list = [0] + list(cps) + [n_frames]
for i in range(len(cps_list) - 1):
    shots.append({
        'start': cps_list[i],
        'end': cps_list[i+1],
        'duration': cps_list[i+1] - cps_list[i]
    })

# Score and select using Knapsack
shot_scores = [importance_model(shot) for shot in shots]
shot_durations = [s['duration'] for s in shots]
selected = knapSack(budget, shot_durations, shot_scores, len(shots))
```

---

## Parameter Guidelines

| Parameter | Typical Range | Effect |
|-----------|--------------|--------|
| `ncp` (cpd_nonlin) | 15-30 | Number of segments |
| `ncp` (cpd_auto) | 50-100 | Search space |
| `vmax` | 0.5-2.0 | Small=more segments, Large=fewer segments |
| `lmin` | 10-30 | Minimum segment length |

---

## Complexity

- **Time**: O(m × n²) where m = change points, n = frames
- **Space**: O(m × n)
- **Practical**: ~1 second for 1000 frames, 20 segments

---

## Key Advantages

- **Unsupervised**: No labeled change points needed
- **Automatic**: `cpd_auto` determines optimal segment count
- **Flexible**: Works with any feature representation
- **Robust**: Handles variable-length segments

---

## References

- **Original**: INRIA LEAR (http://lear.inrialpes.fr/software)
- **Paper**: Potapov et al., ECCV 2014
- **Modified**: DS-Net (https://github.com/li-plus/DSNet)