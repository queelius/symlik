# Thesis Reproduction: Weibull Series Systems

This tutorial demonstrates reproducing results from:

> **"Reliability Estimation in Series Systems: Maximum Likelihood Techniques
> for Right-Censored and Masked Failure Data"**
> Alex Towell, Masters Thesis

The thesis investigates MLE for component reliability from masked failure data in series systems.

## System Configuration

A 5-component series system with Weibull lifetimes:

| Component | Shape (k) | Scale (θ) | Failure Prob |
|-----------|-----------|-----------|--------------|
| 1         | 1.2576    | 994.37    | 0.17         |
| 2         | 1.1635    | 908.95    | 0.21         |
| 3         | 1.1308    | 840.11    | 0.23         |
| 4         | 1.1802    | 940.13    | 0.20         |
| 5         | 1.2034    | 923.16    | 0.20         |

Parameters from Guo, Niu, and Szidarovszky (2013), extended to 5 components.

## C1, C2, C3 Conditions

The likelihood model uses three conditions on the masking mechanism:

- **C1**: True cause is always in the candidate set
- **C2**: P(candidate set) independent of which component in set failed
- **C3**: Masking probabilities independent of parameter θ

Under these conditions, the likelihood contribution for a masked observation with candidate set $C$ is:

$$L \propto \left(\sum_{j \in C} h_j(t)\right) \cdot S_{\text{sys}}(t)$$

## Building the Model

```python
from symlik import ContributionModel
from symlik.series import build_weibull_series_contributions

# Build all contribution types for 5-component system
contributions = build_weibull_series_contributions(m=5)

# Parameters: shapes k1..k5 and scales theta1..theta5
params = [f"{p}{i}" for i in range(1, 6) for p in ("k", "theta")]

model = ContributionModel(
    params=params,
    type_column="obs_type",
    contributions=contributions,
)
```

This creates 32 contribution types:
- `known_1` through `known_5` (exact cause identified)
- `masked_12`, `masked_13`, ... `masked_12345` (all 2^5 - 6 masked subsets)
- `right_censored` (system survived past observation time)

## Bernoulli Masking Model

The thesis uses a Bernoulli masking model that satisfies C1, C2, C3:

```python
import random

def generate_candidate_set(true_cause: int, m: int, p: float) -> list:
    """
    Bernoulli masking model.

    Args:
        true_cause: Index of true failing component (0-based)
        m: Number of components
        p: Probability of including each non-failing component
    """
    candidates = [true_cause]  # C1: true cause always included
    for j in range(m):
        if j != true_cause and random.random() < p:
            candidates.append(j)
    return sorted(candidates)
```

## Generating Synthetic Data

```python
import numpy as np

# True parameters
shapes = [1.2576, 1.1635, 1.1308, 1.1802, 1.2034]
scales = [994.37, 908.95, 840.11, 940.13, 923.16]

# Simulation settings (from thesis)
n = 90          # Sample size
q = 0.825       # Right-censoring quantile
p = 0.215       # Masking probability

# Compute right-censoring time τ such that P(T < τ) = q
from scipy.optimize import brentq

def system_cdf(t, shapes, scales):
    log_survival = sum(-((t / s)**k) for k, s in zip(shapes, scales))
    return 1 - np.exp(log_survival)

tau = brentq(lambda t: system_cdf(t, shapes, scales) - q, 1e-6, 10000)
# τ ≈ 377.7 for these parameters
```

## Data Generation Loop

```python
def generate_series_data(n, shapes, scales, tau, p):
    obs_types = []
    times = []
    m = len(shapes)

    for i in range(n):
        # Generate component lifetimes
        T_components = [s * np.random.weibull(k) for k, s in zip(shapes, scales)]

        # System lifetime = minimum
        T = min(T_components)
        K = T_components.index(T)  # True cause

        if T > tau:
            # Right-censored
            obs_types.append("right_censored")
            times.append(tau)
        else:
            # Generate candidate set
            candidates = generate_candidate_set(K, m, p)

            if len(candidates) == 1:
                obs_types.append(f"known_{K+1}")
            else:
                obs_types.append("masked_" + "".join(str(c+1) for c in candidates))

            times.append(T)

    return {"obs_type": obs_types, "t": times}
```

## Fitting the Model

```python
# Generate data
data = generate_series_data(n, shapes, scales, tau, p)

# Initial values (true parameters or reasonable guesses)
init = {f"k{i}": 1.2 for i in range(1, 6)}
init.update({f"theta{i}": 900 for i in range(1, 6)})

# Parameter bounds
bounds = {
    **{f"k{i}": (0.5, 5) for i in range(1, 6)},
    **{f"theta{i}": (100, 3000) for i in range(1, 6)},
}

# Fit
mle, iterations = model.mle(data=data, init=init, bounds=bounds)
se = model.se(mle, data)

# Display results
for i in range(1, 6):
    print(f"Component {i}: k={mle[f'k{i}']:.4f} (SE={se[f'k{i}']:.4f}), "
          f"θ={mle[f'theta{i}']:.2f} (SE={se[f'theta{i}']:.2f})")
```

## Example Output

With n=90, q=0.825, p=0.215:

```
Observation type distribution:
  known_1: 5, known_2: 2, known_3: 9, known_4: 3, known_5: 6
  masked_12: 3, masked_13: 7, masked_14: 3, ...
  right_censored: 12

Component 1: k=1.11 (SE=0.18), θ=993.9 (SE=156.2)
Component 2: k=1.48 (SE=0.30), θ=909.3 (SE=189.1)
Component 3: k=0.91 (SE=0.12), θ=840.1 (SE=97.3)
...
```

## Simulation Study Design

The thesis conducts simulation studies varying:

1. **Right-censoring level** (q from 60% to 100%)
2. **Masking probability** (p from 10% to 70%)
3. **Sample size** (n from 50 to 500)

Key findings:
- MLE shows positive bias that decreases with larger samples
- BCa bootstrap confidence intervals have good coverage (>90%)
- Shape parameters are harder to estimate than scale parameters
- Convergence issues occur with heavy masking (p > 0.4)

## Running the Full Example

```bash
cd examples/
python thesis_reproduction.py        # Quick demo
python thesis_reproduction.py sim 100  # 100 simulation replicates
```

## References

1. Towell, A. (2023). *Reliability Estimation in Series Systems: Maximum Likelihood Techniques for Right-Censored and Masked Failure Data*. Masters Thesis.

2. Guo, H., Niu, Y., & Szidarovszky, F. (2013). Component failure parameter estimation for a series system based on masked failure data.

3. R package: [wei.series.md.c1.c2.c3](https://github.com/queelius/wei.series.md.c1.c2.c3)
