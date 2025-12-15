# Series Systems

This tutorial covers likelihood models for series systems in reliability analysis.

## What is a Series System?

A **series system** fails when any component fails. Think of components connected in series - if one breaks, the system breaks.

$$T_{\text{system}} = \min(T_1, T_2, \ldots, T_m)$$

Examples:
- A chain (fails when any link breaks)
- A circuit with components in series
- A manufacturing process with sequential steps

## The Challenge

In reliability data, we often don't observe perfect information:

- **Known cause**: We see which component failed
- **Masked cause**: We know the system failed, but only have a candidate set of possible causes
- **Censored**: The system was still working when observation ended

The `symlik.series` module handles all these cases.

## Quick Start

```python
from symlik import ContributionModel
from symlik.series import build_exponential_series_contributions

# 3-component system with exponential lifetimes
contribs = build_exponential_series_contributions(m=3)

model = ContributionModel(
    params=["lambda1", "lambda2", "lambda3"],
    type_column="obs_type",
    contributions=contribs,
)

# Data with various observation types
data = {
    "obs_type": ["known_1", "known_2", "masked_12", "right_censored"],
    "t": [1.2, 0.8, 1.5, 3.0],
}

mle, _ = model.mle(data=data, init={"lambda1": 0.5, "lambda2": 0.5, "lambda3": 0.5})
```

## Series System Likelihood

For a series system with $m$ exponential components (rates $\lambda_1, \ldots, \lambda_m$):

### Known Cause

When component $j$ causes failure at time $t$:

$$\log L = \log(\lambda_j) - t \sum_{i=1}^m \lambda_i$$

The first term is the hazard of the failing component; the second is system survival.

### Masked Cause (C1, C2, C3 Conditions)

When we only know the cause is in candidate set $C$:

$$\log L = \log\left(\sum_{j \in C} \lambda_j\right) - t \sum_{i=1}^m \lambda_i$$

Under the **C1, C2, C3 conditions**:

- **C1**: True cause is always in the candidate set
- **C2**: Candidate set probability independent of which component in set failed
- **C3**: Masking probabilities independent of parameters

The masking probability factors out and doesn't affect the MLE.

### Right-Censored

System survived past time $t$:

$$\log L = -t \sum_{i=1}^m \lambda_i$$

### Left-Censored

System failed before time $t$:

$$\log L = \log\left(1 - \exp\left(-t \sum_{i=1}^m \lambda_i\right)\right)$$

## Building Contributions

### Factory Functions

The easiest way to build a complete model:

```python
from symlik.series import build_exponential_series_contributions

# Creates all observation types for m=3 system
contribs = build_exponential_series_contributions(
    m=3,
    rate_names=["r1", "r2", "r3"],  # Custom parameter names
    include_left_censored=True,
    include_interval_censored=True,
)

# Result includes:
# - known_1, known_2, known_3
# - masked_12, masked_13, masked_23, masked_123
# - right_censored
# - left_censored (if requested)
# - interval_censored (if requested)
```

For Weibull components:

```python
from symlik.series import build_weibull_series_contributions

contribs = build_weibull_series_contributions(
    m=2,
    shape_names=["k1", "k2"],
    scale_names=["theta1", "theta2"],
)
```

### Manual Construction

For more control, build contributions individually:

```python
from symlik.series import (
    exponential_series_known_cause,
    exponential_series_masked_cause,
    exponential_series_right_censored,
)

rates = ["lambda1", "lambda2", "lambda3"]

contribs = {
    "known_1": exponential_series_known_cause(rates, cause_index=0),
    "known_2": exponential_series_known_cause(rates, cause_index=1),
    "known_3": exponential_series_known_cause(rates, cause_index=2),
    "masked_12": exponential_series_masked_cause(rates, candidate_indices=[0, 1]),
    "right_censored": exponential_series_right_censored(rates),
}
```

## Component Building Blocks

For maximum flexibility, work with component hazards directly:

```python
from symlik.series import (
    exponential_component,
    weibull_component,
    series_known_cause,
    series_masked_cause,
)

# Define components
comp1 = exponential_component("lambda1")
comp2 = weibull_component("k2", "theta2")  # Mixed types!

components = [comp1, comp2]

# Build contributions
contrib_known_1 = series_known_cause(components, cause_index=0)
contrib_masked = series_masked_cause(components, candidate_indices=[0, 1])
```

### Custom Components

Define any hazard function:

```python
from symlik.series import custom_component

# Linear increasing hazard: h(t) = αt, H(t) = αt²/2
comp = custom_component(
    hazard=["*", "alpha", "t"],
    cumulative_hazard=["*", 0.5, ["*", "alpha", ["^", "t", 2]]],
    params=["alpha"]
)
```

## Covariate-Dependent Hazards

Component failure rates can depend on covariates (proportional hazards):

```python
from symlik.series import exponential_ph_component, series_known_cause

# Component 1: rate depends on temperature
comp1 = exponential_ph_component(
    baseline_rate="lambda0_1",
    coefficients=["beta_temp"],
    covariates=["temperature"],
)

# Component 2: rate depends on temperature and load
comp2 = exponential_ph_component(
    baseline_rate="lambda0_2",
    coefficients=["beta_temp", "beta_load"],
    covariates=["temperature", "load"],
)

components = [comp1, comp2]
contrib = series_known_cause(components, cause_index=0)
```

The hazard becomes:

$$h_j(t|x) = \lambda_{0,j} \exp(\beta_1 x_1 + \beta_2 x_2 + \cdots)$$

## Complete Example

```python
import numpy as np
import pandas as pd
from symlik import ContributionModel
from symlik.series import build_exponential_series_contributions

# True parameters
true_rates = [0.3, 0.5, 0.2]  # λ₁, λ₂, λ₃
n = 200
mask_prob = 0.3
censor_time = 5.0

# Simulate data
np.random.seed(42)
records = []

for _ in range(n):
    # Component lifetimes
    lifetimes = [np.random.exponential(1/r) for r in true_rates]
    sys_time = min(lifetimes)
    true_cause = lifetimes.index(sys_time)

    if sys_time > censor_time:
        # Right-censored
        records.append({"obs_type": "right_censored", "t": censor_time})
    elif np.random.random() < mask_prob:
        # Masked: candidate set contains true cause + one random other
        others = [i for i in range(3) if i != true_cause]
        partner = np.random.choice(others)
        candidates = sorted([true_cause, partner])
        key = "masked_" + "".join(str(c+1) for c in candidates)
        records.append({"obs_type": key, "t": sys_time})
    else:
        # Known cause
        records.append({"obs_type": f"known_{true_cause+1}", "t": sys_time})

df = pd.DataFrame(records)

# Build and fit model
contribs = build_exponential_series_contributions(3)
model = ContributionModel(
    params=["lambda1", "lambda2", "lambda3"],
    type_column="obs_type",
    contributions=contribs,
)

mle, _ = model.mle(
    data=df,
    init={"lambda1": 0.5, "lambda2": 0.5, "lambda3": 0.5},
    bounds={f"lambda{i}": (0.01, 5) for i in range(1, 4)},
)

se = model.se(mle, df)

# Results
print("Component | True | MLE | SE")
print("-" * 40)
for i in range(3):
    p = f"lambda{i+1}"
    print(f"    {i+1}     | {true_rates[i]:.2f} | {mle[p]:.3f} | {se[p]:.3f}")
```

## Weibull Series Systems

For Weibull components with shape $k$ and scale $\theta$:

```python
from symlik.series import (
    weibull_series_known_cause,
    weibull_series_masked_cause,
    weibull_series_right_censored,
)

shapes = ["k1", "k2"]
scales = ["theta1", "theta2"]

contribs = {
    "known_1": weibull_series_known_cause(shapes, scales, cause_index=0),
    "known_2": weibull_series_known_cause(shapes, scales, cause_index=1),
    "masked_12": weibull_series_masked_cause(shapes, scales, candidate_indices=[0, 1]),
    "right_censored": weibull_series_right_censored(shapes, scales),
}

model = ContributionModel(
    params=["k1", "k2", "theta1", "theta2"],
    type_column="obs_type",
    contributions=contribs,
)
```

## Mixed Component Types

Systems with different component distributions:

```python
from symlik.series import (
    exponential_component,
    weibull_component,
    mixed_series_known_cause,
    mixed_series_right_censored,
)

# Component 1: exponential
# Component 2: Weibull
components = [
    exponential_component("lambda"),
    weibull_component("k", "theta"),
]

contribs = {
    "known_1": mixed_series_known_cause(components, cause_index=0),
    "known_2": mixed_series_known_cause(components, cause_index=1),
    "right_censored": mixed_series_right_censored(components),
}

model = ContributionModel(
    params=["lambda", "k", "theta"],
    type_column="obs_type",
    contributions=contribs,
)
```

## API Reference

### Component Constructors

| Function | Description |
|----------|-------------|
| `exponential_component(rate)` | Constant hazard $h(t) = \lambda$ |
| `weibull_component(shape, scale)` | Weibull hazard |
| `custom_component(hazard, cumulative_hazard)` | Any closed-form hazard |
| `exponential_ph_component(baseline, coeffs, covars)` | Proportional hazards |
| `weibull_ph_component(shape, scale, coeffs, covars)` | Weibull PH |

### Series System Compositions

| Function | Description |
|----------|-------------|
| `series_known_cause(components, cause_index)` | Known failure cause |
| `series_masked_cause(components, candidate_indices)` | Masked cause (C1,C2,C3) |
| `series_right_censored(components)` | Right-censored |
| `series_left_censored(components)` | Left-censored |

### Convenience Functions

| Function | Description |
|----------|-------------|
| `exponential_series_known_cause(rates, idx)` | Exponential, known cause |
| `exponential_series_masked_cause(rates, indices)` | Exponential, masked |
| `weibull_series_known_cause(shapes, scales, idx)` | Weibull, known cause |
| `build_exponential_series_contributions(m)` | All types for m components |
| `build_weibull_series_contributions(m)` | All types for m Weibull components |
