# Series Systems Reference

The `symlik.series` module provides tools for building likelihood models for series systems in reliability analysis.

## Architecture Overview

The module has three levels of abstraction:

```
Level 1: Component Hazards (most general)
    ↓
Level 2: Series System Compositions
    ↓
Level 3: Convenience Functions (easiest to use)
```

## Component Hazards

### ComponentHazard Class

```python
class ComponentHazard:
    hazard: ExprType           # h(t) - instantaneous hazard
    cumulative_hazard: ExprType  # H(t) = ∫₀ᵗ h(s)ds
    params: List[str]          # Parameter names
```

### Exponential Component

```python
from symlik.series import exponential_component

comp = exponential_component(
    rate: str,          # Rate parameter name
    time_var: str = "t" # Time variable name
) -> ComponentHazard
```

$$h(t) = \lambda, \quad H(t) = \lambda t$$

### Weibull Component

```python
from symlik.series import weibull_component

comp = weibull_component(
    shape: str,          # Shape parameter (k)
    scale: str,          # Scale parameter (θ)
    time_var: str = "t"
) -> ComponentHazard
```

$$h(t) = \frac{k}{\theta}\left(\frac{t}{\theta}\right)^{k-1}, \quad H(t) = \left(\frac{t}{\theta}\right)^k$$

### Custom Component

```python
from symlik.series import custom_component

comp = custom_component(
    hazard: ExprType,           # Any s-expression for h(t)
    cumulative_hazard: ExprType, # Any s-expression for H(t)
    params: List[str] = None
) -> ComponentHazard
```

### Exponential Proportional Hazards

```python
from symlik.series import exponential_ph_component

comp = exponential_ph_component(
    baseline_rate: str,      # λ₀
    coefficients: List[str], # [β₁, β₂, ...]
    covariates: List[str],   # [x₁, x₂, ...]
    time_var: str = "t"
) -> ComponentHazard
```

$$h(t|x) = \lambda_0 \exp(\beta_1 x_1 + \beta_2 x_2 + \cdots)$$

### Weibull Proportional Hazards

```python
from symlik.series import weibull_ph_component

comp = weibull_ph_component(
    shape: str,
    baseline_scale: str,
    coefficients: List[str],
    covariates: List[str],
    time_var: str = "t"
) -> ComponentHazard
```

### Log-Linear Exponential

```python
from symlik.series import exponential_log_linear_component

comp = exponential_log_linear_component(
    log_rate_intercept: str,  # α in log(λ) = α + βx
    coefficients: List[str],
    covariates: List[str],
    time_var: str = "t"
) -> ComponentHazard
```

$$\lambda = \exp(\alpha + \beta_1 x_1 + \cdots)$$

## Series System Compositions

These functions combine component hazards into observation-type contributions.

### Known Cause

```python
from symlik.series import series_known_cause

contrib = series_known_cause(
    components: List[ComponentHazard],
    cause_index: int  # 0-based index of failing component
) -> ExprType
```

$$\log L = \log h_j(t) - \sum_{i=1}^m H_i(t)$$

### Masked Cause (C1, C2, C3)

```python
from symlik.series import series_masked_cause

contrib = series_masked_cause(
    components: List[ComponentHazard],
    candidate_indices: List[int]  # 0-based indices in candidate set
) -> ExprType
```

$$\log L = \log\left(\sum_{j \in C} h_j(t)\right) - \sum_{i=1}^m H_i(t)$$

### Right-Censored

```python
from symlik.series import series_right_censored

contrib = series_right_censored(
    components: List[ComponentHazard]
) -> ExprType
```

$$\log L = -\sum_{i=1}^m H_i(t)$$

### Left-Censored

```python
from symlik.series import series_left_censored

contrib = series_left_censored(
    components: List[ComponentHazard]
) -> ExprType
```

$$\log L = \log\left(1 - \exp\left(-\sum_{i=1}^m H_i(t)\right)\right)$$

## Exponential Series Convenience Functions

Pre-built functions for exponential series systems.

```python
from symlik.series import (
    exponential_series_known_cause,
    exponential_series_masked_cause,
    exponential_series_right_censored,
    exponential_series_left_censored,
    exponential_series_interval_censored,
)
```

| Function | Parameters |
|----------|------------|
| `exponential_series_known_cause(rates, cause_index, time_var)` | Known failure cause |
| `exponential_series_masked_cause(rates, candidate_indices, time_var)` | Masked with candidate set |
| `exponential_series_right_censored(rates, time_var)` | Right-censored |
| `exponential_series_left_censored(rates, time_var)` | Left-censored |
| `exponential_series_interval_censored(rates, lower_var, upper_var)` | Interval-censored |

## Weibull Series Convenience Functions

```python
from symlik.series import (
    weibull_series_known_cause,
    weibull_series_masked_cause,
    weibull_series_right_censored,
    weibull_series_left_censored,
)
```

| Function | Parameters |
|----------|------------|
| `weibull_series_known_cause(shapes, scales, cause_index, time_var)` | Known failure cause |
| `weibull_series_masked_cause(shapes, scales, candidate_indices, time_var)` | Masked |
| `weibull_series_right_censored(shapes, scales, time_var)` | Right-censored |
| `weibull_series_left_censored(shapes, scales, time_var)` | Left-censored |

## Mixed Series Functions

For systems with different component types.

```python
from symlik.series import (
    mixed_series_known_cause,
    mixed_series_masked_cause,
    mixed_series_right_censored,
    mixed_series_left_censored,
)
```

All take `components: List[ComponentHazard]` as first argument.

## Factory Functions

Generate complete contribution dictionaries for use with `ContributionModel`.

### build_exponential_series_contributions

```python
from symlik.series import build_exponential_series_contributions

contribs = build_exponential_series_contributions(
    m: int,                              # Number of components
    rate_names: List[str] = None,        # Default: ["lambda1", ...]
    time_var: str = "t",
    include_left_censored: bool = False,
    include_interval_censored: bool = False,
    lower_time_var: str = "t_lower",
    upper_time_var: str = "t_upper",
) -> Dict[str, ExprType]
```

**Returns** dictionary with keys:
- `known_1`, `known_2`, ..., `known_m`
- `masked_12`, `masked_13`, ..., `masked_123...m` (all subsets of size ≥2)
- `right_censored`
- `left_censored` (if requested)
- `interval_censored` (if requested)

### build_weibull_series_contributions

```python
from symlik.series import build_weibull_series_contributions

contribs = build_weibull_series_contributions(
    m: int,
    shape_names: List[str] = None,   # Default: ["k1", "k2", ...]
    scale_names: List[str] = None,   # Default: ["theta1", "theta2", ...]
    time_var: str = "t",
    include_left_censored: bool = False,
) -> Dict[str, ExprType]
```

## Complete Example

```python
from symlik import ContributionModel
from symlik.series import (
    exponential_component,
    weibull_component,
    series_known_cause,
    series_masked_cause,
    series_right_censored,
)

# Mixed system: exponential + Weibull
components = [
    exponential_component("lambda"),
    weibull_component("k", "theta"),
]

# Build contributions manually
contribs = {
    "known_1": series_known_cause(components, 0),
    "known_2": series_known_cause(components, 1),
    "masked_12": series_masked_cause(components, [0, 1]),
    "right_censored": series_right_censored(components),
}

# Create model
model = ContributionModel(
    params=["lambda", "k", "theta"],
    type_column="obs_type",
    contributions=contribs,
)

# Fit
data = {
    "obs_type": ["known_1", "known_2", "masked_12", "right_censored"],
    "t": [1.2, 0.8, 1.5, 3.0],
}

mle, _ = model.mle(
    data=data,
    init={"lambda": 0.5, "k": 1.5, "theta": 1.0},
    bounds={"lambda": (0.01, 5), "k": (0.5, 5), "theta": (0.1, 10)},
)
```

## Mathematical Background

### Series System Survival

For independent components:

$$S_{\text{sys}}(t) = \prod_{i=1}^m S_i(t) = \exp\left(-\sum_{i=1}^m H_i(t)\right)$$

### Series System Hazard

$$h_{\text{sys}}(t) = \sum_{i=1}^m h_i(t)$$

### C1, C2, C3 Conditions

Under these conditions on the masking mechanism:

- **C1**: The true cause is always in the candidate set
- **C2**: $P(C|J \in C)$ is the same for all $J \in C$
- **C3**: Masking probabilities are independent of $\theta$

The likelihood for a masked observation with candidate set $C$ is:

$$L \propto \left(\sum_{j \in C} h_j(t)\right) S_{\text{sys}}(t)$$

The masking probability $P(C|J)$ factors out and doesn't affect the MLE.
