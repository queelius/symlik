# Contribution Models

This tutorial covers `ContributionModel` for heterogeneous data where different observations contribute differently to the likelihood.

## The Problem

Many real datasets have observations of different types:

- **Survival analysis**: Some patients die (complete), others are lost to follow-up (censored)
- **Reliability**: Some failures have known cause, others only a candidate set (masked)
- **Interval data**: Some measurements are exact, others are bounds

Each type contributes a different term to the log-likelihood. `ContributionModel` handles this elegantly.

## Basic Usage

```python
from symlik import ContributionModel
from symlik.contributions import complete_exponential, right_censored_exponential

model = ContributionModel(
    params=["lambda"],
    type_column="obs_type",  # Column that identifies observation type
    contributions={
        "complete": complete_exponential(),
        "censored": right_censored_exponential(),
    }
)
```

Your data needs a column indicating each observation's type:

```python
data = {
    "obs_type": ["complete", "censored", "complete", "complete", "censored"],
    "t": [1.2, 3.0, 0.8, 2.1, 4.5],
}

fit = model.fit(data=data, init={"lambda": 1.0})
```

## Understanding Contributions

A **contribution** is the log-likelihood term for a single observation. Different observation types have different contributions.

### Exponential Example

For exponential lifetime data with rate $\lambda$:

| Observation Type | What We Know | Log-Likelihood Contribution |
|-----------------|--------------|----------------------------|
| Complete | Exact failure time $t$ | $\log(\lambda) - \lambda t$ |
| Right-censored | Survived past time $t$ | $-\lambda t$ |
| Left-censored | Failed before time $t$ | $\log(1 - e^{-\lambda t})$ |

The total log-likelihood sums contributions across all observations:

$$\ell(\lambda) = \sum_{\text{complete}} [\log\lambda - \lambda t_i] + \sum_{\text{censored}} [-\lambda t_i] + \cdots$$

## Available Contributions

The `symlik.contributions` module provides common contributions:

### Exponential

```python
from symlik.contributions import (
    complete_exponential,      # log(λ) - λt
    right_censored_exponential,  # -λt
    left_censored_exponential,   # log(1 - exp(-λt))
    interval_censored_exponential,  # log(S(t_l) - S(t_u))
)
```

### Weibull

```python
from symlik.contributions import (
    complete_weibull,
    right_censored_weibull,
)
```

### Other Distributions

```python
from symlik.contributions import (
    complete_normal,
    complete_poisson,
    complete_bernoulli,
)
```

## Custom Variable Names

Contributions use default variable names (`t` for time, `lambda` for rate), but you can customize:

```python
# Use different column name
contrib = complete_exponential(time_var="duration", rate="failure_rate")

model = ContributionModel(
    params=["failure_rate"],
    type_column="status",
    contributions={"observed": contrib}
)

data = {
    "status": ["observed", "observed"],
    "duration": [1.5, 2.3],  # Matches time_var
}
```

## Example: Survival Analysis

A complete example with mixed censoring:

```python
import numpy as np
import pandas as pd
from symlik import ContributionModel
from symlik.contributions import (
    complete_exponential,
    right_censored_exponential,
    left_censored_exponential,
)

# Simulate data
np.random.seed(42)
true_lambda = 0.5
n = 200

# Generate true failure times
true_times = np.random.exponential(1/true_lambda, n)

# Apply censoring
records = []
for t in true_times:
    if t < 0.5:  # Left-censored
        records.append({"obs_type": "left_censored", "t": 0.5})
    elif t > 3.0:  # Right-censored
        records.append({"obs_type": "right_censored", "t": 3.0})
    else:  # Complete
        records.append({"obs_type": "complete", "t": t})

df = pd.DataFrame(records)

# Build model
model = ContributionModel(
    params=["lambda"],
    type_column="obs_type",
    contributions={
        "complete": complete_exponential(),
        "right_censored": right_censored_exponential(),
        "left_censored": left_censored_exponential(),
    }
)

# Fit
fit = model.fit(data=df, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})
# Standard errors via fit.se

print(f"True λ: {true_lambda}")
print(f"MLE λ: {fit.params['lambda']:.4f} ± {fit.se['lambda']:.4f}")
```

## DataFrame Support

`ContributionModel` accepts pandas DataFrames, polars DataFrames, or plain dictionaries:

```python
# All of these work
model.fit(data=df, ...)           # pandas DataFrame
model.fit(data=pl_df, ...)        # polars DataFrame
model.fit(data={"t": [...], ...})  # dict of lists
```

## Multiple Parameters

Models can have multiple parameters:

```python
from symlik.contributions import complete_weibull, right_censored_weibull

model = ContributionModel(
    params=["k", "lambda"],  # Shape and scale
    type_column="obs_type",
    contributions={
        "complete": complete_weibull(),
        "censored": right_censored_weibull(),
    }
)

fit = model.fit(
    data=df,
    init={"k": 1.0, "lambda": 1.0},
    bounds={"k": (0.1, 10), "lambda": (0.1, 10)}
)
```

## Writing Custom Contributions

Create your own contributions as s-expressions:

```python
# Custom: log-normal complete observation
# log f(t) = -log(t) - log(σ) - 0.5*log(2π) - (log(t)-μ)²/(2σ²)
def complete_lognormal(time_var="t", mu="mu", sigma="sigma"):
    log_t_minus_mu = ["-", ["log", time_var], mu]
    return ["+",
            ["*", -1, ["log", time_var]],
            ["*", -1, ["log", sigma]],
            ["*", -0.5, ["log", ["*", 2, 3.14159]]],
            ["*", -0.5, ["/", ["^", log_t_minus_mu, 2], ["^", sigma, 2]]]]
```

## Inspecting the Model

```python
# View the composite log-likelihood
print(model._composite_loglik)

# Symbolic score (gradient)
score = model.score()

# Symbolic Hessian
hess = model.hessian()

# Numerical evaluation
data_and_params = {**df.to_dict('list'), "lambda": 0.5}
ll = model.evaluate(data_and_params)
```

## Next: Series Systems

For reliability analysis of multi-component systems, see [Series Systems](series-systems.md).
