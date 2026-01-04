# ContributionModel Reference

The `ContributionModel` class handles heterogeneous likelihood contributions where different observations have different likelihood forms.

## Class: ContributionModel

```python
from symlik import ContributionModel

model = ContributionModel(
    params: List[str],
    type_column: str,
    contributions: Dict[str, ExprType],
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `params` | `List[str]` | Parameter names shared across all contribution types |
| `type_column` | `str` | Name of the column containing observation types |
| `contributions` | `Dict[str, ExprType]` | Mapping from type names to log-likelihood s-expressions |

### Methods

#### fit

```python
fit = model.fit(
    data: Union[dict, DataFrame],
    init: Dict[str, float],
    max_iter: int = 100,
    tol: float = 1e-8,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> FittedLikelihoodModel
```

Find maximum likelihood estimates and return a fitted model object.

| Parameter | Description |
|-----------|-------------|
| `data` | Data as dict, pandas DataFrame, or polars DataFrame |
| `init` | Initial parameter values |
| `max_iter` | Maximum Newton-Raphson iterations |
| `tol` | Convergence tolerance (score norm) |
| `bounds` | Optional parameter bounds `{param: (min, max)}` |

**Returns**: `FittedLikelihoodModel` with properties:
- `params` - MLE estimates
- `se` - Standard errors
- `llf`, `aic`, `bic` - Fit statistics
- `conf_int()`, `wald_test()`, `summary()` - Inference methods

#### evaluate

```python
ll = model.evaluate(data_and_params: Dict[str, Any]) -> float
```

Evaluate log-likelihood. The input dict must contain both data columns (including type column) and parameter values.

#### score / hessian / information

```python
score = model.score() -> List[ExprType]
hess = model.hessian() -> List[List[ExprType]]
info = model.information() -> List[List[ExprType]]
```

Return symbolic derivatives of the composite log-likelihood.

#### score_at / hessian_at / information_at

```python
score_vals = model.score_at(data_and_params) -> np.ndarray
hess_vals = model.hessian_at(data_and_params) -> np.ndarray
info_vals = model.information_at(data_and_params) -> np.ndarray
```

Evaluate derivatives numerically.

## Contribution Constructors

The `symlik.contributions` module provides pre-built contribution expressions.

### Exponential Distribution

```python
from symlik.contributions import (
    complete_exponential,
    right_censored_exponential,
    left_censored_exponential,
    interval_censored_exponential,
)
```

| Function | Formula | Description |
|----------|---------|-------------|
| `complete_exponential(time_var, rate)` | $\log\lambda - \lambda t$ | Observed failure |
| `right_censored_exponential(time_var, rate)` | $-\lambda t$ | Survived past $t$ |
| `left_censored_exponential(time_var, rate)` | $\log(1 - e^{-\lambda t})$ | Failed before $t$ |
| `interval_censored_exponential(lower, upper, rate)` | $\log(S(t_l) - S(t_u))$ | Failed in $(t_l, t_u]$ |

**Default parameters**: `time_var="t"`, `rate="lambda"`

### Weibull Distribution

```python
from symlik.contributions import (
    complete_weibull,
    right_censored_weibull,
)
```

| Function | Description |
|----------|-------------|
| `complete_weibull(time_var, shape, scale)` | Complete Weibull observation |
| `right_censored_weibull(time_var, shape, scale)` | Right-censored Weibull |

**Default parameters**: `time_var="t"`, `shape="k"`, `scale="lambda"`

### Normal Distribution

```python
from symlik.contributions import complete_normal
```

| Function | Description |
|----------|-------------|
| `complete_normal(data_var, mean, var)` | Normal observation |

**Default parameters**: `data_var="x"`, `mean="mu"`, `var="sigma2"`

### Discrete Distributions

```python
from symlik.contributions import (
    complete_poisson,
    complete_bernoulli,
)
```

| Function | Description |
|----------|-------------|
| `complete_poisson(count_var, rate)` | Poisson count (ignoring factorial) |
| `complete_bernoulli(outcome_var, prob)` | Bernoulli outcome |

### Series System Contributions

```python
from symlik.contributions import (
    series_exponential_known_cause,
    series_exponential_masked_cause,
    series_exponential_right_censored,
)
```

See [Series Systems](../tutorials/series-systems.md) for detailed documentation.

## Example

```python
from symlik import ContributionModel
from symlik.contributions import (
    complete_exponential,
    right_censored_exponential,
)

# Model with complete and right-censored observations
model = ContributionModel(
    params=["lambda"],
    type_column="status",
    contributions={
        "observed": complete_exponential(time_var="duration"),
        "censored": right_censored_exponential(time_var="duration"),
    }
)

# Data
data = {
    "status": ["observed", "observed", "censored", "observed", "censored"],
    "duration": [1.2, 0.8, 3.0, 1.5, 2.5],
}

# Fit and access results
fit = model.fit(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})

print(f"λ = {fit.params['lambda']:.3f} ± {fit.se['lambda']:.3f}")
print(f"95% CI: {fit.conf_int()['lambda']}")
print(fit.summary())
```

## Internal Details

### Data Preparation

`ContributionModel` splits data by observation type before evaluation:

```python
# Input
{"obs_type": ["A", "B", "A"], "t": [1, 2, 3]}

# Becomes
{"t_A": [1, 3], "t_B": [2]}
```

### Composite Log-Likelihood

The model builds a composite log-likelihood that sums contributions by type:

```python
# Conceptually:
log_lik = sum_over_type_A(contrib_A) + sum_over_type_B(contrib_B) + ...
```

This is then wrapped in a `LikelihoodModel` for optimization.
