# LikelihoodModel Reference

The core class for defining and fitting statistical models.

## Constructor

```python
from symlik import LikelihoodModel

model = LikelihoodModel(log_lik, params)
```

**Arguments:**

- `log_lik` (ExprType): Log-likelihood as an s-expression
- `params` (List[str]): List of parameter names to estimate

**Example:**

```python
# Exponential log-likelihood
log_lik = ['sum', 'i', ['len', 'x'],
           ['+', ['log', 'lambda'],
            ['*', -1, ['*', 'lambda', ['@', 'x', 'i']]]]]

model = LikelihoodModel(log_lik, params=['lambda'])
```

---

## Fitting Models

### fit(data, init, max_iter=100, tol=1e-8, bounds=None)

```python
fit = model.fit(data, init, bounds=bounds)
```

Fit the model using maximum likelihood estimation. Returns a `FittedLikelihoodModel` object with estimates, standard errors, and inference methods.

**Arguments:**

- `data` (Dict[str, Any] | DataFrame): Data values (dict or pandas/polars DataFrame)
- `init` (Dict[str, float]): Initial parameter guesses
- `max_iter` (int): Maximum iterations. Default: 100
- `tol` (float): Convergence tolerance (score norm). Default: 1e-8
- `bounds` (Dict[str, Tuple[float, float]]): Parameter bounds. Default: None

**Returns:** `FittedLikelihoodModel`

**Example:**

```python
fit = model.fit(
    data={'x': [1, 2, 3, 4, 5]},
    init={'lambda': 1.0},
    bounds={'lambda': (0.01, 10.0)}
)

# Access results
print(f"MLE: {fit.params['lambda']:.4f}")
print(f"SE: {fit.se['lambda']:.4f}")
print(f"Log-likelihood: {fit.llf:.4f}")
print(f"AIC: {fit.aic:.4f}")
```

---

## FittedLikelihoodModel

The object returned by `model.fit()`. Follows statsmodels conventions.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `params` | `Dict[str, float]` | MLE parameter estimates |
| `se` / `bse` | `Dict[str, float]` | Standard errors (Wald) |
| `llf` | `float` | Log-likelihood at MLE |
| `aic` | `float` | Akaike Information Criterion |
| `bic` | `float` | Bayesian Information Criterion |
| `nobs` | `int` | Number of observations |
| `df_model` | `int` | Number of parameters |
| `n_iter` | `int` | Optimization iterations |
| `converged` | `bool` | Whether optimization converged |

### Methods

#### conf_int(alpha=0.05)

```python
ci = fit.conf_int(alpha=0.05)
# ci['lambda'] = (lower, upper)
```

Compute Wald confidence intervals.

**Returns:** `Dict[str, Tuple[float, float]]`

---

#### cov_params()

```python
cov = fit.cov_params()
```

Returns the variance-covariance matrix of parameter estimates.

**Returns:** `numpy.ndarray`

---

#### wald_test(null)

```python
result = fit.wald_test({'lambda': 0.5})
# result = {'statistic': ..., 'p_value': ..., 'df': ...}
```

Test null hypothesis using Wald test.

**Arguments:**

- `null` (Dict[str, float]): Null hypothesis values

**Returns:** Dict with `statistic`, `p_value`, `df`

---

#### score_test(null)

```python
result = fit.score_test({'lambda': 0.5})
```

Test null hypothesis using Score (Lagrange multiplier) test.

**Returns:** Dict with `statistic`, `p_value`, `df`

---

#### lr_test(null)

```python
result = fit.lr_test({'lambda': 0.5})
```

Test null hypothesis using Likelihood Ratio test.

**Returns:** Dict with `statistic`, `p_value`, `df`

---

#### summary(alpha=0.05)

```python
print(fit.summary())
```

Returns a formatted summary table with estimates, standard errors, confidence intervals, and fit statistics.

**Returns:** `str`

---

## Symbolic Methods

### score()

```python
score_exprs = model.score()
```

Returns the score vector (gradient of log-likelihood).

**Returns:** List of symbolic expressions, one per parameter.

**Example:**

```python
score = model.score()
print(f"d(log L)/d(lambda) = {score[0]}")
```

---

### hessian()

```python
hess_exprs = model.hessian()
```

Returns the Hessian matrix of log-likelihood.

**Returns:** 2D list of symbolic expressions.

---

### information()

```python
info_exprs = model.information()
```

Returns the observed Fisher information (negative Hessian).

**Returns:** 2D list of symbolic expressions.

---

## Numerical Evaluation

### evaluate(env)

```python
ll_value = model.evaluate(env)
```

Evaluate log-likelihood at given values.

**Arguments:**

- `env` (Dict[str, Any]): Variable bindings (data and parameters)

**Returns:** float

---

### score_at(env)

```python
score_values = model.score_at(env)
```

Evaluate score vector at given values.

**Returns:** numpy.ndarray

---

### hessian_at(env)

```python
H = model.hessian_at(env)
```

Evaluate Hessian matrix at given values.

**Returns:** numpy.ndarray (2D)

---

### information_at(env)

```python
I = model.information_at(env)
```

Evaluate Fisher information at given values.

**Returns:** numpy.ndarray (2D)

---

## Properties

### log_lik

The log-likelihood s-expression.

### params

List of parameter names.

---

## Notes

**Caching:** Score and Hessian expressions are computed once and cached. Repeated calls are fast.

**Bounds:** When parameters have constraints (e.g., positive), use bounds to keep the optimizer in valid regions.

**Convergence:** Check `fit.converged` to verify optimization succeeded. If False, try different starting values or bounds.

**Singular information:** If the information matrix is singular at the MLE, standard errors may be NaN for affected parameters.
