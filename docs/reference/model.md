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

**Example:**

```python
H = model.hessian()
print(f"d2(log L)/dlambda2 = {H[0][0]}")
```

---

### information()

```python
info_exprs = model.information()
```

Returns the observed Fisher information (negative Hessian).

**Returns:** 2D list of symbolic expressions.

---

## Numerical Methods

### evaluate(env)

```python
ll_value = model.evaluate(env)
```

Evaluate log-likelihood at given values.

**Arguments:**

- `env` (Dict[str, Any]): Variable bindings (data and parameters)

**Returns:** float

**Example:**

```python
data = {'x': [1, 2, 3]}
params = {'lambda': 0.5}
ll = model.evaluate({**data, **params})
```

---

### score_at(env)

```python
score_values = model.score_at(env)
```

Evaluate score vector at given values.

**Arguments:**

- `env` (Dict[str, Any]): Variable bindings

**Returns:** numpy.ndarray

---

### hessian_at(env)

```python
H = model.hessian_at(env)
```

Evaluate Hessian matrix at given values.

**Arguments:**

- `env` (Dict[str, Any]): Variable bindings

**Returns:** numpy.ndarray (2D)

---

### information_at(env)

```python
I = model.information_at(env)
```

Evaluate Fisher information at given values.

**Arguments:**

- `env` (Dict[str, Any]): Variable bindings

**Returns:** numpy.ndarray (2D)

---

## Inference Methods

### mle(data, init, max_iter=100, tol=1e-8, bounds=None)

```python
mle_estimates, iterations = model.mle(data, init, bounds=bounds)
```

Find maximum likelihood estimates using Newton-Raphson.

**Arguments:**

- `data` (Dict[str, Any]): Data values
- `init` (Dict[str, float]): Initial parameter guesses
- `max_iter` (int): Maximum iterations. Default: 100
- `tol` (float): Convergence tolerance (score norm). Default: 1e-8
- `bounds` (Dict[str, Tuple[float, float]]): Parameter bounds. Default: None

**Returns:** Tuple of (estimates dict, iteration count)

**Example:**

```python
mle, iters = model.mle(
    data={'x': [1, 2, 3, 4, 5]},
    init={'lambda': 1.0},
    bounds={'lambda': (0.01, 10.0)}
)
print(f"MLE: {mle['lambda']:.4f} in {iters} iterations")
```

---

### se(mle, data)

```python
standard_errors = model.se(mle, data)
```

Compute Wald standard errors at MLE.

**Formula:** \(\text{SE}(\hat\theta) = \sqrt{\text{diag}(I(\hat\theta)^{-1})}\)

**Arguments:**

- `mle` (Dict[str, float]): MLE parameter estimates
- `data` (Dict[str, Any]): Data values

**Returns:** Dict[str, float] mapping parameters to standard errors

**Example:**

```python
se = model.se(mle, data)
print(f"SE(lambda): {se['lambda']:.4f}")

# 95% confidence interval
ci_lower = mle['lambda'] - 1.96 * se['lambda']
ci_upper = mle['lambda'] + 1.96 * se['lambda']
```

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

**Convergence:** If `mle()` returns `max_iter` iterations, check if the optimizer converged. Try different starting values or bounds.

**Singular information:** If the information matrix is singular at the MLE, `se()` returns NaN for affected parameters.
