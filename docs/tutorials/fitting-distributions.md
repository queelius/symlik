# Fitting Distributions

This tutorial covers fitting the pre-built distributions in symlik to data.

## Exponential Distribution

Models waiting times, lifetimes, and inter-arrival times.

**Density:** \(f(x; \lambda) = \lambda e^{-\lambda x}\) for \(x \geq 0\)

**MLE:** \(\hat\lambda = 1/\bar{x}\)

```python
from symlik.distributions import exponential
import numpy as np

# Simulate data with true rate = 2
np.random.seed(42)
data = {'x': np.random.exponential(scale=1/2, size=100).tolist()}

# Fit
model = exponential()
fit = model.fit(data=data, init={'lambda': 1.0}, bounds={'lambda': (0.01, 10)})

print(f"True rate: 2.0")
print(f"Estimated: {fit.params['lambda']:.3f} (SE: {fit.se['lambda']:.3f})")
```

## Normal Distribution

The workhorse of statistics. symlik provides two versions.

### Both Parameters Unknown

```python
from symlik.distributions import normal

data = {'x': [4.2, 5.1, 4.8, 5.3, 4.9, 5.2, 4.7]}
model = normal()
fit = model.fit(
    data=data,
    init={'mu': 0, 'sigma2': 1},
    bounds={'sigma2': (0.01, 100)}  # Variance must be positive
)

print(f"Mean: {fit.params['mu']:.3f} (SE: {fit.se['mu']:.3f})")
print(f"Variance: {fit.params['sigma2']:.3f} (SE: {fit.se['sigma2']:.3f})")
```

### Known Variance

When variance is known (e.g., from prior studies), use `normal_mean()`:

```python
from symlik.distributions import normal_mean

data = {'x': [4.2, 5.1, 4.8, 5.3, 4.9]}
model = normal_mean(known_var=1.0)  # Assume sigma^2 = 1
fit = model.fit(data=data, init={'mu': 0})

print(f"Mean: {fit.params['mu']:.3f}")
```

## Poisson Distribution

Models count data: number of events in a fixed time/space.

**Probability:** \(P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}\)

**MLE:** \(\hat\lambda = \bar{x}\)

```python
from symlik.distributions import poisson

# Number of customer arrivals per hour
data = {'x': [3, 5, 2, 4, 6, 3, 4, 5, 2, 4]}
model = poisson()
fit = model.fit(data=data, init={'lambda': 1.0}, bounds={'lambda': (0.01, 100)})

print(f"Rate: {fit.params['lambda']:.3f} (SE: {fit.se['lambda']:.3f})")
print(f"Sample mean: {np.mean(data['x']):.3f}")  # Should match
```

## Bernoulli Distribution

Models binary outcomes: success/failure, yes/no.

**MLE:** \(\hat{p} = k/n\) where \(k\) is number of successes

```python
from symlik.distributions import bernoulli

# Survey responses: 1 = yes, 0 = no
data = {'x': [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]}
model = bernoulli()
fit = model.fit(data=data, init={'p': 0.5}, bounds={'p': (0.01, 0.99)})

print(f"Proportion: {fit.params['p']:.3f} (SE: {fit.se['p']:.3f})")
```

## Gamma Distribution

Generalizes exponential. Models positive continuous data with varying shapes.

```python
from symlik.distributions import gamma

# Simulate gamma data: shape=2, rate=1
np.random.seed(42)
data = {'x': np.random.gamma(shape=2, scale=1, size=200).tolist()}

model = gamma()
fit = model.fit(
    data=data,
    init={'alpha': 1.0, 'beta': 1.0},
    bounds={'alpha': (0.1, 10), 'beta': (0.1, 10)}
)

print(f"Shape (alpha): {fit.params['alpha']:.3f}")
print(f"Rate (beta): {fit.params['beta']:.3f}")
```

## Weibull Distribution

Models time-to-failure with varying hazard rates.

```python
from symlik.distributions import weibull

# Simulate Weibull data
np.random.seed(42)
data = {'x': np.random.weibull(a=2, size=100).tolist()}

model = weibull()
fit = model.fit(
    data=data,
    init={'k': 1.0, 'lambda': 1.0},
    bounds={'k': (0.1, 10), 'lambda': (0.1, 10)}
)

print(f"Shape (k): {fit.params['k']:.3f}")
print(f"Scale (lambda): {fit.params['lambda']:.3f}")
```

## Beta Distribution

Models proportions and probabilities (data in \((0, 1)\)).

```python
from symlik.distributions import beta

# Proportion data (e.g., percentage of time on task)
data = {'x': [0.3, 0.5, 0.4, 0.6, 0.35, 0.55, 0.45]}

model = beta()
fit = model.fit(
    data=data,
    init={'alpha': 1.0, 'beta': 1.0},
    bounds={'alpha': (0.1, 10), 'beta': (0.1, 10)}
)

print(f"Alpha: {fit.params['alpha']:.3f}")
print(f"Beta: {fit.params['beta']:.3f}")
```

## Working with Fitted Models

All fitted models provide a consistent interface:

```python
# Parameter estimates
fit.params['lambda']

# Standard errors
fit.se['lambda']

# Confidence intervals
ci = fit.conf_int(alpha=0.05)  # 95% CI
ci['lambda']  # (lower, upper)

# Fit statistics
fit.llf   # Log-likelihood
fit.aic   # Akaike Information Criterion
fit.bic   # Bayesian Information Criterion
fit.nobs  # Number of observations

# Summary table
print(fit.summary())

# Hypothesis testing
result = fit.wald_test({'lambda': 1.0})
print(f"p-value: {result['p_value']:.4f}")
```

## Tips for Fitting

**Choose good starting values.** Bad initial guesses can cause convergence issues. Use sample statistics as guides:

- Exponential: start with `1/mean`
- Poisson: start with `mean`
- Normal: start with sample mean and variance

**Use appropriate bounds.** Many parameters have natural constraints:

- Rate/scale parameters: `(0.01, upper)`
- Probabilities: `(0.01, 0.99)`
- Shape parameters: `(0.1, upper)`

**Check convergence.** The fitted model has a `converged` property. If False, the optimizer may not have found the MLE.

```python
if not fit.converged:
    print("Warning: optimization may not have converged")
```

## Next: Custom Models

When pre-built distributions are not enough, [build your own models](custom-models.md).
