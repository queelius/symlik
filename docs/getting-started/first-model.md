# Your First Model

This guide walks through fitting an exponential distribution to data. Along the way, you will see how symlik handles the statistical computations.

## The Setup

Suppose you have measurements of how long customers wait in a queue:

```python
wait_times = [2.1, 0.8, 1.5, 3.2, 1.1, 0.9, 2.4, 1.8]
```

The exponential distribution is a natural model for waiting times. Its probability density is:

\[ f(x; \lambda) = \lambda e^{-\lambda x} \]

where \(\lambda\) is the rate parameter (higher means shorter waits).

## Fitting the Model

```python
from symlik.distributions import exponential

# Create the model
model = exponential()

# Fit to data
data = {'x': [2.1, 0.8, 1.5, 3.2, 1.1, 0.9, 2.4, 1.8]}
fit = model.fit(data=data, init={'lambda': 1.0})

print(f"Estimated rate: {fit.params['lambda']:.4f}")
print(f"Converged: {fit.converged}")
```

The `fit()` method uses Newton-Raphson optimization with symbolically-computed derivatives and returns a `FittedLikelihoodModel` with all results.

## Understanding the Output

The MLE for an exponential distribution is \(\hat\lambda = 1/\bar{x}\):

```python
import numpy as np
sample_mean = np.mean(data['x'])
print(f"Sample mean: {sample_mean:.4f}")
print(f"1/mean: {1/sample_mean:.4f}")
print(f"MLE: {fit.params['lambda']:.4f}")
# These should match
```

## Standard Errors and Confidence Intervals

Standard errors and confidence intervals are available directly from the fitted model:

```python
# Standard error
print(f"SE(lambda): {fit.se['lambda']:.4f}")

# 95% confidence interval
ci = fit.conf_int()
print(f"95% CI: ({ci['lambda'][0]:.3f}, {ci['lambda'][1]:.3f})")
```

The standard error comes from the observed Fisher information:

\[ \text{SE}(\hat\lambda) = \sqrt{I(\hat\lambda)^{-1}} \]

For the exponential distribution, this works out to \(\hat\lambda / \sqrt{n}\).

## Model Fit Statistics

The fitted model provides information criteria and other fit statistics:

```python
print(f"Log-likelihood: {fit.llf:.4f}")
print(f"AIC: {fit.aic:.4f}")
print(f"BIC: {fit.bic:.4f}")
print(f"Observations: {fit.nobs}")
```

## Summary Table

Get a formatted summary of all results:

```python
print(fit.summary())
```

## What Happened Behind the Scenes

When you called `model.fit()`, symlik:

1. **Differentiated** the log-likelihood symbolically to get the score function
2. **Differentiated again** to get the Hessian matrix
3. **Iterated** Newton-Raphson: \(\theta_{new} = \theta - H^{-1} \nabla \ell\)
4. **Stopped** when the score was close to zero

You can inspect these symbolic quantities:

```python
# The score function (should be zero at MLE)
score = model.score()
print(f"Score expression: {score}")

# Evaluate at MLE
score_at_mle = model.score_at({**data, **fit.params})
print(f"Score at MLE: {score_at_mle}")  # Should be ~0
```

## Adding Bounds

Some parameters have natural constraints. For rate parameters, we need \(\lambda > 0\):

```python
fit = model.fit(
    data=data,
    init={'lambda': 1.0},
    bounds={'lambda': (0.01, 10.0)}
)
```

Bounds keep the optimizer in valid parameter space.

## Next Steps

- Learn the [s-expression syntax](s-expressions.md) to write your own models
- See [Fitting Distributions](../tutorials/fitting-distributions.md) for more distribution examples
- Build [Custom Models](../tutorials/custom-models.md) for non-standard likelihoods
