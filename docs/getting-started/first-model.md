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
mle, iterations = model.mle(data=data, init={'lambda': 1.0})

print(f"Estimated rate: {mle['lambda']:.4f}")
print(f"Converged in {iterations} iterations")
```

The `mle()` method uses Newton-Raphson optimization with symbolically-computed derivatives.

## Understanding the Output

The MLE for an exponential distribution is \(\hat\lambda = 1/\bar{x}\):

```python
import numpy as np
sample_mean = np.mean(data['x'])
print(f"Sample mean: {sample_mean:.4f}")
print(f"1/mean: {1/sample_mean:.4f}")
print(f"MLE: {mle['lambda']:.4f}")
# These should match
```

## Computing Standard Errors

Standard errors quantify uncertainty in your estimates:

```python
se = model.se(mle, data)
print(f"SE(lambda): {se['lambda']:.4f}")
```

The standard error comes from the observed Fisher information:

\[ \text{SE}(\hat\lambda) = \sqrt{I(\hat\lambda)^{-1}} \]

For the exponential distribution, this works out to \(\hat\lambda / \sqrt{n}\).

## Confidence Intervals

Build a 95% confidence interval:

```python
ci_lower = mle['lambda'] - 1.96 * se['lambda']
ci_upper = mle['lambda'] + 1.96 * se['lambda']
print(f"95% CI: ({ci_lower:.3f}, {ci_upper:.3f})")
```

## What Happened Behind the Scenes

When you called `model.mle()`, symlik:

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
score_at_mle = model.score_at({**data, **mle})
print(f"Score at MLE: {score_at_mle}")  # Should be ~0
```

## Adding Bounds

Some parameters have natural constraints. For rate parameters, we need \(\lambda > 0\):

```python
mle, _ = model.mle(
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
