# symlik

**Symbolic Likelihood Models for Statistical Inference**

symlik lets you define statistical models symbolically and automatically derives everything needed for inference.

## The Idea

In maximum likelihood estimation, you need derivatives of the log-likelihood:

- The **score function** (gradient) tells you which direction increases likelihood
- The **Fisher information** (negative Hessian) measures how much the data tells you about parameters
- **Standard errors** come from inverting the information matrix

Traditionally, you derive these by hand or approximate them numerically. symlik takes a different approach: write the log-likelihood symbolically, and let the computer do the calculus.

## Quick Examples

### Simple Distribution Fitting

```python
from symlik.distributions import exponential

model = exponential()
data = {'x': [2.1, 0.8, 1.5, 3.2, 1.1]}
mle, _ = model.mle(data=data, init={'lambda': 1.0})
se = model.se(mle, data)

print(f"Rate: {mle['lambda']:.3f} ± {se['lambda']:.3f}")
# Rate: 0.580 ± 0.259
```

### Heterogeneous Data (Censoring, Masking)

```python
from symlik import ContributionModel
from symlik.contributions import complete_exponential, right_censored_exponential

model = ContributionModel(
    params=["lambda"],
    type_column="status",
    contributions={
        "observed": complete_exponential(),
        "censored": right_censored_exponential(),
    }
)

data = {
    "status": ["observed", "censored", "observed", "observed", "censored"],
    "t": [1.2, 3.0, 0.8, 2.1, 4.5],
}

mle, _ = model.mle(data=data, init={"lambda": 1.0})
```

### Series System Reliability

```python
from symlik import ContributionModel
from symlik.series import build_exponential_series_contributions

# 3-component series system
contribs = build_exponential_series_contributions(m=3)

model = ContributionModel(
    params=["lambda1", "lambda2", "lambda3"],
    type_column="obs_type",
    contributions=contribs,
)
```

## What You Can Do

**Use pre-built distributions** for common models like normal, exponential, Poisson, Weibull, and more.

**Handle heterogeneous data** with `ContributionModel` - mix complete observations, censored data, and masked cause failures in a single model.

**Model series systems** for reliability analysis - exponential or Weibull components, known or masked cause, with or without covariates.

**Build custom models** by writing log-likelihoods as s-expressions:

```python
from symlik import LikelihoodModel

log_lik = ['sum', 'i', ['len', 'x'],
           ['+', ['log', 'theta'],
            ['*', -1, ['*', 'theta', ['@', 'x', 'i']]]]]

model = LikelihoodModel(log_lik, params=['theta'])
```

**Use symbolic calculus directly** for other mathematical work:

```python
from symlik import diff, gradient, hessian

deriv = diff(['^', 'x', 3], 'x')  # 3x^2
```

## Next Steps

- [Installation](getting-started/installation.md) - Get symlik running
- [First Model](getting-started/first-model.md) - Fit your first distribution
- [Contribution Models](tutorials/contribution-models.md) - Handle censored and heterogeneous data
- [Series Systems](tutorials/series-systems.md) - Reliability analysis for multi-component systems
