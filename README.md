# symlik

**Symbolic Likelihood Models for Statistical Inference**

symlik lets you define statistical models symbolically and automatically derives everything needed for inference: score functions, Hessians, Fisher information, standard errors, and maximum likelihood estimates.

[![Documentation](https://img.shields.io/badge/docs-queelius.github.io%2Fsymlik-blue)](https://queelius.github.io/symlik/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why symlik?

Traditional statistical computing requires manually deriving score functions and information matrices, or relying on numerical approximations. symlik takes a different approach: write the log-likelihood symbolically, and let the computer handle the calculus.

```python
from symlik.distributions import exponential

model = exponential()
data = {'x': [1.2, 0.8, 2.1, 1.5]}

mle, _ = model.mle(data=data, init={'lambda': 1.0})
se = model.se(mle, data)

print(f"Rate: {mle['lambda']:.3f} ± {se['lambda']:.3f}")
# Rate: 0.714 ± 0.357
```

## Installation

```bash
pip install symlik
```

**Requirements:** Python 3.8+, NumPy, and [rerum](https://github.com/queelius/rerum) (symbolic rewriting engine).

## Key Features

- **Symbolic differentiation** - Automatic score functions and Hessians
- **Heterogeneous data** - Handle censoring, masking, and mixed observation types
- **Series systems** - Reliability analysis for multi-component systems
- **DataFrame support** - Works with pandas, polars, or plain dicts
- **Extensible** - Custom distributions, operators, and hazard functions

## Quick Start

### Pre-built Distributions

```python
from symlik.distributions import normal, poisson, exponential

# Normal distribution
model = normal()
mle, _ = model.mle(
    data={'x': [4.2, 5.1, 4.8, 5.3, 4.9]},
    init={'mu': 0, 'sigma2': 1},
    bounds={'sigma2': (0.01, None)}
)
```

### Custom Models

Define log-likelihoods using s-expressions:

```python
from symlik import LikelihoodModel

# Exponential: ℓ(λ) = Σ[log(λ) - λxᵢ]
log_lik = ['sum', 'i', ['len', 'x'],
           ['+', ['log', 'lambda'],
            ['*', -1, ['*', 'lambda', ['@', 'x', 'i']]]]]

model = LikelihoodModel(log_lik, params=['lambda'])

# Symbolic derivatives available
score = model.score()       # Gradient
hess = model.hessian()      # Hessian matrix
info = model.information()  # Fisher information
```

### Heterogeneous Data (Censoring)

Handle mixed observation types with `ContributionModel`:

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

Model multi-component systems with known cause, masked cause, or censored observations:

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

# Handles: known_1, known_2, known_3, masked_12, masked_13,
#          masked_23, masked_123, right_censored
```

**Masked cause** observations satisfy C1, C2, C3 conditions:
- C1: True cause always in candidate set
- C2: Candidate set probability independent of which component failed
- C3: Masking probabilities independent of parameters

### DataFrame Support

Works seamlessly with pandas and polars:

```python
import pandas as pd

df = pd.DataFrame({
    "status": ["observed", "censored", "observed"],
    "t": [1.2, 3.0, 0.8],
})

mle, _ = model.mle(data=df, init={"lambda": 1.0})  # Just works!
```

## S-Expression Syntax

symlik uses s-expressions to represent mathematical formulas:

| Expression | Meaning |
|------------|---------|
| `['+', 'x', 'y']` | x + y |
| `['*', 2, 'x']` | 2x |
| `['^', 'x', 2]` | x² |
| `['/', 'x', 'y']` | x / y |
| `['log', 'x']` | ln(x) |
| `['exp', 'x']` | eˣ |
| `['sum', 'i', 'n', body]` | Σᵢ₌₁ⁿ body |
| `['@', 'x', 'i']` | xᵢ (1-based indexing) |
| `['len', 'x']` | length of x |

## Available Distributions

| Distribution | Parameters | Use Case |
|--------------|------------|----------|
| `exponential()` | λ (rate) | Waiting times, lifetimes |
| `normal()` | μ, σ² | Continuous measurements |
| `poisson()` | λ | Count data |
| `bernoulli()` | p | Binary outcomes |
| `binomial()` | p | Proportion estimation |
| `gamma()` | α, β | Positive continuous data |
| `weibull()` | k, λ | Reliability, survival |
| `beta()` | α, β | Proportions, probabilities |

## Contribution Types

| Function | Formula | Use Case |
|----------|---------|----------|
| `complete_exponential()` | log(λ) - λt | Observed failure |
| `right_censored_exponential()` | -λt | Survived past t |
| `left_censored_exponential()` | log(1 - e⁻λᵗ) | Failed before t |
| `complete_weibull()` | Weibull density | Observed failure |
| `series_exponential_known_cause()` | Known component | Reliability |
| `series_exponential_masked_cause()` | Candidate set | Masked cause |

## Series System Components

Build custom reliability models:

```python
from symlik.series import (
    exponential_component,
    weibull_component,
    exponential_ph_component,  # Proportional hazards
    custom_component,          # Any hazard function
    series_known_cause,
    series_masked_cause,
)

# Mixed system: exponential + Weibull
components = [
    exponential_component("lambda"),
    weibull_component("k", "theta"),
]

# Covariate-dependent hazards
comp = exponential_ph_component(
    baseline_rate="lambda0",
    coefficients=["beta"],
    covariates=["temperature"],
)
```

## Direct Calculus Operations

Use symlik's calculus module for standalone symbolic math:

```python
from symlik import diff, gradient, hessian, simplify

# Differentiate x³ + 2x
expr = ['+', ['^', 'x', 3], ['*', 2, 'x']]
deriv = diff(expr, 'x')  # 3x² + 2

# Gradient of f(x,y) = x² + xy
expr = ['+', ['^', 'x', 2], ['*', 'x', 'y']]
grad = gradient(expr, ['x', 'y'])  # [2x + y, x]
```

## Documentation

Full documentation: **[queelius.github.io/symlik](https://queelius.github.io/symlik/)**

- [Getting Started](https://queelius.github.io/symlik/getting-started/installation/)
- [Contribution Models](https://queelius.github.io/symlik/tutorials/contribution-models/)
- [Series Systems](https://queelius.github.io/symlik/tutorials/series-systems/)
- [API Reference](https://queelius.github.io/symlik/reference/model/)

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or pull request on [GitHub](https://github.com/queelius/symlik).
