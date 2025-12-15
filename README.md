# symlik

**Symbolic Likelihood Models for Statistical Inference**

symlik lets you define statistical models symbolically and automatically derives everything you need for inference: gradients, Hessians, standard errors, and maximum likelihood estimates.

## Why symlik?

Traditional statistical computing requires you to manually derive score functions and information matrices, or rely on numerical approximations. symlik takes a different approach: you write down the log-likelihood as a symbolic expression, and the library handles the calculus.

```python
from symlik.distributions import exponential

# Create a model for exponential data
model = exponential()

# Fit to data - derivatives computed symbolically
mle, _ = model.mle(data={'x': [1.2, 0.8, 2.1, 1.5]}, init={'lambda': 1.0})
se = model.se(mle, data={'x': [1.2, 0.8, 2.1, 1.5]})

print(f"Rate estimate: {mle['lambda']:.3f} (SE: {se['lambda']:.3f})")
```

## Installation

```bash
pip install symlik
```

**Requirements:** Python 3.8+, NumPy, and [rerum](https://github.com/alextowell/rerum) (a symbolic rewriting engine).

## Quick Start

### Using Pre-built Distributions

symlik includes common distributions ready to use:

```python
from symlik.distributions import normal, poisson, bernoulli

# Normal distribution (estimate mean and variance)
model = normal()
mle, _ = model.mle(
    data={'x': [4.2, 5.1, 4.8, 5.3, 4.9]},
    init={'mu': 0, 'sigma2': 1},
    bounds={'sigma2': (0.01, None)}
)

# Poisson distribution (estimate rate)
model = poisson()
mle, _ = model.mle(data={'x': [2, 3, 1, 4, 2]}, init={'lambda': 1.0})
```

### Building Custom Models

Define your own log-likelihood using s-expressions:

```python
from symlik import LikelihoodModel

# Log-likelihood for exponential: sum of [log(lambda) - lambda * x_i]
log_lik = ['sum', 'i', ['len', 'x'],
           ['+', ['log', 'lambda'],
            ['*', -1, ['*', 'lambda', ['@', 'x', 'i']]]]]

model = LikelihoodModel(log_lik, params=['lambda'])

# Get symbolic derivatives
score = model.score()      # Gradient of log-likelihood
hess = model.hessian()     # Hessian matrix
info = model.information() # Fisher information (negative Hessian)
```

### The S-Expression Syntax

symlik uses s-expressions (like Lisp) to represent mathematical formulas:

| Expression | Meaning |
|------------|---------|
| `['+', 'x', 'y']` | x + y |
| `['*', 2, 'x']` | 2x |
| `['^', 'x', 2]` | x^2 |
| `['log', 'x']` | ln(x) |
| `['sum', 'i', 'n', body]` | sum from i=1 to n |
| `['@', 'x', 'i']` | x[i] (1-based indexing) |
| `['len', 'x']` | length of x |

## Key Features

- **Symbolic differentiation**: Automatic score functions and Hessians
- **Newton-Raphson MLE**: Fast convergence with symbolic derivatives
- **Wald standard errors**: Computed from observed Fisher information
- **Extensible**: Add custom operators or use the calculus functions directly

## Direct Calculus Operations

Use symlik's calculus module for standalone symbolic math:

```python
from symlik import diff, gradient, hessian, simplify

# Differentiate x^3 + 2x
expr = ['+', ['^', 'x', 3], ['*', 2, 'x']]
deriv = diff(expr, 'x')  # Returns: ['+', ['*', 3, ['^', 'x', 2]], 2]

# Compute gradient of f(x,y) = x^2 + xy
expr = ['+', ['^', 'x', 2], ['*', 'x', 'y']]
grad = gradient(expr, ['x', 'y'])  # [df/dx, df/dy]
```

## Available Distributions

| Distribution | Parameters | MLE Formula |
|--------------|------------|-------------|
| `exponential()` | lambda (rate) | 1/mean |
| `normal()` | mu, sigma2 | sample mean, sample variance |
| `normal_mean()` | mu (known variance) | sample mean |
| `poisson()` | lambda | sample mean |
| `bernoulli()` | p | proportion of 1s |
| `binomial()` | p | successes/trials |
| `gamma()` | alpha, beta | (numerical) |
| `weibull()` | k, lambda | (numerical) |
| `beta()` | alpha, beta | (numerical) |

## Documentation

Full documentation is available at the [documentation site](https://alextowell.github.io/symlik/).

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or pull request on [GitHub](https://github.com/alextowell/symlik).
