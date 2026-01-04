# Custom Models

This tutorial shows how to build your own likelihood models from scratch.

## The LikelihoodModel Class

To create a custom model, you need two things:

1. A log-likelihood as an s-expression
2. A list of parameter names

```python
from symlik import LikelihoodModel

model = LikelihoodModel(log_lik, params=['theta1', 'theta2'])
```

## Example: Custom Exponential

Let's rebuild the exponential distribution to see how it works.

The log-likelihood is:

\[ \ell(\lambda) = \sum_{i=1}^{n} \left[ \log(\lambda) - \lambda x_i \right] \]

In s-expression form:

```python
from symlik import LikelihoodModel

log_lik = ['sum', 'i', ['len', 'x'],
           ['+', ['log', 'lambda'],
            ['*', -1, ['*', 'lambda', ['@', 'x', 'i']]]]]

model = LikelihoodModel(log_lik, params=['lambda'])

# Test it
data = {'x': [1.0, 2.0, 3.0]}
fit = model.fit(data=data, init={'lambda': 1.0}, bounds={'lambda': (0.01, 10)})
print(f"MLE: {fit.params['lambda']:.4f}")  # Should be 1/mean = 0.5
```

## Inspecting Symbolic Derivatives

The model computes derivatives automatically. You can inspect them:

```python
# Score function (gradient)
score = model.score()
print(f"Score: {score}")

# Hessian matrix
hess = model.hessian()
print(f"Hessian: {hess}")

# Fisher information
info = model.information()
print(f"Information: {info}")
```

## Example: Linear Regression

Least squares regression is maximum likelihood under normal errors.

For \(y_i = \beta_0 + \beta_1 x_i + \epsilon_i\) with \(\epsilon_i \sim N(0, \sigma^2)\):

\[ \ell(\beta_0, \beta_1) = -\frac{1}{2\sigma^2} \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)^2 \]

```python
from symlik import LikelihoodModel

# Assuming known sigma^2 = 1 for simplicity
residual = ['-', ['@', 'y', 'i'],
            ['+', 'beta0', ['*', 'beta1', ['@', 'x', 'i']]]]

log_lik = ['*', -0.5,
           ['sum', 'i', ['len', 'y'],
            ['^', residual, 2]]]

model = LikelihoodModel(log_lik, params=['beta0', 'beta1'])

# Generate data: y = 2 + 3x + noise
import numpy as np
np.random.seed(42)
x = np.linspace(0, 1, 20)
y = 2 + 3*x + np.random.normal(0, 0.5, 20)

data = {'x': x.tolist(), 'y': y.tolist()}
fit = model.fit(data=data, init={'beta0': 0, 'beta1': 0})

print(f"Intercept: {fit.params['beta0']:.3f} (true: 2)")
print(f"Slope: {fit.params['beta1']:.3f} (true: 3)")
```

## Example: Mixture Model (Simplified)

A mixture of two exponentials with known mixing proportion:

\[ f(x) = p \cdot \lambda_1 e^{-\lambda_1 x} + (1-p) \cdot \lambda_2 e^{-\lambda_2 x} \]

For known \(p = 0.5\), the log-likelihood of one observation is:

```python
from symlik import LikelihoodModel

# Component densities (as log then exp for numerical stability)
comp1 = ['*', 'lambda1', ['exp', ['*', -1, ['*', 'lambda1', ['@', 'x', 'i']]]]]
comp2 = ['*', 'lambda2', ['exp', ['*', -1, ['*', 'lambda2', ['@', 'x', 'i']]]]]

# Mixture (p=0.5)
mixture = ['*', 0.5, ['+', comp1, comp2]]

log_lik = ['sum', 'i', ['len', 'x'], ['log', mixture]]

model = LikelihoodModel(log_lik, params=['lambda1', 'lambda2'])
```

Note: Mixture models can have multiple local optima. Starting values matter.

## Alternative Parameterizations

Sometimes a different parameterization improves optimization.

**Log-transform for positive parameters:**

Instead of estimating \(\lambda > 0\), estimate \(\theta = \log(\lambda)\):

```python
# Use exp(theta) where you would use lambda
log_lik = ['sum', 'i', ['len', 'x'],
           ['+', 'theta',  # log(exp(theta)) = theta
            ['*', -1, ['*', ['exp', 'theta'], ['@', 'x', 'i']]]]]

model = LikelihoodModel(log_lik, params=['theta'])

# After fitting, transform back
fit = model.fit(data=data, init={'theta': 0})
lambda_estimate = np.exp(fit.params['theta'])
```

## Evaluating the Log-Likelihood

Check your log-likelihood by evaluating it:

```python
from symlik import evaluate

# Direct evaluation
data = {'x': [1, 2, 3]}
params = {'lambda': 0.5}

ll_value = model.evaluate({**data, **params})
print(f"Log-likelihood at lambda=0.5: {ll_value:.4f}")

# Compare to manual calculation
import math
manual_ll = sum(math.log(0.5) - 0.5*xi for xi in data['x'])
print(f"Manual calculation: {manual_ll:.4f}")
```

## Debugging Tips

**Check dimensions.** Make sure your summation bounds match your data:

```python
# This iterates i from 1 to len(x)
['sum', 'i', ['len', 'x'], body]
```

**Test with simple data.** Use small datasets where you can verify by hand.

**Print intermediate values.** Evaluate sub-expressions to find errors:

```python
sub_expr = ['@', 'x', 1]  # First element
print(evaluate(sub_expr, {'x': [10, 20, 30]}))  # Should be 10
```

**Check for numerical issues.** Log of zero, division by zero, and overflow can cause problems. Use bounds to keep parameters in valid ranges.

## Next: Symbolic Calculus

Learn to use symlik's [calculus functions](symbolic-calculus.md) for standalone symbolic math.
