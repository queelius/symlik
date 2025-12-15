# Symbolic Calculus

symlik's calculus module provides symbolic differentiation beyond just likelihood models. This tutorial shows how to use it.

## Basic Differentiation

The `diff` function computes symbolic derivatives:

```python
from symlik import diff

# d/dx(x^3) = 3x^2
expr = ['^', 'x', 3]
deriv = diff(expr, 'x')
print(deriv)  # ['*', 3, ['^', 'x', 2]]
```

## Gradient

For functions of multiple variables, `gradient` returns a list of partial derivatives:

```python
from symlik import gradient

# f(x,y) = x^2 + xy + y^2
expr = ['+', ['+', ['^', 'x', 2], ['*', 'x', 'y']], ['^', 'y', 2]]

grad = gradient(expr, ['x', 'y'])
print(f"df/dx = {grad[0]}")  # 2x + y
print(f"df/dy = {grad[1]}")  # x + 2y
```

## Hessian

The `hessian` function computes the matrix of second partial derivatives:

```python
from symlik import hessian

expr = ['+', ['^', 'x', 3], ['*', 'x', 'y']]
H = hessian(expr, ['x', 'y'])

print(f"d2f/dx2 = {H[0][0]}")   # 6x
print(f"d2f/dxdy = {H[0][1]}")  # 1
print(f"d2f/dydx = {H[1][0]}")  # 1
print(f"d2f/dy2 = {H[1][1]}")   # 0
```

## Evaluating at Points

Combine symbolic differentiation with numerical evaluation:

```python
from symlik import diff, evaluate

expr = ['+', ['^', 'x', 3], ['*', 2, 'x']]
deriv = diff(expr, 'x')  # 3x^2 + 2

# Evaluate at x = 2
value = evaluate(deriv, {'x': 2})
print(f"f'(2) = {value}")  # 3*4 + 2 = 14
```

Or use the convenience functions:

```python
from symlik import diff_at, gradient_at, hessian_at

expr = ['+', ['^', 'x', 2], ['^', 'y', 2]]

# Gradient at (3, 4)
grad = gradient_at(expr, ['x', 'y'], {'x': 3, 'y': 4})
print(f"Gradient at (3,4): {grad}")  # [6, 8]

# Hessian at (3, 4)
H = hessian_at(expr, ['x', 'y'], {'x': 3, 'y': 4})
print(f"Hessian at (3,4):\n{H}")  # [[2, 0], [0, 2]]
```

## Simplification

The `simplify` function reduces expressions:

```python
from symlik import simplify

# x + 0 = x
expr = ['+', 'x', 0]
print(simplify(expr))  # 'x'

# x * 1 = x
expr = ['*', 'x', 1]
print(simplify(expr))  # 'x'

# x^0 = 1
expr = ['^', 'x', 0]
print(simplify(expr))  # 1
```

Derivatives are automatically simplified:

```python
from symlik import diff

# d/dx(5) = 0, not ['*', 0, 1] or something complicated
expr = 5
deriv = diff(expr, 'x')
print(deriv)  # 0
```

## The Jacobian

For vector-valued functions, `jacobian` computes all partial derivatives:

```python
from symlik import jacobian

# f(x,y) = (x*y, x+y)
exprs = [['*', 'x', 'y'], ['+', 'x', 'y']]
J = jacobian(exprs, ['x', 'y'])

print(f"df1/dx = {J[0][0]}")  # y
print(f"df1/dy = {J[0][1]}")  # x
print(f"df2/dx = {J[1][0]}")  # 1
print(f"df2/dy = {J[1][1]}")  # 1
```

## The Laplacian

The sum of second partial derivatives:

```python
from symlik import laplacian

# f(x,y) = x^2 + y^2
expr = ['+', ['^', 'x', 2], ['^', 'y', 2]]
lap = laplacian(expr, ['x', 'y'])
print(f"Laplacian: {lap}")  # 2 + 2 = 4
```

## Supported Functions

symlik differentiates these functions using standard rules:

| Function | Derivative |
|----------|------------|
| `['^', 'x', n]` | \(nx^{n-1}\) |
| `['exp', 'x']` | \(e^x\) |
| `['log', 'x']` | \(1/x\) |
| `['sin', 'x']` | \(\cos(x)\) |
| `['cos', 'x']` | \(-\sin(x)\) |
| `['tan', 'x']` | \(\sec^2(x)\) |
| `['sqrt', 'x']` | \(1/(2\sqrt{x})\) |

Chain rule is applied automatically:

```python
from symlik import diff

# d/dx(sin(x^2)) = cos(x^2) * 2x
expr = ['sin', ['^', 'x', 2]]
deriv = diff(expr, 'x')
print(deriv)
```

## Symbolic Integration

symlik also supports symbolic integration for basic functions:

```python
from symlik import integrate

# Integral of x^2 dx = x^3/3
expr = ['^', 'x', 2]
antideriv = integrate(expr, 'x')
print(antideriv)  # ['/', ['^', 'x', 3], 3]
```

Integration is less complete than differentiation. When symlik cannot find a closed form, it returns the unevaluated integral.

## Working with Summations

Differentiation distributes over summations:

```python
from symlik import diff

# d/dmu sum((x_i - mu)^2) = sum(-2(x_i - mu))
expr = ['sum', 'i', ['len', 'x'],
        ['^', ['-', ['@', 'x', 'i'], 'mu'], 2]]

deriv = diff(expr, 'mu')
# Derivative passes through the sum
```

Data operations (`@`, `len`, `total`) are treated as constants with respect to parameters:

```python
from symlik import diff

# d/dlambda (x[i]) = 0
expr = ['@', 'x', 'i']
deriv = diff(expr, 'lambda')
print(deriv)  # 0
```

## Example: Optimization

Use derivatives for gradient-based optimization:

```python
import numpy as np
from symlik import gradient_at

# Minimize f(x,y) = (x-1)^2 + (y-2)^2
expr = ['+', ['^', ['-', 'x', 1], 2], ['^', ['-', 'y', 2], 2]]

# Gradient descent
point = {'x': 0.0, 'y': 0.0}
learning_rate = 0.1

for _ in range(50):
    grad = gradient_at(expr, ['x', 'y'], point)
    point['x'] -= learning_rate * grad[0]
    point['y'] -= learning_rate * grad[1]

print(f"Minimum at ({point['x']:.3f}, {point['y']:.3f})")  # (1, 2)
```

## Performance Notes

Symbolic expressions are rewritten using term rewriting rules. For complex expressions:

- Derivatives are computed once and cached in `LikelihoodModel`
- Numerical evaluation is fast after symbolic simplification
- Very deep expressions may simplify slowly

For most statistical applications, performance is not a concern.
