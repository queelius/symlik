# Calculus Functions Reference

Symbolic and numerical calculus operations.

## Symbolic Differentiation

### diff(expr, var)

```python
from symlik import diff

derivative = diff(expr, var)
```

Compute the symbolic derivative.

**Arguments:**

- `expr` (ExprType): S-expression to differentiate
- `var` (str): Variable to differentiate with respect to

**Returns:** Symbolic derivative expression

**Example:**

```python
diff(['^', 'x', 3], 'x')  # ['*', 3, ['^', 'x', 2]]
diff(['sin', 'x'], 'x')   # ['cos', 'x']
diff(['exp', ['^', 'x', 2]], 'x')  # Chain rule applied
```

---

### gradient(expr, vars)

```python
from symlik import gradient

grad = gradient(expr, vars)
```

Compute the gradient (vector of partial derivatives).

**Arguments:**

- `expr` (ExprType): S-expression
- `vars` (List[str]): List of variables

**Returns:** List of symbolic partial derivatives

**Example:**

```python
expr = ['+', ['^', 'x', 2], ['*', 'x', 'y']]
grad = gradient(expr, ['x', 'y'])
# grad[0] = df/dx
# grad[1] = df/dy
```

---

### hessian(expr, vars)

```python
from symlik import hessian

H = hessian(expr, vars)
```

Compute the Hessian matrix (second partial derivatives).

**Arguments:**

- `expr` (ExprType): S-expression
- `vars` (List[str]): List of variables

**Returns:** 2D list of symbolic second derivatives

**Example:**

```python
expr = ['+', ['^', 'x', 2], ['*', 'x', 'y']]
H = hessian(expr, ['x', 'y'])
# H[i][j] = d2f/dxi dxj
```

---

### jacobian(exprs, vars)

```python
from symlik import jacobian

J = jacobian(exprs, vars)
```

Compute the Jacobian matrix for a vector-valued function.

**Arguments:**

- `exprs` (List[ExprType]): List of S-expressions [f1, f2, ...]
- `vars` (List[str]): List of variables

**Returns:** 2D list where J[i][j] = dfi/dxj

---

### laplacian(expr, vars)

```python
from symlik import laplacian

lap = laplacian(expr, vars)
```

Compute the Laplacian (sum of second partial derivatives).

**Arguments:**

- `expr` (ExprType): S-expression
- `vars` (List[str]): List of variables

**Returns:** Symbolic Laplacian expression

---

## Symbolic Integration

### integrate(expr, var)

```python
from symlik import integrate

antideriv = integrate(expr, var)
```

Compute the symbolic antiderivative.

**Arguments:**

- `expr` (ExprType): S-expression to integrate
- `var` (str): Integration variable

**Returns:** Symbolic antiderivative, or `['int', expr, var]` if no closed form

**Example:**

```python
integrate(['^', 'x', 2], 'x')  # ['/', ['^', 'x', 3], 3]
integrate(['exp', 'x'], 'x')   # ['exp', 'x']
```

---

## Simplification

### simplify(expr)

```python
from symlik import simplify

simplified = simplify(expr)
```

Simplify an expression algebraically.

**Arguments:**

- `expr` (ExprType): S-expression to simplify

**Returns:** Simplified expression

**Example:**

```python
simplify(['+', 'x', 0])      # 'x'
simplify(['*', 'x', 1])      # 'x'
simplify(['^', 'x', 0])      # 1
simplify(['*', 'x', 'x'])    # ['^', 'x', 2]
```

---

## Numerical Evaluation

### diff_at(expr, var, env)

```python
from symlik import diff_at

value = diff_at(expr, var, env)
```

Evaluate derivative at a point.

**Arguments:**

- `expr` (ExprType): S-expression
- `var` (str): Variable to differentiate
- `env` (Dict[str, Any]): Variable bindings

**Returns:** float

---

### gradient_at(expr, vars, env)

```python
from symlik import gradient_at

grad = gradient_at(expr, vars, env)
```

Evaluate gradient at a point.

**Arguments:**

- `expr` (ExprType): S-expression
- `vars` (List[str]): List of variables
- `env` (Dict[str, Any]): Variable bindings

**Returns:** numpy.ndarray

---

### hessian_at(expr, vars, env)

```python
from symlik import hessian_at

H = hessian_at(expr, vars, env)
```

Evaluate Hessian at a point.

**Arguments:**

- `expr` (ExprType): S-expression
- `vars` (List[str]): List of variables
- `env` (Dict[str, Any]): Variable bindings

**Returns:** numpy.ndarray (2D)

---

## Expression Evaluation

### evaluate(expr, env, ops=None)

```python
from symlik import evaluate

result = evaluate(expr, env)
```

Evaluate a symbolic expression numerically.

**Arguments:**

- `expr` (ExprType): S-expression to evaluate
- `env` (Dict[str, Any]): Variable bindings
- `ops` (Dict[str, Callable]): Custom operators. Default: STANDARD_OPS

**Returns:** float

**Example:**

```python
evaluate(['+', ['*', 2, 'x'], ['^', 'y', 2]], {'x': 3, 'y': 4})
# Returns: 2*3 + 4^2 = 22.0
```

---

## Standard Operators

The `STANDARD_OPS` dictionary defines built-in operators:

| Operator | Function |
|----------|----------|
| `+` | Addition (variadic) |
| `-` | Subtraction or negation |
| `*` | Multiplication (variadic) |
| `/` | Division |
| `^` | Power |
| `sin`, `cos`, `tan` | Trigonometric |
| `arcsin`, `arccos`, `arctan` | Inverse trig |
| `sinh`, `cosh`, `tanh` | Hyperbolic |
| `exp` | Exponential |
| `log`, `ln` | Natural logarithm |
| `sqrt` | Square root |
| `abs` | Absolute value |
| `lgamma`, `gamma` | Gamma functions |
| `erf`, `erfc` | Error functions |

---

## Special Forms

These have special evaluation semantics:

| Form | Meaning |
|------|---------|
| `['sum', 'i', n, body]` | \(\sum_{i=1}^{n} \text{body}\) |
| `['prod', 'i', n, body]` | \(\prod_{i=1}^{n} \text{body}\) |
| `['@', 'x', 'i']` | x[i] (1-based indexing) |
| `['len', 'x']` | Length of x |
| `['total', 'x']` | Sum of x |
| `['if', cond, then, else]` | Conditional |

**Constants:** `'e'` and `'pi'` evaluate to mathematical constants.
