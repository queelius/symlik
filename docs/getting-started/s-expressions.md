# S-Expression Syntax

symlik represents mathematical expressions as s-expressions, a notation from Lisp. This page teaches you the syntax so you can write custom models.

## Basic Structure

An s-expression is either:

- A **number**: `3`, `2.5`, `-1`
- A **symbol**: `'x'`, `'lambda'`, `'mu'`
- A **list**: `[operator, arg1, arg2, ...]`

For example, \(x + 1\) becomes:

```python
['+', 'x', 1]
```

And \(2x^2 + 3x\) becomes:

```python
['+', ['*', 2, ['^', 'x', 2]], ['*', 3, 'x']]
```

## Arithmetic Operators

| Expression | S-expression | Notes |
|------------|--------------|-------|
| \(x + y\) | `['+', 'x', 'y']` | Addition |
| \(x - y\) | `['-', 'x', 'y']` | Subtraction |
| \(-x\) | `['-', 'x']` | Negation |
| \(x \cdot y\) | `['*', 'x', 'y']` | Multiplication |
| \(x / y\) | `['/', 'x', 'y']` | Division |
| \(x^n\) | `['^', 'x', 'n']` | Power |

## Mathematical Functions

| Function | S-expression |
|----------|--------------|
| \(\ln(x)\) | `['log', 'x']` |
| \(e^x\) | `['exp', 'x']` |
| \(\sqrt{x}\) | `['sqrt', 'x']` |
| \(\sin(x)\) | `['sin', 'x']` |
| \(\cos(x)\) | `['cos', 'x']` |
| \(\log \Gamma(x)\) | `['lgamma', 'x']` |

## Working with Data

Statistical models need to reference data. symlik provides special operators:

| Operator | Meaning | Example |
|----------|---------|---------|
| `['len', 'x']` | Length of x | Sample size |
| `['@', 'x', 'i']` | x[i] (1-based) | i-th observation |
| `['total', 'x']` | Sum of x | \(\sum x_i\) |

**Note:** Indexing is 1-based, so `['@', 'x', 1]` is the first element.

## Summation

The `sum` operator iterates over indices:

```python
['sum', 'i', n, body]
```

This computes \(\sum_{i=1}^{n} \text{body}\) where `i` takes values 1 through n.

**Example:** Sum of squared deviations from mean \(\mu\):

```python
['sum', 'i', ['len', 'x'],
 ['^', ['-', ['@', 'x', 'i'], 'mu'], 2]]
```

This represents \(\sum_{i=1}^{n} (x_i - \mu)^2\).

## Building a Log-Likelihood

Let's build the exponential log-likelihood step by step.

The likelihood for one observation is \(L_i = \lambda e^{-\lambda x_i}\), so the log-likelihood contribution is:

\[ \ell_i = \log(\lambda) - \lambda x_i \]

In s-expression form:

```python
['+', ['log', 'lambda'], ['*', -1, ['*', 'lambda', ['@', 'x', 'i']]]]
```

Summing over all observations:

```python
log_lik = ['sum', 'i', ['len', 'x'],
           ['+', ['log', 'lambda'],
            ['*', -1, ['*', 'lambda', ['@', 'x', 'i']]]]]
```

## Nesting and Readability

Complex expressions can get hard to read. Use Python formatting:

```python
# Normal log-likelihood (more complex)
n = ['len', 'x']
sum_sq = ['sum', 'i', n,
          ['^', ['-', ['@', 'x', 'i'], 'mu'], 2]]

log_lik = ['+',
           ['*', -0.5, ['*', n, ['log', ['*', 2, 3.14159]]]],
           ['+',
            ['*', -0.5, ['*', n, ['log', 'sigma2']]],
            ['*', -0.5, ['/', sum_sq, 'sigma2']]]]
```

## Evaluating Expressions

Test your expressions with `evaluate`:

```python
from symlik import evaluate

expr = ['+', ['*', 2, 'x'], ['^', 'y', 2]]
result = evaluate(expr, {'x': 3, 'y': 4})
print(result)  # 2*3 + 4^2 = 22
```

## Common Patterns

**Dot product** \(\sum x_i y_i\):
```python
['sum', 'i', ['len', 'x'],
 ['*', ['@', 'x', 'i'], ['@', 'y', 'i']]]
```

**Sample mean** \(\bar{x} = \sum x_i / n\):
```python
['/', ['total', 'x'], ['len', 'x']]
```

**Log of product** \(\log \prod f_i = \sum \log f_i\):
```python
['sum', 'i', n, ['log', body]]
```

## Next Steps

Now that you know the syntax, try [building custom models](../tutorials/custom-models.md).
