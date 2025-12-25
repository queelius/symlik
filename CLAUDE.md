# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**symlik** is a symbolic likelihood models library for statistical inference. It combines symbolic differentiation (via the `rerum` package) with numerical evaluation for maximum likelihood estimation.

## Commands

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/ -v

# Run single test
pytest tests/test_evaluate.py::TestSummation::test_sum_simple -v

# Run tests with coverage
pytest tests/ --cov=symlik --cov-report=term-missing

# Type checking
mypy symlik/

# Formatting
black symlik/ tests/
```

## Architecture

### Two-Layer Design

The core insight is clean separation between **symbolic rewriting** and **numerical evaluation**:

1. **Symbolic layer** (`rerum` package): Pattern matching and rule-based term rewriting. Handles differentiation, algebraic simplification. Rules are stored in `.rerum` DSL files.

2. **Numerical layer** (`symlik.evaluate`): Evaluates expressions numerically. Has two types of operations:
   - **Special forms** (hardcoded): `sum`, `prod`, `@`, `len`, `total`, `dd`, `int`, `if` - these control evaluation flow
   - **Strict operators** (extensible via `STANDARD_OPS`): `+`, `-`, `*`, `/`, `sin`, `cos`, `exp`, `log`, etc.

Irreducible symbolic forms (like `['dd', expr, var]` that can't be simplified) fall back to numerical methods at evaluation time (finite differences for derivatives, Simpson's rule for integrals).

### Expression Format

Expressions are S-expressions represented as Python lists:
```python
['*', 2, ['sin', 'x']]           # 2 * sin(x)
['sum', 'i', ['len', 'x'], 'i']  # Σᵢ i for i=1 to len(x)
['@', 'x', 'i']                  # x[i] (1-based indexing)
['dd', ['sin', 'x'], 'x']        # d/dx(sin(x))
```

### Key Components

- **`LikelihoodModel`** (`model.py`): Core abstraction combining symbolic log-likelihood with automatic differentiation. Provides `score()` (gradient), `hessian()`, `mle()`, and `se()` (standard errors).

- **`ContributionModel`** (`contribution.py`): For heterogeneous data with different observation types (e.g., complete vs censored). Dispatches on a type column, builds composite log-likelihood by summing over each type's contribution.

- **`symlik.distributions`**: Pre-built distribution constructors (exponential, normal, poisson, bernoulli, gamma, weibull, beta) that return configured `LikelihoodModel` instances.

- **`symlik.series`**: Series system reliability modeling. Provides `ComponentHazard` building blocks and factories like `build_exponential_series_contributions()` for multi-component systems with known cause, masked cause, and censored observations.

- **`symlik.contributions`**: Pre-built contribution functions for common cases (`complete_exponential`, `right_censored_exponential`, etc.).

- **`symlik.rules/`**: Contains `.rerum` rule files loaded by `rerum.RuleEngine`:
  - `derivative.rerum`: Differentiation rules (power rule, chain rule, etc.)
  - `algebra.rerum`: Algebraic simplification (identity, zero, combining terms)
  - `integral.rerum`: Basic integration rules

- **`symlik.calculus`**: Thin wrappers (`diff`, `gradient`, `hessian`, `integrate`) that construct symbolic expressions and invoke the rewriting engine.

### Rule DSL Format

Rules in `.rerum` files use pattern matching syntax:
```
@rule-name "description": (pattern) => (skeleton)
```

Pattern syntax: `?x` binds variable, `?x:const` matches constants, `?x:var` matches symbols
Skeleton syntax: `:x` substitutes bound variable

Example:
```
@dd-power-var "Power rule": (dd (^ ?x:var ?n:const) ?x) => (* :n (^ :x (- :n 1)))
```

## Dependencies

- `rerum`: Symbolic rewriting engine (pattern matching + rule application)
- `numpy`: Numerical computation for MLE optimization
