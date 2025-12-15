"""
Expression evaluation with numerical computation.

This module evaluates symbolic expressions numerically, handling:
- Standard mathematical operators via extensible prelude
- Special forms (sum, prod, dd, int, @) with hardcoded semantics
- Irreducible symbolic forms via numerical methods
"""

import math
import operator
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Union

ExprType = Union[int, float, str, List]

# ============================================================
# Standard Operators Prelude (user-extensible)
# ============================================================

STANDARD_OPS: Dict[str, Callable[[List[float]], float]] = {
    # Arithmetic
    "+": lambda args: sum(args),
    "-": lambda args: -args[0] if len(args) == 1 else args[0] - args[1],
    "*": lambda args: reduce(operator.mul, args, 1),
    "/": lambda args: args[0] / args[1] if args[1] != 0 else float('inf'),
    "^": lambda args: args[0] ** args[1],

    # Trigonometric
    "sin": lambda args: math.sin(args[0]),
    "cos": lambda args: math.cos(args[0]),
    "tan": lambda args: math.tan(args[0]),
    "arcsin": lambda args: math.asin(args[0]),
    "arccos": lambda args: math.acos(args[0]),
    "arctan": lambda args: math.atan(args[0]),

    # Hyperbolic
    "sinh": lambda args: math.sinh(args[0]),
    "cosh": lambda args: math.cosh(args[0]),
    "tanh": lambda args: math.tanh(args[0]),

    # Exponential/Logarithmic
    "exp": lambda args: math.exp(args[0]),
    "log": lambda args: math.log(args[0]) if args[0] > 0 else float('-inf'),
    "ln": lambda args: math.log(args[0]) if args[0] > 0 else float('-inf'),
    "sqrt": lambda args: math.sqrt(args[0]) if args[0] >= 0 else float('nan'),

    # Other
    "abs": lambda args: abs(args[0]),
    "lgamma": lambda args: math.lgamma(args[0]),
    "gamma": lambda args: math.gamma(args[0]),
    "erf": lambda args: math.erf(args[0]),
    "erfc": lambda args: math.erfc(args[0]),
}


# ============================================================
# Core Evaluation Function
# ============================================================

def evaluate(
    expr: ExprType,
    env: Dict[str, Any],
    ops: Optional[Dict[str, Callable]] = None,
) -> float:
    """
    Evaluate a symbolic expression numerically.

    Args:
        expr: S-expression to evaluate
        env: Environment mapping variable names to values
        ops: Optional custom operators (defaults to STANDARD_OPS)

    Returns:
        Numerical result

    Examples:
        >>> evaluate(['+', 'x', 1], {'x': 2})
        3.0
        >>> evaluate(['sin', ['^', 'x', 2]], {'x': 1.0})
        0.8414...
        >>> evaluate(['sum', 'i', 3, ['@', 'x', 'i']], {'x': [10, 20, 30]})
        60.0
    """
    if ops is None:
        ops = STANDARD_OPS

    # Constants
    if isinstance(expr, (int, float)):
        return float(expr)

    # Variables
    if isinstance(expr, str):
        if expr == "e":
            return math.e
        if expr == "pi":
            return math.pi
        if expr in env:
            return float(env[expr]) if isinstance(env[expr], (int, float)) else env[expr]
        raise ValueError(f"Unbound variable: {expr}")

    # Compound expressions
    if not isinstance(expr, list) or len(expr) == 0:
        raise ValueError(f"Invalid expression: {expr}")

    op = expr[0]
    args = expr[1:]

    # ============================================================
    # Special Forms (hardcoded - control evaluation)
    # ============================================================

    # Summation: ['sum', idx_var, n, body]
    if op == "sum":
        idx_var, n_expr, body = args
        n = int(evaluate(n_expr, env, ops))
        total = 0.0
        for i in range(1, n + 1):
            local_env = {**env, idx_var: i}
            total += evaluate(body, local_env, ops)
        return total

    # Product: ['prod', idx_var, n, body]
    if op == "prod":
        idx_var, n_expr, body = args
        n = int(evaluate(n_expr, env, ops))
        result = 1.0
        for i in range(1, n + 1):
            local_env = {**env, idx_var: i}
            result *= evaluate(body, local_env, ops)
        return result

    # Indexing: ['@', data, idx] (1-based)
    if op == "@":
        data_expr, idx_expr = args
        # Data can be a variable name or nested expression
        if isinstance(data_expr, str):
            data = env[data_expr]
        else:
            data = evaluate(data_expr, env, ops)
        idx = int(evaluate(idx_expr, env, ops))
        return float(data[idx - 1])  # 1-based indexing

    # Index function (alternative syntax): ['index', data, idx]
    if op == "index":
        data_expr, idx_expr = args
        if isinstance(data_expr, str):
            data = env[data_expr]
        else:
            data = evaluate(data_expr, env, ops)
        idx = int(evaluate(idx_expr, env, ops))
        return float(data[idx - 1])

    # Length: ['len', data]
    if op == "len":
        data_expr = args[0]
        if isinstance(data_expr, str):
            data = env[data_expr]
        else:
            data = evaluate(data_expr, env, ops)
        return float(len(data))

    # Total (sum of data): ['total', data]
    if op == "total":
        data_expr = args[0]
        if isinstance(data_expr, str):
            data = env[data_expr]
        else:
            data = evaluate(data_expr, env, ops)
        return float(sum(data))

    # Derivative (numerical): ['dd', inner_expr, var]
    if op == "dd":
        inner_expr, var = args
        return _finite_difference(inner_expr, var, env, ops)

    # Integral (numerical): ['int', inner_expr, var] or ['int', inner_expr, var, [a, b]]
    if op == "int":
        if len(args) == 2:
            inner_expr, var = args
            # Indefinite integral - evaluate from 0 to current value
            if var in env:
                return _numerical_integrate(inner_expr, var, 0.0, env[var], env, ops)
            raise ValueError(f"Cannot evaluate indefinite integral without {var} in env")
        elif len(args) == 3:
            inner_expr, var, bounds = args
            a = evaluate(bounds[0], env, ops) if isinstance(bounds[0], list) else float(bounds[0])
            b = evaluate(bounds[1], env, ops) if isinstance(bounds[1], list) else float(bounds[1])
            return _numerical_integrate(inner_expr, var, a, b, env, ops)

    # Conditional: ['if', cond, then, else]
    if op == "if":
        cond, then_expr, else_expr = args
        cond_val = evaluate(cond, env, ops)
        return evaluate(then_expr if cond_val else else_expr, env, ops)

    # ============================================================
    # Strict Operators (evaluate args first, then apply)
    # ============================================================

    evaluated_args = [evaluate(arg, env, ops) for arg in args]

    if op in ops:
        try:
            return ops[op](evaluated_args)
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            return float('nan')

    raise ValueError(f"Unknown operator: {op}")


# ============================================================
# Numerical Methods for Irreducible Forms
# ============================================================

def _finite_difference(
    expr: ExprType,
    var: str,
    env: Dict[str, Any],
    ops: Dict[str, Callable],
    h: float = 1e-7,
) -> float:
    """Compute derivative using central finite difference."""
    x = env[var]

    env_plus = {**env, var: x + h}
    env_minus = {**env, var: x - h}

    f_plus = evaluate(expr, env_plus, ops)
    f_minus = evaluate(expr, env_minus, ops)

    return (f_plus - f_minus) / (2 * h)


def _numerical_integrate(
    expr: ExprType,
    var: str,
    a: float,
    b: float,
    env: Dict[str, Any],
    ops: Dict[str, Callable],
    n_points: int = 100,
) -> float:
    """Compute definite integral using Simpson's rule."""
    if a == b:
        return 0.0

    h = (b - a) / n_points

    def f(x):
        local_env = {**env, var: x}
        return evaluate(expr, local_env, ops)

    # Simpson's rule
    result = f(a) + f(b)
    for i in range(1, n_points):
        x = a + i * h
        if i % 2 == 0:
            result += 2 * f(x)
        else:
            result += 4 * f(x)

    return result * h / 3


# ============================================================
# Convenience Functions
# ============================================================

def expr_to_function(expr: ExprType, var: str, ops: Optional[Dict] = None):
    """
    Convert an expression to a callable function of one variable.

    Args:
        expr: S-expression
        var: Variable name
        ops: Optional custom operators

    Returns:
        Callable that takes a float and returns a float
    """
    def f(x: float) -> float:
        return evaluate(expr, {var: x}, ops)
    return f


def expr_to_multivar_function(expr: ExprType, vars: List[str], ops: Optional[Dict] = None):
    """
    Convert an expression to a callable function of multiple variables.

    Args:
        expr: S-expression
        vars: List of variable names
        ops: Optional custom operators

    Returns:
        Callable that takes an array and returns a float
    """
    def f(x) -> float:
        env = {v: float(x[i]) for i, v in enumerate(vars)}
        return evaluate(expr, env, ops)
    return f
