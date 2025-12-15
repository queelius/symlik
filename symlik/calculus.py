"""
Symbolic calculus operations.

Thin wrappers around rerum's rewriting engine for differentiation,
integration, and algebraic simplification.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from .rules import get_default_engine, create_full_engine
from .evaluate import evaluate, ExprType


def simplify(expr: ExprType) -> ExprType:
    """
    Simplify an expression algebraically.

    Args:
        expr: S-expression to simplify

    Returns:
        Simplified expression

    Example:
        >>> simplify(['+', ['*', 'x', 0], ['*', 'y', 1]])
        'y'
    """
    return get_default_engine().simplify(expr)


def diff(expr: ExprType, var: str) -> ExprType:
    """
    Compute the symbolic derivative.

    Args:
        expr: S-expression to differentiate
        var: Variable to differentiate with respect to

    Returns:
        Symbolic derivative, or ['dd', expr, var] if irreducible

    Example:
        >>> diff(['^', 'x', 2], 'x')
        ['*', 2, 'x']
        >>> diff(['sin', 'x'], 'x')
        ['cos', 'x']
    """
    return simplify(["dd", expr, var])


def integrate(expr: ExprType, var: str) -> ExprType:
    """
    Compute the symbolic antiderivative.

    Args:
        expr: S-expression to integrate
        var: Integration variable

    Returns:
        Symbolic antiderivative, or ['int', expr, var] if irreducible

    Example:
        >>> integrate(['^', 'x', 2], 'x')
        ['/', ['^', 'x', 3], 3]
    """
    engine = create_full_engine()
    return engine.simplify(["int", expr, var])


def gradient(expr: ExprType, vars: List[str]) -> List[ExprType]:
    """
    Compute the gradient (vector of partial derivatives).

    Args:
        expr: S-expression
        vars: List of variables

    Returns:
        List of partial derivatives [∂f/∂x₁, ∂f/∂x₂, ...]

    Example:
        >>> gradient(['+', ['^', 'x', 2], ['^', 'y', 2]], ['x', 'y'])
        [['*', 2, 'x'], ['*', 2, 'y']]
    """
    return [diff(expr, v) for v in vars]


def hessian(expr: ExprType, vars: List[str]) -> List[List[ExprType]]:
    """
    Compute the Hessian matrix (matrix of second partial derivatives).

    Args:
        expr: S-expression
        vars: List of variables

    Returns:
        2D list [[∂²f/∂xᵢ∂xⱼ]]

    Example:
        >>> hessian(['+', ['^', 'x', 2], ['*', 'x', 'y']], ['x', 'y'])
        [[2, 1], [1, 0]]
    """
    grad = gradient(expr, vars)
    return [[diff(g, v) for v in vars] for g in grad]


def jacobian(exprs: List[ExprType], vars: List[str]) -> List[List[ExprType]]:
    """
    Compute the Jacobian matrix for a vector-valued function.

    Args:
        exprs: List of S-expressions [f₁, f₂, ...]
        vars: List of variables [x₁, x₂, ...]

    Returns:
        2D list [[∂fᵢ/∂xⱼ]]

    Example:
        >>> jacobian([['*', 'x', 'y'], ['+', 'x', 'y']], ['x', 'y'])
        [['y', 'x'], [1, 1]]
    """
    return [gradient(f, vars) for f in exprs]


def laplacian(expr: ExprType, vars: List[str]) -> ExprType:
    """
    Compute the Laplacian (sum of second partial derivatives).

    ∇²f = Σᵢ ∂²f/∂xᵢ²

    Args:
        expr: S-expression
        vars: List of variables

    Returns:
        Symbolic Laplacian

    Example:
        >>> laplacian(['+', ['^', 'x', 2], ['^', 'y', 2]], ['x', 'y'])
        4
    """
    hess = hessian(expr, vars)
    diagonal = [hess[i][i] for i in range(len(vars))]

    if len(diagonal) == 0:
        return 0
    if len(diagonal) == 1:
        return diagonal[0]

    result = ["+", diagonal[0], diagonal[1]]
    for d in diagonal[2:]:
        result = ["+", result, d]

    return simplify(result)


# ============================================================
# Evaluation helpers (combining symbolic + numerical)
# ============================================================

def diff_at(expr: ExprType, var: str, env: Dict[str, Any]) -> float:
    """
    Evaluate derivative at a point.

    Args:
        expr: S-expression
        var: Variable to differentiate
        env: Environment with variable values

    Returns:
        Numerical derivative value
    """
    deriv = diff(expr, var)
    return evaluate(deriv, env)


def gradient_at(expr: ExprType, vars: List[str], env: Dict[str, Any]) -> np.ndarray:
    """
    Evaluate gradient at a point.

    Args:
        expr: S-expression
        vars: List of variables
        env: Environment with variable values

    Returns:
        Array of gradient values
    """
    grad = gradient(expr, vars)
    return np.array([evaluate(g, env) for g in grad])


def hessian_at(expr: ExprType, vars: List[str], env: Dict[str, Any]) -> np.ndarray:
    """
    Evaluate Hessian at a point.

    Args:
        expr: S-expression
        vars: List of variables
        env: Environment with variable values

    Returns:
        2D array of Hessian values
    """
    hess = hessian(expr, vars)
    n = len(vars)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = evaluate(hess[i][j], env)
    return H
