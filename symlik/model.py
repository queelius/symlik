"""
Likelihood Model for symbolic statistical inference.

The core abstraction: a LikelihoodModel combines a symbolic log-likelihood
expression with automatic differentiation for statistical inference.
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

from .rules import get_default_engine
from .evaluate import evaluate, ExprType
from .utils import to_data_dict

if TYPE_CHECKING:
    from .fitted import FittedLikelihoodModel


class LikelihoodModel:
    """
    A symbolic likelihood model with automatic differentiation.

    Provides:
    - Symbolic score (gradient of log-likelihood)
    - Symbolic Hessian and information matrix
    - MLE estimation via fit()
    - Full inference through FittedLikelihoodModel

    Example:
        >>> from symlik.distributions import exponential
        >>> model = exponential()
        >>> fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})
        >>> fit.params
        {'lambda': 0.333...}
        >>> fit.se
        {'lambda': 0.149...}
        >>> fit.aic
        12.34
        >>> print(fit.summary())

    For custom models:
        >>> # Exponential log-likelihood: ℓ(λ) = Σᵢ [log(λ) - λxᵢ]
        >>> model = LikelihoodModel(
        ...     log_lik=['sum', 'i', ['len', 'x'],
        ...              ['+', ['log', 'lambda'],
        ...               ['*', -1, ['*', 'lambda', ['@', 'x', 'i']]]]],
        ...     params=['lambda']
        ... )
        >>> fit = model.fit({'x': [1, 2, 3]}, init={'lambda': 1.0})
    """

    def __init__(self, log_lik: ExprType, params: List[str]):
        """
        Initialize a likelihood model.

        Args:
            log_lik: Log-likelihood as an S-expression
            params: List of parameter names to estimate
        """
        self.log_lik = log_lik
        self.params = params

        # Symbolic simplifier using rerum
        self._engine = get_default_engine()

        # Cache for computed derivatives
        self._score_cache: Optional[List[ExprType]] = None
        self._hessian_cache: Optional[List[List[ExprType]]] = None

    def _simplify(self, expr: ExprType) -> ExprType:
        """Simplify an expression using the rewriting engine."""
        return self._engine.simplify(expr)

    def score(self) -> List[ExprType]:
        """
        Compute the score vector (gradient of log-likelihood).

        U(θ) = ∂ℓ/∂θ

        Returns:
            List of symbolic partial derivatives [∂ℓ/∂θ₁, ∂ℓ/∂θ₂, ...]
        """
        if self._score_cache is None:
            self._score_cache = []
            for param in self.params:
                deriv = self._simplify(["dd", self.log_lik, param])
                self._score_cache.append(deriv)
        return self._score_cache

    def hessian(self) -> List[List[ExprType]]:
        """
        Compute the Hessian matrix of log-likelihood.

        H(θ) = ∂²ℓ/∂θᵢ∂θⱼ

        Returns:
            2D list of symbolic second derivatives
        """
        if self._hessian_cache is None:
            score = self.score()
            n = len(self.params)
            self._hessian_cache = []
            for i in range(n):
                row = []
                for j in range(n):
                    second_deriv = self._simplify(["dd", score[i], self.params[j]])
                    row.append(second_deriv)
                self._hessian_cache.append(row)
        return self._hessian_cache

    def information(self) -> List[List[ExprType]]:
        """
        Compute observed Fisher information (negative Hessian).

        I(θ) = -∂²ℓ/∂θᵢ∂θⱼ

        Returns:
            2D list representing -H(θ)
        """
        hess = self.hessian()
        n = len(self.params)
        return [[self._simplify(["*", -1, hess[i][j]]) for j in range(n)] for i in range(n)]

    def evaluate(self, env: Dict[str, Any]) -> float:
        """
        Evaluate log-likelihood at given parameter/data values.

        Args:
            env: Dictionary mapping variables to values

        Returns:
            Numerical log-likelihood value
        """
        return evaluate(self.log_lik, env)

    def score_at(self, env: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate score vector at given parameter/data values.

        Args:
            env: Dictionary mapping variables to values

        Returns:
            Array of score values
        """
        return np.array([evaluate(s, env) for s in self.score()])

    def hessian_at(self, env: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate Hessian matrix at given parameter/data values.

        Args:
            env: Dictionary mapping variables to values

        Returns:
            2D array of Hessian values
        """
        hess = self.hessian()
        n = len(self.params)
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                H[i, j] = evaluate(hess[i][j], env)
        return H

    def information_at(self, env: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate observed information at given parameter/data values.

        Args:
            env: Dictionary mapping variables to values

        Returns:
            2D array (information matrix)
        """
        return -self.hessian_at(env)

    def fit(
        self,
        data: Any,
        init: Dict[str, float],
        max_iter: int = 100,
        tol: float = 1e-8,
        bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    ) -> 'FittedLikelihoodModel':
        """
        Fit the model to data.

        Uses Newton-Raphson optimization to find maximum likelihood estimates,
        then returns a FittedLikelihoodModel with full inference capabilities.

        Args:
            data: Data values as dict, pandas DataFrame, or polars DataFrame
            init: Initial parameter guesses
            max_iter: Maximum iterations (default 100)
            tol: Convergence tolerance for score norm (default 1e-8)
            bounds: Optional parameter bounds {param: (min, max)}

        Returns:
            FittedLikelihoodModel with estimation results

        Example:
            >>> model = exponential()
            >>> fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})
            >>> fit.params        # MLE estimates
            >>> fit.se            # Standard errors
            >>> fit.conf_int()    # Confidence intervals
            >>> fit.summary()     # Full summary table
        """
        from .fitted import FittedLikelihoodModel

        data_dict = to_data_dict(data)
        params, n_iter = self._optimize(data_dict, init, max_iter, tol, bounds)

        # Check convergence
        env = dict(data_dict)
        env.update(params)
        try:
            score = self.score_at(env)
            converged = bool(np.linalg.norm(score) < tol)
        except Exception:
            converged = False

        return FittedLikelihoodModel(self, data_dict, params, n_iter, converged)

    def _optimize(
        self,
        data_dict: Dict[str, Any],
        init: Dict[str, float],
        max_iter: int,
        tol: float,
        bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]],
    ) -> Tuple[Dict[str, float], int]:
        """
        Internal Newton-Raphson optimization.

        Returns:
            Tuple of (parameter estimates dict, number of iterations)
        """
        theta = np.array([init[p] for p in self.params])

        for iteration in range(max_iter):
            # Build environment
            env = dict(data_dict)
            for i, p in enumerate(self.params):
                env[p] = float(theta[i])

            # Evaluate score and Hessian
            try:
                score = self.score_at(env)
                hess = self.hessian_at(env)
            except (OverflowError, FloatingPointError, ValueError):
                # Parameters out of range - try gradient ascent
                try:
                    score = self.score_at(env)
                    theta = theta + 0.01 * score
                except Exception:
                    break
                continue

            # Check for non-finite values
            if not np.all(np.isfinite(score)) or not np.all(np.isfinite(hess)):
                break

            # Check convergence
            if np.linalg.norm(score) < tol:
                break

            # Newton step: θ_new = θ - H⁻¹ · score
            try:
                step = np.linalg.solve(hess, score)
                # Damped step for stability
                step_size = min(1.0, 1.0 / (1.0 + np.linalg.norm(step)))
                theta_new = theta - step_size * step

                # Apply bounds
                if bounds:
                    for i, p in enumerate(self.params):
                        if p in bounds:
                            lo, hi = bounds[p]
                            if lo is not None:
                                theta_new[i] = max(theta_new[i], lo)
                            if hi is not None:
                                theta_new[i] = min(theta_new[i], hi)

                theta = theta_new
            except np.linalg.LinAlgError:
                # Hessian singular - use gradient ascent
                theta = theta + 0.01 * score

        return {p: float(theta[i]) for i, p in enumerate(self.params)}, iteration + 1

    def __repr__(self) -> str:
        return f"LikelihoodModel(params={self.params})"
