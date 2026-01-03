"""
Contribution Model for heterogeneous likelihood contributions.

Supports observation types with different likelihood forms, such as:
- Complete observations vs censored observations
- Masked cause failures vs known cause failures
- Different censoring mechanisms in survival analysis
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

from .model import LikelihoodModel
from .evaluate import ExprType, evaluate
from .utils import to_data_dict

if TYPE_CHECKING:
    from .fitted import FittedLikelihoodModel


class ContributionModel:
    """
    A likelihood model with type-dispatched contributions.

    Each observation can have a different likelihood contribution based on its type.
    All contribution types share the same parameters, but contribute differently
    to the total log-likelihood.

    Example:
        >>> # Mixed complete and right-censored exponential data
        >>> model = ContributionModel(
        ...     params=["lambda"],
        ...     type_column="obs_type",
        ...     contributions={
        ...         "complete": ["+", ["log", "lambda"], ["*", -1, ["*", "lambda", "t"]]],
        ...         "censored": ["*", -1, ["*", "lambda", "t"]],
        ...     }
        ... )
        >>> data = {
        ...     "obs_type": ["complete", "censored", "complete"],
        ...     "t": [1.0, 2.0, 0.5],
        ... }
        >>> fit = model.fit(data=data, init={"lambda": 1.0})
        >>> fit.params
        >>> fit.summary()
    """

    def __init__(
        self,
        params: List[str],
        type_column: str,
        contributions: Dict[str, ExprType],
    ):
        """
        Initialize a contribution model.

        Args:
            params: Parameter names shared across all contribution types
            type_column: Name of the column containing observation types
            contributions: Dict mapping type names to log-likelihood expressions.
                          Each expression should be for a single observation,
                          referencing data columns by name.
        """
        self.params = params
        self.type_column = type_column
        self.contributions = contributions

        # Discover data columns from contribution expressions
        self._data_columns = self._discover_columns()

        # Build the composite log-likelihood and internal model
        self._composite_loglik = self._build_composite_loglik()
        self._model = LikelihoodModel(self._composite_loglik, params)

    def _discover_columns(self) -> List[str]:
        """Discover data column names referenced in contribution expressions."""
        columns = set()
        for contrib in self.contributions.values():
            columns.update(self._find_variables(contrib))
        # Remove parameters - they're not data columns
        columns -= set(self.params)
        # Remove built-in constants
        columns -= {"e", "pi"}
        return sorted(columns)

    def _find_variables(self, expr: ExprType) -> set:
        """Recursively find all variable names in an expression."""
        if isinstance(expr, str):
            return {expr}
        if isinstance(expr, (int, float)):
            return set()
        if isinstance(expr, list) and len(expr) > 0:
            result = set()
            for item in expr[1:]:  # Skip operator
                result.update(self._find_variables(item))
            return result
        return set()

    def _rewrite_columns(self, expr: ExprType, type_name: str) -> ExprType:
        """
        Rewrite column references to use type-specific indexed access.

        Transforms: "t" -> ["@", "t_<type_name>", "i"]
        """
        if isinstance(expr, str):
            if expr in self._data_columns:
                return ["@", f"{expr}_{type_name}", "i"]
            return expr
        if isinstance(expr, (int, float)):
            return expr
        if isinstance(expr, list) and len(expr) > 0:
            return [expr[0]] + [self._rewrite_columns(arg, type_name) for arg in expr[1:]]
        return expr

    def _build_composite_loglik(self) -> ExprType:
        """
        Build composite log-likelihood that sums over each type's contributions.

        Creates: sum over type1 + sum over type2 + ...
        Where each sum iterates over observations of that type.
        """
        terms = []

        for type_name, contrib_expr in self.contributions.items():
            # Rewrite column references to use type-specific arrays
            rewritten = self._rewrite_columns(contrib_expr, type_name)

            # Use first data column to determine count for this type
            if self._data_columns:
                count_col = f"{self._data_columns[0]}_{type_name}"
            else:
                # No data columns - just parameters, shouldn't happen in practice
                count_col = f"_count_{type_name}"

            # Sum over all observations of this type
            term = ["sum", "i", ["len", count_col], rewritten]
            terms.append(term)

        # Combine all type contributions
        if len(terms) == 0:
            return 0
        elif len(terms) == 1:
            return terms[0]
        else:
            return ["+"] + terms

    def _prepare_data(self, data: Any) -> Dict[str, Any]:
        """
        Split data by observation type.

        Accepts dict, pandas DataFrame, or polars DataFrame.

        Transforms:
            {"obs_type": ["A", "B", "A"], "t": [1, 2, 3]}
        Into:
            {"t_A": [1, 3], "t_B": [2]}
        """
        # Convert DataFrame-like objects to dict
        data = to_data_dict(data)

        if self.type_column not in data:
            raise ValueError(f"Type column '{self.type_column}' not found in data")

        types = data[self.type_column]
        n = len(types)

        # Validate all types are known
        unknown_types = set(types) - set(self.contributions.keys())
        if unknown_types:
            raise ValueError(f"Unknown observation types: {unknown_types}")

        result = {}

        # For each type, filter the data columns
        for type_name in self.contributions:
            mask = [t == type_name for t in types]

            for col in self._data_columns:
                if col in data:
                    values = data[col]
                    if isinstance(values, list) and len(values) == n:
                        result[f"{col}_{type_name}"] = [
                            v for v, m in zip(values, mask) if m
                        ]

        return result

    def score(self) -> List[ExprType]:
        """
        Compute the score vector (gradient of log-likelihood).

        Returns:
            List of symbolic partial derivatives
        """
        return self._model.score()

    def hessian(self) -> List[List[ExprType]]:
        """
        Compute the Hessian matrix of log-likelihood.

        Returns:
            2D list of symbolic second derivatives
        """
        return self._model.hessian()

    def information(self) -> List[List[ExprType]]:
        """
        Compute observed Fisher information (negative Hessian).

        Returns:
            2D list representing -H(theta)
        """
        return self._model.information()

    def evaluate(self, data_and_params: Dict[str, Any]) -> float:
        """
        Evaluate log-likelihood at given parameter/data values.

        Args:
            data_and_params: Dictionary with both data and parameter values.
                            Data should include the type column and all data columns.

        Returns:
            Numerical log-likelihood value
        """
        # Separate data from params
        data = {k: v for k, v in data_and_params.items() if k not in self.params}
        params = {k: v for k, v in data_and_params.items() if k in self.params}

        # Prepare type-split data
        prepared = self._prepare_data(data)
        prepared.update(params)

        return self._model.evaluate(prepared)

    def score_at(self, data_and_params: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate score vector at given parameter/data values.

        Args:
            data_and_params: Dictionary with data and parameter values

        Returns:
            Array of score values
        """
        data = {k: v for k, v in data_and_params.items() if k not in self.params}
        params = {k: v for k, v in data_and_params.items() if k in self.params}

        prepared = self._prepare_data(data)
        prepared.update(params)

        return self._model.score_at(prepared)

    def hessian_at(self, data_and_params: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate Hessian matrix at given parameter/data values.

        Args:
            data_and_params: Dictionary with data and parameter values

        Returns:
            2D array of Hessian values
        """
        data = {k: v for k, v in data_and_params.items() if k not in self.params}
        params = {k: v for k, v in data_and_params.items() if k in self.params}

        prepared = self._prepare_data(data)
        prepared.update(params)

        return self._model.hessian_at(prepared)

    def information_at(self, data_and_params: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate observed information at given parameter/data values.

        Args:
            data_and_params: Dictionary with data and parameter values

        Returns:
            2D array (information matrix)
        """
        return -self.hessian_at(data_and_params)

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
            data: Data values as dict, pandas DataFrame, or polars DataFrame.
                  Must include the type column and all data columns.
            init: Initial parameter guesses
            max_iter: Maximum iterations (default 100)
            tol: Convergence tolerance for score norm (default 1e-8)
            bounds: Optional parameter bounds {param: (min, max)}

        Returns:
            FittedLikelihoodModel with estimation results

        Example:
            >>> model = ContributionModel(...)
            >>> fit = model.fit(data, init={'lambda': 1.0})
            >>> fit.params        # MLE estimates
            >>> fit.se            # Standard errors
            >>> fit.conf_int()    # Confidence intervals
            >>> fit.summary()     # Full summary table
        """
        from .fitted import FittedLikelihoodModel

        # Prepare type-split data
        prepared = self._prepare_data(data)

        # Use internal model's optimization
        params, n_iter = self._model._optimize(prepared, init, max_iter, tol, bounds)

        # Check convergence
        env = dict(prepared)
        env.update(params)
        try:
            score = self._model.score_at(env)
            converged = bool(np.linalg.norm(score) < tol)
        except Exception:
            converged = False

        # Return FittedLikelihoodModel with internal model and prepared data
        return FittedLikelihoodModel(self._model, prepared, params, n_iter, converged)

    def __repr__(self) -> str:
        types = list(self.contributions.keys())
        return f"ContributionModel(params={self.params}, types={types})"
