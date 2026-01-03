"""
Fitted likelihood model results.

Follows the statsmodels convention: model.fit() returns a results object
with properties for estimates and methods for inference.
"""

from typing import Any, Dict, List, Tuple, TYPE_CHECKING
import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from .model import LikelihoodModel


class FittedLikelihoodModel:
    """
    Results from fitting a LikelihoodModel to data.

    This class follows the statsmodels convention:
    - Properties for simple cached values (params, se, llf, aic, bic)
    - Methods for computed/parameterized results (conf_int, wald_test)

    Example:
        >>> from symlik.distributions import exponential
        >>> model = exponential()
        >>> fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})
        >>> fit.params
        {'lambda': 0.333...}
        >>> fit.aic
        12.34
        >>> fit.conf_int()
        {'lambda': (0.21, 0.45)}
        >>> print(fit.summary())
    """

    def __init__(
        self,
        model: 'LikelihoodModel',
        data: Dict[str, Any],
        params: Dict[str, float],
        n_iter: int,
        converged: bool,
    ):
        """
        Initialize fitted model results.

        Args:
            model: The LikelihoodModel that was fit
            data: Data dictionary used for fitting
            params: MLE parameter estimates
            n_iter: Number of optimization iterations
            converged: Whether optimization converged
        """
        self._model = model
        self._data = data
        self._params = params
        self._n_iter = n_iter
        self._converged = converged

        # Lazy caches
        self._llf: float = None
        self._se: Dict[str, float] = None
        self._cov: np.ndarray = None
        self._info: np.ndarray = None

    # ================================================================
    # Properties
    # ================================================================

    @property
    def model(self) -> 'LikelihoodModel':
        """The underlying LikelihoodModel specification."""
        return self._model

    @property
    def params(self) -> Dict[str, float]:
        """Parameter estimates (MLE)."""
        return self._params.copy()

    @property
    def param_names(self) -> List[str]:
        """Names of parameters in order."""
        return self._model.params

    @property
    def nobs(self) -> int:
        """Number of observations."""
        for val in self._data.values():
            if isinstance(val, (list, np.ndarray)):
                return len(val)
        raise ValueError("Cannot determine sample size from data")

    @property
    def df_model(self) -> int:
        """Model degrees of freedom (number of estimated parameters)."""
        return len(self._model.params)

    @property
    def llf(self) -> float:
        """Log-likelihood at MLE."""
        if self._llf is None:
            env = dict(self._data)
            env.update(self._params)
            self._llf = self._model.evaluate(env)
        return self._llf

    @property
    def aic(self) -> float:
        """
        Akaike Information Criterion.

        AIC = -2*llf + 2*k

        Lower is better. Penalizes model complexity less than BIC.
        """
        return -2 * self.llf + 2 * self.df_model

    @property
    def bic(self) -> float:
        """
        Bayesian Information Criterion.

        BIC = -2*llf + k*log(n)

        Lower is better. Penalizes model complexity more than AIC for n > 7.
        """
        return -2 * self.llf + self.df_model * np.log(self.nobs)

    @property
    def se(self) -> Dict[str, float]:
        """
        Standard errors of parameter estimates.

        Computed from the inverse of the observed Fisher information:
        SE(θ̂) = sqrt(diag(I(θ̂)⁻¹))
        """
        if self._se is None:
            self._compute_covariance()
        return self._se.copy()

    # Alias for statsmodels compatibility
    @property
    def bse(self) -> Dict[str, float]:
        """Standard errors (alias for se, statsmodels compatibility)."""
        return self.se

    @property
    def converged(self) -> bool:
        """Whether optimization converged."""
        return self._converged

    @property
    def n_iter(self) -> int:
        """Number of iterations used in optimization."""
        return self._n_iter

    # ================================================================
    # Private Methods
    # ================================================================

    def _compute_covariance(self) -> None:
        """Compute and cache covariance matrix and standard errors."""
        if self._cov is not None:
            return

        env = dict(self._data)
        env.update(self._params)
        self._info = self._model.information_at(env)

        try:
            self._cov = np.linalg.inv(self._info)
            se_vals = np.sqrt(np.diag(self._cov))
            self._se = {p: float(se_vals[i]) for i, p in enumerate(self._model.params)}
        except np.linalg.LinAlgError:
            n = len(self._model.params)
            self._cov = np.full((n, n), np.nan)
            self._se = {p: np.nan for p in self._model.params}

    def _get_env(self, param_values: Dict[str, float] = None) -> Dict[str, Any]:
        """Build evaluation environment with data and parameters."""
        env = dict(self._data)
        env.update(param_values or self._params)
        return env

    # ================================================================
    # Methods
    # ================================================================

    def cov_params(self) -> np.ndarray:
        """
        Covariance matrix of parameter estimates.

        Returns the inverse of the observed Fisher information matrix.

        Returns:
            2D numpy array of shape (k, k) where k is number of parameters
        """
        if self._cov is None:
            self._compute_covariance()
        return self._cov.copy()

    def information_matrix(self) -> np.ndarray:
        """
        Observed Fisher information matrix at MLE.

        I(θ̂) = -∂²ℓ/∂θ∂θ' evaluated at θ̂

        Returns:
            2D numpy array of shape (k, k)
        """
        if self._info is None:
            self._compute_covariance()
        return self._info.copy()

    def conf_int(self, alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
        """
        Wald confidence intervals for all parameters.

        CI = θ̂ ± z_{α/2} * SE(θ̂)

        Args:
            alpha: Significance level (default 0.05 for 95% CI)

        Returns:
            Dict mapping parameter names to (lower, upper) tuples

        Example:
            >>> fit.conf_int()           # 95% CI
            {'lambda': (0.21, 0.45)}
            >>> fit.conf_int(alpha=0.01)  # 99% CI
        """
        z = stats.norm.ppf(1 - alpha / 2)
        se = self.se
        return {
            p: (self._params[p] - z * se[p], self._params[p] + z * se[p])
            for p in self._model.params
        }

    def wald_test(self, null: Dict[str, float]) -> Dict[str, Any]:
        """
        Wald test for parameter hypotheses.

        Tests H₀: θ = θ₀ vs H₁: θ ≠ θ₀

        W = (θ̂ - θ₀)' I(θ̂) (θ̂ - θ₀) ~ χ²(k)

        Args:
            null: Null hypothesis values for parameters to test.
                  Only parameters in this dict are tested.

        Returns:
            Dict with 'statistic', 'df', 'pvalue'

        Example:
            >>> fit.wald_test({'lambda': 1.0})
            {'statistic': 15.2, 'df': 1, 'pvalue': 0.0001}
        """
        tested_params = [p for p in self._model.params if p in null]
        tested_indices = [self._model.params.index(p) for p in tested_params]

        diff = np.array([self._params[p] - null[p] for p in tested_params])

        if self._info is None:
            self._compute_covariance()

        I_sub = self._info[np.ix_(tested_indices, tested_indices)]

        try:
            statistic = float(diff @ I_sub @ diff)
        except (np.linalg.LinAlgError, ValueError):
            statistic = np.nan

        df = len(tested_params)
        pvalue = 1 - stats.chi2.cdf(statistic, df) if np.isfinite(statistic) else np.nan

        return {'statistic': statistic, 'df': df, 'pvalue': pvalue}

    def score_test(self, null: Dict[str, float]) -> Dict[str, Any]:
        """
        Score (Lagrange multiplier) test.

        Tests H₀ without requiring MLE under alternative.

        S = U(θ₀)' I(θ₀)⁻¹ U(θ₀) ~ χ²(k)

        Args:
            null: Full parameter values under null hypothesis

        Returns:
            Dict with 'statistic', 'df', 'pvalue'

        Example:
            >>> fit.score_test({'lambda': 1.0})
        """
        env = self._get_env(null)

        U = self._model.score_at(env)
        I = self._model.information_at(env)

        try:
            I_inv = np.linalg.inv(I)
            statistic = float(U @ I_inv @ U)
        except np.linalg.LinAlgError:
            statistic = np.nan

        df = len(self._model.params)
        pvalue = 1 - stats.chi2.cdf(statistic, df) if np.isfinite(statistic) else np.nan

        return {'statistic': statistic, 'df': df, 'pvalue': pvalue}

    def lr_test(self, null: Dict[str, float]) -> Dict[str, Any]:
        """
        Likelihood ratio test.

        LR = 2(ℓ(θ̂) - ℓ(θ₀)) ~ χ²(k)

        Args:
            null: Parameter values under null hypothesis.
                  For composite null, include all parameters with
                  constrained params set to null values.

        Returns:
            Dict with 'statistic', 'df', 'pvalue'

        Example:
            >>> fit.lr_test({'lambda': 1.0})
        """
        env_null = self._get_env(null)
        ll_null = self._model.evaluate(env_null)

        statistic = 2 * (self.llf - ll_null)

        # Count constrained parameters (those that differ from MLE)
        df = sum(
            1 for p in self._model.params
            if p in null and abs(null[p] - self._params.get(p, null[p])) > 1e-10
        )

        pvalue = 1 - stats.chi2.cdf(statistic, df) if df > 0 else np.nan

        return {'statistic': statistic, 'df': df, 'pvalue': pvalue}

    def summary(self, alpha: float = 0.05) -> str:
        """
        Generate a summary table of estimation results.

        Args:
            alpha: Significance level for confidence intervals

        Returns:
            Formatted string summary

        Example:
            >>> print(fit.summary())
            ==================================================
                      Likelihood Model Results
            ==================================================
            No. Observations:              50
            No. Parameters:                 1
            Log-Likelihood:           -45.1234
            AIC:                       92.2468
            BIC:                       94.1234
            Converged:                   True
            --------------------------------------------------
            Parameter       Estimate     Std.Err     [0.025     0.975]
            --------------------------------------------------
            lambda            0.3333      0.0471     0.2410     0.4256
            ==================================================
        """
        width = 60
        lines = []

        lines.append("=" * width)
        lines.append("Likelihood Model Results".center(width))
        lines.append("=" * width)
        lines.append(f"{'No. Observations:':<25} {self.nobs:>12}")
        lines.append(f"{'No. Parameters:':<25} {self.df_model:>12}")
        lines.append(f"{'Log-Likelihood:':<25} {self.llf:>12.4f}")
        lines.append(f"{'AIC:':<25} {self.aic:>12.4f}")
        lines.append(f"{'BIC:':<25} {self.bic:>12.4f}")
        lines.append(f"{'Converged:':<25} {str(self.converged):>12}")
        lines.append("-" * width)

        # Confidence interval bounds for header
        lo_pct = alpha / 2
        hi_pct = 1 - alpha / 2

        header = f"{'Parameter':<12} {'Estimate':>12} {'Std.Err':>12} {f'[{lo_pct:.3f}':>10} {f'{hi_pct:.3f}]':>10}"
        lines.append(header)
        lines.append("-" * width)

        cis = self.conf_int(alpha=alpha)
        se = self.se

        for param in self._model.params:
            est = self._params[param]
            se_val = se[param]
            lo, hi = cis[param]
            line = f"{param:<12} {est:>12.4f} {se_val:>12.4f} {lo:>10.4f} {hi:>10.4f}"
            lines.append(line)

        lines.append("=" * width)
        return "\n".join(lines)

    def __repr__(self) -> str:
        params_str = ", ".join(f"{p}={v:.4g}" for p, v in self._params.items())
        return f"<FittedLikelihoodModel({params_str}), llf={self.llf:.4f}>"

    def __str__(self) -> str:
        return self.summary()
