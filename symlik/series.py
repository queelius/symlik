"""
Series System Likelihood Contributions.

This module provides modular building blocks for constructing likelihood
contributions for series systems (systems that fail when any component fails).

Architecture:
    1. Component hazard/cumulative hazard functions
    2. Series system observation type combinators
    3. Pre-built convenience functions for common cases

For a series system with m components:
    - System lifetime: T = min(T₁, ..., Tₘ)
    - System survival: S_sys(t) = exp(-Σᵢ Hᵢ(t))
    - System hazard: h_sys(t) = Σᵢ hᵢ(t)

Where Hᵢ(t) is cumulative hazard and hᵢ(t) is instantaneous hazard.

Observation Types:
    - Known cause: Exact component identified
    - Masked cause: Only candidate set known (C1, C2, C3 conditions)
    - Right-censored: System survived past observation time
    - Left-censored: System failed before first observation
    - Interval-censored: System failed within a time interval
"""

from typing import List, Optional, Tuple, Union
from .evaluate import ExprType


# ============================================================
# Component Hazard Building Blocks
# ============================================================

class ComponentHazard:
    """
    Represents a component's hazard and cumulative hazard functions.

    For use in series system likelihood construction.
    """

    def __init__(
        self,
        hazard: ExprType,
        cumulative_hazard: ExprType,
        params: List[str],
    ):
        """
        Args:
            hazard: Instantaneous hazard h(t) as s-expression
            cumulative_hazard: Cumulative hazard H(t) = ∫₀ᵗ h(s)ds
            params: Parameter names used in this component
        """
        self.hazard = hazard
        self.cumulative_hazard = cumulative_hazard
        self.params = params


def exponential_component(
    rate: str,
    time_var: str = "t",
) -> ComponentHazard:
    """
    Exponential component for series system.

    h(t) = λ  (constant hazard)
    H(t) = λt

    Args:
        rate: Name of rate parameter (λ)
        time_var: Name of time variable

    Returns:
        ComponentHazard with exponential hazard functions
    """
    return ComponentHazard(
        hazard=rate,  # h(t) = λ
        cumulative_hazard=["*", rate, time_var],  # H(t) = λt
        params=[rate],
    )


def weibull_component(
    shape: str,
    scale: str,
    time_var: str = "t",
) -> ComponentHazard:
    """
    Weibull component for series system.

    h(t) = (k/θ)(t/θ)^(k-1)
    H(t) = (t/θ)^k

    Args:
        shape: Name of shape parameter (k)
        scale: Name of scale parameter (θ)
        time_var: Name of time variable

    Returns:
        ComponentHazard with Weibull hazard functions
    """
    t_over_scale = ["/", time_var, scale]

    return ComponentHazard(
        # h(t) = (k/θ) * (t/θ)^(k-1)
        hazard=["*",
                ["/", shape, scale],
                ["^", t_over_scale, ["-", shape, 1]]],
        # H(t) = (t/θ)^k
        cumulative_hazard=["^", t_over_scale, shape],
        params=[shape, scale],
    )


def gamma_component(
    shape: str,
    rate: str,
    time_var: str = "t",
) -> ComponentHazard:
    """
    Gamma component for series system (approximate, using Weibull approximation).

    Note: The gamma distribution doesn't have a closed-form cumulative hazard.
    This provides an approximation suitable for MLE purposes.

    For exact gamma series systems, consider numerical integration.

    Args:
        shape: Name of shape parameter (α)
        rate: Name of rate parameter (β)
        time_var: Name of time variable

    Returns:
        ComponentHazard with approximate gamma hazard functions
    """
    # Approximate gamma with Weibull: match mean and variance
    # Mean: α/β, Var: α/β²
    # This is a rough approximation; exact gamma requires special functions

    # For now, use the exact hazard but approximate cumulative hazard
    # h(t) = β^α * t^(α-1) * exp(-βt) / Γ(α) / S(t)
    # This is complex; for practical use, consider Weibull approximation

    raise NotImplementedError(
        "Gamma component requires incomplete gamma function. "
        "Consider using Weibull approximation or numerical methods."
    )


def lognormal_component(
    mu: str,
    sigma: str,
    time_var: str = "t",
) -> ComponentHazard:
    """
    Log-normal component for series system.

    Note: Log-normal doesn't have closed-form hazard/cumulative hazard.
    This is a placeholder for future implementation with numerical methods.

    Args:
        mu: Name of log-mean parameter
        sigma: Name of log-scale parameter
        time_var: Name of time variable

    Returns:
        ComponentHazard (requires numerical evaluation)
    """
    raise NotImplementedError(
        "Log-normal component requires numerical hazard evaluation. "
        "Consider using Weibull approximation."
    )


def custom_component(
    hazard: ExprType,
    cumulative_hazard: ExprType,
    params: Optional[List[str]] = None,
) -> ComponentHazard:
    """
    Create a component with custom hazard and cumulative hazard functions.

    This is the most general constructor, allowing arbitrary s-expressions
    for hazard functions that can depend on:
    - Parameters (to be estimated)
    - Covariates (observation-specific data)
    - Time variable

    Args:
        hazard: Instantaneous hazard h(t) as s-expression
        cumulative_hazard: Cumulative hazard H(t) = ∫₀ᵗ h(s)ds as s-expression
        params: Optional list of parameter names (for documentation)

    Returns:
        ComponentHazard object

    Example:
        >>> # Proportional hazards: h(t) = λ * exp(β * x)
        >>> # where x is a covariate
        >>> comp = custom_component(
        ...     hazard=["*", "lambda", ["exp", ["*", "beta", "x"]]],
        ...     cumulative_hazard=["*", ["*", "lambda", "t"], ["exp", ["*", "beta", "x"]]],
        ...     params=["lambda", "beta"]
        ... )
    """
    return ComponentHazard(
        hazard=hazard,
        cumulative_hazard=cumulative_hazard,
        params=params or [],
    )


# ============================================================
# Covariate-Dependent Components (Proportional Hazards)
# ============================================================

def exponential_ph_component(
    baseline_rate: str,
    coefficients: List[str],
    covariates: List[str],
    time_var: str = "t",
) -> ComponentHazard:
    """
    Exponential proportional hazards component.

    h(t|x) = λ₀ * exp(β₁x₁ + β₂x₂ + ...)
    H(t|x) = λ₀ * t * exp(β₁x₁ + β₂x₂ + ...)

    This allows failure rates to depend on observation-specific covariates.

    Args:
        baseline_rate: Name of baseline rate parameter (λ₀)
        coefficients: Names of coefficient parameters [β₁, β₂, ...]
        covariates: Names of covariate variables [x₁, x₂, ...]
        time_var: Name of time variable

    Returns:
        ComponentHazard with covariate-dependent hazard

    Example:
        >>> # Component rate depends on temperature
        >>> comp = exponential_ph_component(
        ...     baseline_rate="lambda0",
        ...     coefficients=["beta_temp"],
        ...     covariates=["temperature"]
        ... )
    """
    if len(coefficients) != len(covariates):
        raise ValueError("coefficients and covariates must have same length")

    # Build linear predictor: β₁x₁ + β₂x₂ + ...
    if len(coefficients) == 0:
        linear_pred = 0
    else:
        terms = [["*", b, x] for b, x in zip(coefficients, covariates)]
        linear_pred = terms[0] if len(terms) == 1 else ["+"] + terms

    # exp(linear predictor)
    exp_lp = ["exp", linear_pred] if linear_pred != 0 else 1

    # h(t|x) = λ₀ * exp(βx)
    if exp_lp == 1:
        hazard = baseline_rate
    else:
        hazard = ["*", baseline_rate, exp_lp]

    # H(t|x) = λ₀ * t * exp(βx)
    if exp_lp == 1:
        cumulative_hazard = ["*", baseline_rate, time_var]
    else:
        cumulative_hazard = ["*", ["*", baseline_rate, time_var], exp_lp]

    return ComponentHazard(
        hazard=hazard,
        cumulative_hazard=cumulative_hazard,
        params=[baseline_rate] + coefficients,
    )


def weibull_ph_component(
    shape: str,
    baseline_scale: str,
    coefficients: List[str],
    covariates: List[str],
    time_var: str = "t",
) -> ComponentHazard:
    """
    Weibull proportional hazards component (AFT parameterization).

    For Weibull with covariates, we use the accelerated failure time form:
        h(t|x) = (k/θ₀) * (t/θ₀)^(k-1) * exp(k * βx)
        H(t|x) = (t/θ₀)^k * exp(k * βx)

    This corresponds to multiplying lifetime by exp(-βx).

    Args:
        shape: Name of shape parameter (k)
        baseline_scale: Name of baseline scale parameter (θ₀)
        coefficients: Names of coefficient parameters [β₁, β₂, ...]
        covariates: Names of covariate variables [x₁, x₂, ...]
        time_var: Name of time variable

    Returns:
        ComponentHazard with covariate-dependent Weibull hazard
    """
    if len(coefficients) != len(covariates):
        raise ValueError("coefficients and covariates must have same length")

    t_over_scale = ["/", time_var, baseline_scale]

    # Linear predictor: β₁x₁ + β₂x₂ + ...
    if len(coefficients) == 0:
        linear_pred = 0
    else:
        terms = [["*", b, x] for b, x in zip(coefficients, covariates)]
        linear_pred = terms[0] if len(terms) == 1 else ["+"] + terms

    # Baseline Weibull hazard: (k/θ₀) * (t/θ₀)^(k-1)
    baseline_hazard = ["*",
                       ["/", shape, baseline_scale],
                       ["^", t_over_scale, ["-", shape, 1]]]

    # Baseline cumulative hazard: (t/θ₀)^k
    baseline_cum_hazard = ["^", t_over_scale, shape]

    if linear_pred == 0:
        hazard = baseline_hazard
        cumulative_hazard = baseline_cum_hazard
    else:
        # exp(k * βx) multiplier
        exp_k_lp = ["exp", ["*", shape, linear_pred]]
        hazard = ["*", baseline_hazard, exp_k_lp]
        cumulative_hazard = ["*", baseline_cum_hazard, exp_k_lp]

    return ComponentHazard(
        hazard=hazard,
        cumulative_hazard=cumulative_hazard,
        params=[shape, baseline_scale] + coefficients,
    )


def exponential_log_linear_component(
    log_rate_intercept: str,
    coefficients: List[str],
    covariates: List[str],
    time_var: str = "t",
) -> ComponentHazard:
    """
    Exponential component with log-linear rate model.

    log(λ) = α + β₁x₁ + β₂x₂ + ...
    λ = exp(α + βx)

    h(t|x) = exp(α + βx)
    H(t|x) = t * exp(α + βx)

    This is equivalent to exponential_ph_component but parameterized
    with log-rate intercept instead of baseline rate.

    Args:
        log_rate_intercept: Name of log-rate intercept parameter (α)
        coefficients: Names of coefficient parameters [β₁, β₂, ...]
        covariates: Names of covariate variables [x₁, x₂, ...]
        time_var: Name of time variable

    Returns:
        ComponentHazard with log-linear rate model
    """
    # Build α + β₁x₁ + β₂x₂ + ...
    terms = [log_rate_intercept]
    terms.extend(["*", b, x] for b, x in zip(coefficients, covariates))

    if len(terms) == 1:
        linear_comb = terms[0]
    else:
        linear_comb = ["+"] + terms

    # λ = exp(α + βx)
    rate = ["exp", linear_comb]

    return ComponentHazard(
        hazard=rate,
        cumulative_hazard=["*", time_var, rate],
        params=[log_rate_intercept] + coefficients,
    )


# ============================================================
# Series System Composition Helpers
# ============================================================

def _sum_expr(exprs: List[ExprType]) -> ExprType:
    """Sum a list of expressions."""
    if len(exprs) == 0:
        return 0
    elif len(exprs) == 1:
        return exprs[0]
    else:
        return ["+"] + list(exprs)


def _system_cumulative_hazard(components: List[ComponentHazard]) -> ExprType:
    """
    Total cumulative hazard for a series system.

    H_sys(t) = Σᵢ Hᵢ(t)
    """
    return _sum_expr([c.cumulative_hazard for c in components])


def _system_hazard_subset(
    components: List[ComponentHazard],
    indices: List[int],
) -> ExprType:
    """
    Sum of hazards over a subset of components.

    Σⱼ∈C hⱼ(t)
    """
    hazards = [components[i].hazard for i in indices]
    return _sum_expr(hazards)


def _neg_cumulative_hazard(components: List[ComponentHazard]) -> ExprType:
    """
    Negative cumulative hazard (log survival).

    log S_sys(t) = -H_sys(t) = -Σᵢ Hᵢ(t)
    """
    H_sys = _system_cumulative_hazard(components)
    return ["*", -1, H_sys]


# ============================================================
# Series System Observation Type Contributions
# ============================================================

def series_known_cause(
    components: List[ComponentHazard],
    cause_index: int,
) -> ExprType:
    """
    Log-likelihood contribution for series system with known failure cause.

    When component j causes failure at time t:
        log L = log(hⱼ(t)) + log(S_sys(t))
              = log(hⱼ(t)) - Σᵢ Hᵢ(t)

    Args:
        components: List of ComponentHazard objects
        cause_index: Index (0-based) of component that caused failure

    Returns:
        S-expression for log-likelihood contribution
    """
    # log(h_j(t))
    log_hazard = ["log", components[cause_index].hazard]

    # -Σᵢ Hᵢ(t)
    neg_cum_hazard = _neg_cumulative_hazard(components)

    return ["+", log_hazard, neg_cum_hazard]


def series_masked_cause(
    components: List[ComponentHazard],
    candidate_indices: List[int],
) -> ExprType:
    """
    Log-likelihood contribution for series system with masked failure cause.

    Under C1, C2, C3 conditions, when failure has candidate set C:
        log L = log(Σⱼ∈C hⱼ(t)) + log(S_sys(t))
              = log(Σⱼ∈C hⱼ(t)) - Σᵢ Hᵢ(t)

    C1, C2, C3 Conditions:
        C1: True cause is always in the candidate set
        C2: P(candidate set) independent of which component in set failed
        C3: Masking probabilities independent of parameter vector θ

    Args:
        components: List of ComponentHazard objects
        candidate_indices: Indices (0-based) of components in candidate set

    Returns:
        S-expression for log-likelihood contribution
    """
    # log(Σⱼ∈C hⱼ(t))
    hazard_sum = _system_hazard_subset(components, candidate_indices)
    log_hazard_sum = ["log", hazard_sum]

    # -Σᵢ Hᵢ(t)
    neg_cum_hazard = _neg_cumulative_hazard(components)

    return ["+", log_hazard_sum, neg_cum_hazard]


def series_right_censored(
    components: List[ComponentHazard],
) -> ExprType:
    """
    Log-likelihood contribution for right-censored series system.

    System survived past observation time t:
        log L = log(S_sys(t)) = -Σᵢ Hᵢ(t)

    Args:
        components: List of ComponentHazard objects

    Returns:
        S-expression for log-likelihood contribution
    """
    return _neg_cumulative_hazard(components)


def series_left_censored(
    components: List[ComponentHazard],
) -> ExprType:
    """
    Log-likelihood contribution for left-censored series system.

    System failed before observation time t:
        log L = log(F_sys(t)) = log(1 - S_sys(t))
              = log(1 - exp(-Σᵢ Hᵢ(t)))

    Args:
        components: List of ComponentHazard objects

    Returns:
        S-expression for log-likelihood contribution
    """
    # -Σᵢ Hᵢ(t)
    neg_cum_hazard = _neg_cumulative_hazard(components)

    # log(1 - exp(-Σᵢ Hᵢ(t)))
    return ["log", ["-", 1, ["exp", neg_cum_hazard]]]


def series_interval_censored(
    components: List[ComponentHazard],
    lower_time_var: str = "t_lower",
    upper_time_var: str = "t_upper",
) -> ExprType:
    """
    Log-likelihood contribution for interval-censored series system.

    System failed in interval (t_l, t_u]:
        log L = log(S_sys(t_l) - S_sys(t_u))
              = log(exp(-H_sys(t_l)) - exp(-H_sys(t_u)))

    Note: This requires time-dependent cumulative hazard evaluation.
    The components must use time variables that can be substituted.

    Args:
        components: List of ComponentHazard objects
        lower_time_var: Variable name for lower bound
        upper_time_var: Variable name for upper bound

    Returns:
        S-expression for log-likelihood contribution

    Warning:
        This function assumes components were created with the same time_var
        that will be used for substitution. For interval censoring, you may
        need to create separate component hazards for each time bound.
    """
    # For interval censoring, we need H(t_l) and H(t_u)
    # This is tricky because components are defined with a single time_var
    #
    # Approach: Return expression that user evaluates with substitution
    # Or: require user to pass cumulative hazards at both times

    raise NotImplementedError(
        "Interval censoring for series systems requires careful handling "
        "of time variable substitution. Use series_interval_censored_simple() "
        "for exponential components, or construct manually for Weibull."
    )


def series_interval_censored_exponential(
    rates: List[str],
    lower_time_var: str = "t_lower",
    upper_time_var: str = "t_upper",
) -> ExprType:
    """
    Log-likelihood for interval-censored series with exponential components.

    For exponential, H(t) = λt, so:
        log L = log(exp(-λ_sys * t_l) - exp(-λ_sys * t_u))

    where λ_sys = Σλᵢ

    Args:
        rates: List of rate parameter names
        lower_time_var: Variable name for lower bound
        upper_time_var: Variable name for upper bound

    Returns:
        S-expression for log-likelihood contribution
    """
    rate_sum = _sum_expr(rates)

    # S(t_l) = exp(-λ_sys * t_l)
    s_lower = ["exp", ["*", -1, ["*", rate_sum, lower_time_var]]]

    # S(t_u) = exp(-λ_sys * t_u)
    s_upper = ["exp", ["*", -1, ["*", rate_sum, upper_time_var]]]

    # log(S(t_l) - S(t_u))
    return ["log", ["-", s_lower, s_upper]]


# ============================================================
# Convenience Functions: Exponential Series Systems
# ============================================================

def exponential_series_known_cause(
    rates: List[str],
    cause_index: int,
    time_var: str = "t",
) -> ExprType:
    """
    Log-likelihood for exponential series system with known failure cause.

    log L = log(λⱼ) - t * Σᵢλᵢ

    Args:
        rates: List of rate parameter names [λ₁, λ₂, ...]
        cause_index: Index (0-based) of failing component
        time_var: Name of time variable

    Returns:
        S-expression for log-likelihood contribution
    """
    components = [exponential_component(r, time_var) for r in rates]
    return series_known_cause(components, cause_index)


def exponential_series_masked_cause(
    rates: List[str],
    candidate_indices: List[int],
    time_var: str = "t",
) -> ExprType:
    """
    Log-likelihood for exponential series system with masked failure cause.

    Under C1, C2, C3:
        log L = log(Σⱼ∈C λⱼ) - t * Σᵢλᵢ

    Args:
        rates: List of rate parameter names [λ₁, λ₂, ...]
        candidate_indices: Indices (0-based) in candidate set
        time_var: Name of time variable

    Returns:
        S-expression for log-likelihood contribution
    """
    components = [exponential_component(r, time_var) for r in rates]
    return series_masked_cause(components, candidate_indices)


def exponential_series_right_censored(
    rates: List[str],
    time_var: str = "t",
) -> ExprType:
    """
    Log-likelihood for right-censored exponential series system.

    log L = -t * Σᵢλᵢ

    Args:
        rates: List of rate parameter names [λ₁, λ₂, ...]
        time_var: Name of time variable

    Returns:
        S-expression for log-likelihood contribution
    """
    components = [exponential_component(r, time_var) for r in rates]
    return series_right_censored(components)


def exponential_series_left_censored(
    rates: List[str],
    time_var: str = "t",
) -> ExprType:
    """
    Log-likelihood for left-censored exponential series system.

    log L = log(1 - exp(-t * Σᵢλᵢ))

    Args:
        rates: List of rate parameter names [λ₁, λ₂, ...]
        time_var: Name of time variable

    Returns:
        S-expression for log-likelihood contribution
    """
    components = [exponential_component(r, time_var) for r in rates]
    return series_left_censored(components)


def exponential_series_interval_censored(
    rates: List[str],
    lower_time_var: str = "t_lower",
    upper_time_var: str = "t_upper",
) -> ExprType:
    """
    Log-likelihood for interval-censored exponential series system.

    log L = log(exp(-t_l * Σλᵢ) - exp(-t_u * Σλᵢ))

    Args:
        rates: List of rate parameter names [λ₁, λ₂, ...]
        lower_time_var: Variable name for lower bound
        upper_time_var: Variable name for upper bound

    Returns:
        S-expression for log-likelihood contribution
    """
    return series_interval_censored_exponential(rates, lower_time_var, upper_time_var)


# ============================================================
# Convenience Functions: Weibull Series Systems
# ============================================================

def weibull_series_known_cause(
    shapes: List[str],
    scales: List[str],
    cause_index: int,
    time_var: str = "t",
) -> ExprType:
    """
    Log-likelihood for Weibull series system with known failure cause.

    log L = log(hⱼ(t)) - Σᵢ(t/θᵢ)^kᵢ

    where hⱼ(t) = (kⱼ/θⱼ)(t/θⱼ)^(kⱼ-1)

    Args:
        shapes: List of shape parameter names [k₁, k₂, ...]
        scales: List of scale parameter names [θ₁, θ₂, ...]
        cause_index: Index (0-based) of failing component
        time_var: Name of time variable

    Returns:
        S-expression for log-likelihood contribution
    """
    if len(shapes) != len(scales):
        raise ValueError("shapes and scales must have same length")

    components = [
        weibull_component(k, theta, time_var)
        for k, theta in zip(shapes, scales)
    ]
    return series_known_cause(components, cause_index)


def weibull_series_masked_cause(
    shapes: List[str],
    scales: List[str],
    candidate_indices: List[int],
    time_var: str = "t",
) -> ExprType:
    """
    Log-likelihood for Weibull series system with masked failure cause.

    Under C1, C2, C3:
        log L = log(Σⱼ∈C hⱼ(t)) - Σᵢ(t/θᵢ)^kᵢ

    Args:
        shapes: List of shape parameter names [k₁, k₂, ...]
        scales: List of scale parameter names [θ₁, θ₂, ...]
        candidate_indices: Indices (0-based) in candidate set
        time_var: Name of time variable

    Returns:
        S-expression for log-likelihood contribution
    """
    if len(shapes) != len(scales):
        raise ValueError("shapes and scales must have same length")

    components = [
        weibull_component(k, theta, time_var)
        for k, theta in zip(shapes, scales)
    ]
    return series_masked_cause(components, candidate_indices)


def weibull_series_right_censored(
    shapes: List[str],
    scales: List[str],
    time_var: str = "t",
) -> ExprType:
    """
    Log-likelihood for right-censored Weibull series system.

    log L = -Σᵢ(t/θᵢ)^kᵢ

    Args:
        shapes: List of shape parameter names [k₁, k₂, ...]
        scales: List of scale parameter names [θ₁, θ₂, ...]
        time_var: Name of time variable

    Returns:
        S-expression for log-likelihood contribution
    """
    if len(shapes) != len(scales):
        raise ValueError("shapes and scales must have same length")

    components = [
        weibull_component(k, theta, time_var)
        for k, theta in zip(shapes, scales)
    ]
    return series_right_censored(components)


def weibull_series_left_censored(
    shapes: List[str],
    scales: List[str],
    time_var: str = "t",
) -> ExprType:
    """
    Log-likelihood for left-censored Weibull series system.

    log L = log(1 - exp(-Σᵢ(t/θᵢ)^kᵢ))

    Args:
        shapes: List of shape parameter names [k₁, k₂, ...]
        scales: List of scale parameter names [θ₁, θ₂, ...]
        time_var: Name of time variable

    Returns:
        S-expression for log-likelihood contribution
    """
    if len(shapes) != len(scales):
        raise ValueError("shapes and scales must have same length")

    components = [
        weibull_component(k, theta, time_var)
        for k, theta in zip(shapes, scales)
    ]
    return series_left_censored(components)


# ============================================================
# Mixed Component Systems
# ============================================================

def mixed_series_known_cause(
    components: List[ComponentHazard],
    cause_index: int,
) -> ExprType:
    """
    Log-likelihood for series system with mixed component types and known cause.

    Example: Component 1 is exponential, component 2 is Weibull.

    Args:
        components: List of ComponentHazard objects (can be different types)
        cause_index: Index (0-based) of failing component

    Returns:
        S-expression for log-likelihood contribution
    """
    return series_known_cause(components, cause_index)


def mixed_series_masked_cause(
    components: List[ComponentHazard],
    candidate_indices: List[int],
) -> ExprType:
    """
    Log-likelihood for series system with mixed component types and masked cause.

    Args:
        components: List of ComponentHazard objects (can be different types)
        candidate_indices: Indices (0-based) in candidate set

    Returns:
        S-expression for log-likelihood contribution
    """
    return series_masked_cause(components, candidate_indices)


def mixed_series_right_censored(
    components: List[ComponentHazard],
) -> ExprType:
    """
    Log-likelihood for right-censored series system with mixed component types.

    Args:
        components: List of ComponentHazard objects (can be different types)

    Returns:
        S-expression for log-likelihood contribution
    """
    return series_right_censored(components)


def mixed_series_left_censored(
    components: List[ComponentHazard],
) -> ExprType:
    """
    Log-likelihood for left-censored series system with mixed component types.

    Args:
        components: List of ComponentHazard objects (can be different types)

    Returns:
        S-expression for log-likelihood contribution
    """
    return series_left_censored(components)


# ============================================================
# Factory Functions for Building Contribution Models
# ============================================================

def build_exponential_series_contributions(
    m: int,
    rate_names: Optional[List[str]] = None,
    time_var: str = "t",
    include_left_censored: bool = False,
    include_interval_censored: bool = False,
    lower_time_var: str = "t_lower",
    upper_time_var: str = "t_upper",
) -> dict:
    """
    Build a complete set of contributions for an m-component exponential series.

    Generates contributions for:
        - known_j: Known cause (component j failed), j = 1..m
        - masked_X: Masked cause with candidate set X (all 2^m - m - 1 subsets)
        - right_censored: Right-censored
        - left_censored: Left-censored (optional)
        - interval_censored: Interval-censored (optional)

    Args:
        m: Number of components
        rate_names: Parameter names for rates [λ₁, ..., λₘ].
                   Defaults to ["lambda1", "lambda2", ...]
        time_var: Time variable name
        include_left_censored: Include left-censored contribution
        include_interval_censored: Include interval-censored contribution
        lower_time_var: Lower bound time variable for interval censoring
        upper_time_var: Upper bound time variable for interval censoring

    Returns:
        Dictionary mapping contribution type names to s-expressions

    Example:
        >>> contribs = build_exponential_series_contributions(3)
        >>> # Returns: {
        >>> #   "known_1": ..., "known_2": ..., "known_3": ...,
        >>> #   "masked_12": ..., "masked_13": ..., "masked_23": ...,
        >>> #   "masked_123": ...,
        >>> #   "right_censored": ...
        >>> # }
    """
    if rate_names is None:
        rate_names = [f"lambda{i+1}" for i in range(m)]

    if len(rate_names) != m:
        raise ValueError(f"rate_names must have {m} elements")

    contributions = {}

    # Known cause contributions
    for j in range(m):
        key = f"known_{j+1}"
        contributions[key] = exponential_series_known_cause(
            rate_names, j, time_var
        )

    # Masked cause contributions (all subsets of size 2 to m)
    from itertools import combinations
    for size in range(2, m + 1):
        for combo in combinations(range(m), size):
            key = "masked_" + "".join(str(i+1) for i in combo)
            contributions[key] = exponential_series_masked_cause(
                rate_names, list(combo), time_var
            )

    # Right-censored
    contributions["right_censored"] = exponential_series_right_censored(
        rate_names, time_var
    )

    # Left-censored (optional)
    if include_left_censored:
        contributions["left_censored"] = exponential_series_left_censored(
            rate_names, time_var
        )

    # Interval-censored (optional)
    if include_interval_censored:
        contributions["interval_censored"] = exponential_series_interval_censored(
            rate_names, lower_time_var, upper_time_var
        )

    return contributions


def build_weibull_series_contributions(
    m: int,
    shape_names: Optional[List[str]] = None,
    scale_names: Optional[List[str]] = None,
    time_var: str = "t",
    include_left_censored: bool = False,
) -> dict:
    """
    Build a complete set of contributions for an m-component Weibull series.

    Args:
        m: Number of components
        shape_names: Parameter names for shapes [k₁, ..., kₘ].
                    Defaults to ["k1", "k2", ...]
        scale_names: Parameter names for scales [θ₁, ..., θₘ].
                    Defaults to ["theta1", "theta2", ...]
        time_var: Time variable name
        include_left_censored: Include left-censored contribution

    Returns:
        Dictionary mapping contribution type names to s-expressions
    """
    if shape_names is None:
        shape_names = [f"k{i+1}" for i in range(m)]
    if scale_names is None:
        scale_names = [f"theta{i+1}" for i in range(m)]

    if len(shape_names) != m or len(scale_names) != m:
        raise ValueError(f"shape_names and scale_names must have {m} elements")

    contributions = {}

    # Known cause contributions
    for j in range(m):
        key = f"known_{j+1}"
        contributions[key] = weibull_series_known_cause(
            shape_names, scale_names, j, time_var
        )

    # Masked cause contributions
    from itertools import combinations
    for size in range(2, m + 1):
        for combo in combinations(range(m), size):
            key = "masked_" + "".join(str(i+1) for i in combo)
            contributions[key] = weibull_series_masked_cause(
                shape_names, scale_names, list(combo), time_var
            )

    # Right-censored
    contributions["right_censored"] = weibull_series_right_censored(
        shape_names, scale_names, time_var
    )

    # Left-censored (optional)
    if include_left_censored:
        contributions["left_censored"] = weibull_series_left_censored(
            shape_names, scale_names, time_var
        )

    return contributions
