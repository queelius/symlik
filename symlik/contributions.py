"""
Convenience constructors for common likelihood contributions.

These are log-likelihood contributions for single observations,
designed for use with ContributionModel.
"""

from typing import List
from .evaluate import ExprType


# ============================================================
# Exponential Distribution Contributions
# ============================================================

def complete_exponential(time_var: str = "t", rate: str = "lambda") -> ExprType:
    """
    Log-likelihood contribution for a complete (uncensored) exponential observation.

    For an exact failure time t with rate lambda:
        log f(t|lambda) = log(lambda) - lambda*t

    Args:
        time_var: Name of the time/duration variable
        rate: Name of the rate parameter

    Returns:
        S-expression for log-likelihood contribution
    """
    return ["+", ["log", rate], ["*", -1, ["*", rate, time_var]]]


def right_censored_exponential(time_var: str = "t", rate: str = "lambda") -> ExprType:
    """
    Log-likelihood contribution for a right-censored exponential observation.

    For a censoring time t (subject survived past t):
        log S(t|lambda) = -lambda*t

    where S(t) is the survival function.

    Args:
        time_var: Name of the censoring time variable
        rate: Name of the rate parameter

    Returns:
        S-expression for log-likelihood contribution
    """
    return ["*", -1, ["*", rate, time_var]]


def left_censored_exponential(time_var: str = "t", rate: str = "lambda") -> ExprType:
    """
    Log-likelihood contribution for a left-censored exponential observation.

    For a left-censoring time t (subject failed before t):
        log F(t|lambda) = log(1 - exp(-lambda*t))

    where F(t) is the CDF.

    Args:
        time_var: Name of the censoring time variable
        rate: Name of the rate parameter

    Returns:
        S-expression for log-likelihood contribution
    """
    return ["log", ["-", 1, ["exp", ["*", -1, ["*", rate, time_var]]]]]


def interval_censored_exponential(
    lower_var: str = "t_lower",
    upper_var: str = "t_upper",
    rate: str = "lambda"
) -> ExprType:
    """
    Log-likelihood contribution for interval-censored exponential observation.

    For failure in interval (t_lower, t_upper]:
        log[S(t_lower) - S(t_upper)] = log[exp(-lambda*t_lower) - exp(-lambda*t_upper)]

    Args:
        lower_var: Name of lower bound time variable
        upper_var: Name of upper bound time variable
        rate: Name of the rate parameter

    Returns:
        S-expression for log-likelihood contribution
    """
    s_lower = ["exp", ["*", -1, ["*", rate, lower_var]]]
    s_upper = ["exp", ["*", -1, ["*", rate, upper_var]]]
    return ["log", ["-", s_lower, s_upper]]


# ============================================================
# Weibull Distribution Contributions
# ============================================================

def complete_weibull(
    time_var: str = "t",
    shape: str = "k",
    scale: str = "lambda"
) -> ExprType:
    """
    Log-likelihood contribution for complete Weibull observation.

    log f(t|k,lambda) = log(k) - k*log(lambda) + (k-1)*log(t) - (t/lambda)^k

    Args:
        time_var: Name of time variable
        shape: Name of shape parameter (k)
        scale: Name of scale parameter (lambda)

    Returns:
        S-expression for log-likelihood contribution
    """
    return ["+",
            ["+",
             ["log", shape],
             ["*", -1, ["*", shape, ["log", scale]]]],
            ["+",
             ["*", ["-", shape, 1], ["log", time_var]],
             ["*", -1, ["^", ["/", time_var, scale], shape]]]]


def right_censored_weibull(
    time_var: str = "t",
    shape: str = "k",
    scale: str = "lambda"
) -> ExprType:
    """
    Log-likelihood contribution for right-censored Weibull observation.

    log S(t|k,lambda) = -(t/lambda)^k

    Args:
        time_var: Name of time variable
        shape: Name of shape parameter (k)
        scale: Name of scale parameter (lambda)

    Returns:
        S-expression for log-likelihood contribution
    """
    return ["*", -1, ["^", ["/", time_var, scale], shape]]


# ============================================================
# Normal Distribution Contributions
# ============================================================

def complete_normal(
    data_var: str = "x",
    mean: str = "mu",
    var: str = "sigma2"
) -> ExprType:
    """
    Log-likelihood contribution for complete normal observation.

    log f(x|mu,sigma2) = -0.5*log(2*pi*sigma2) - (x-mu)^2/(2*sigma2)

    Args:
        data_var: Name of data variable
        mean: Name of mean parameter
        var: Name of variance parameter

    Returns:
        S-expression for log-likelihood contribution
    """
    return ["+",
            ["*", -0.5, ["log", ["*", 2, ["*", 3.141592653589793, var]]]],
            ["*", -0.5, ["*", ["/", 1, var], ["^", ["-", data_var, mean], 2]]]]


def left_truncated_normal(
    data_var: str = "x",
    truncation: str = "a",
    mean: str = "mu",
    var: str = "sigma2"
) -> ExprType:
    """
    Log-likelihood contribution for left-truncated normal observation.

    log f(x|mu,sigma2,a) = log f(x) - log(1 - Phi((a-mu)/sigma))

    This is a simplified version that doesn't include the truncation adjustment.
    For full truncation handling, use numerical methods.

    Args:
        data_var: Name of data variable
        truncation: Name of truncation point variable
        mean: Name of mean parameter
        var: Name of variance parameter

    Returns:
        S-expression for (unadjusted) log-likelihood contribution
    """
    # Note: Full implementation would require the normal CDF (Phi)
    # This returns just the density part
    return complete_normal(data_var, mean, var)


# ============================================================
# Poisson Distribution Contributions
# ============================================================

def complete_poisson(count_var: str = "k", rate: str = "lambda") -> ExprType:
    """
    Log-likelihood contribution for Poisson observation (ignoring k!).

    log f(k|lambda) = k*log(lambda) - lambda - log(k!)
                    ~ k*log(lambda) - lambda  (ignoring factorial)

    Args:
        count_var: Name of count variable
        rate: Name of rate parameter

    Returns:
        S-expression for log-likelihood contribution
    """
    return ["+",
            ["*", count_var, ["log", rate]],
            ["*", -1, rate]]


# ============================================================
# Bernoulli Distribution Contributions
# ============================================================

def complete_bernoulli(outcome_var: str = "x", prob: str = "p") -> ExprType:
    """
    Log-likelihood contribution for Bernoulli observation.

    log f(x|p) = x*log(p) + (1-x)*log(1-p)

    Args:
        outcome_var: Name of outcome variable (0 or 1)
        prob: Name of success probability parameter

    Returns:
        S-expression for log-likelihood contribution
    """
    return ["+",
            ["*", outcome_var, ["log", prob]],
            ["*", ["-", 1, outcome_var], ["log", ["-", 1, prob]]]]


# ============================================================
# Series System Contributions (Masked Component Cause)
# ============================================================
#
# For series systems under C1, C2, C3 conditions:
#   C1: True cause is always in the candidate set
#   C2: Candidate set probability independent of which component in set failed
#   C3: Masking probabilities independent of parameter vector θ
#
# See: Usher & Hodgson (1988), Miyakawa (1984)

def series_exponential_known_cause(
    time_var: str = "t",
    rates: List[str] = None,
    cause_index: int = 0,
) -> ExprType:
    """
    Log-likelihood for series system with known failure cause (no masking).

    For a series system with m exponential components with rates λ₁,...,λₘ,
    when component j causes failure at time t:
        log L = -t * Σᵢλᵢ + log(λⱼ)

    Args:
        time_var: Name of failure time variable
        rates: List of rate parameter names (e.g., ["lambda1", "lambda2", "lambda3"])
        cause_index: Index (0-based) of the component that caused failure

    Returns:
        S-expression for log-likelihood contribution
    """
    if rates is None:
        rates = ["lambda1", "lambda2", "lambda3"]

    # System survival: -t * Σλᵢ
    rate_sum = rates[0] if len(rates) == 1 else ["+"] + list(rates)
    survival = ["*", -1, ["*", time_var, rate_sum]]

    # Hazard of failing component: log(λⱼ)
    hazard = ["log", rates[cause_index]]

    return ["+", survival, hazard]


def series_exponential_masked_cause(
    time_var: str = "t",
    rates: List[str] = None,
    candidate_indices: List[int] = None,
) -> ExprType:
    """
    Log-likelihood for series system with masked failure cause under C1, C2, C3.

    For a series system with m exponential components with rates λ₁,...,λₘ,
    when failure at time t has candidate set C (indices of possible causes):
        log L = -t * Σᵢλᵢ + log(Σⱼ∈C λⱼ)

    Under C1, C2, C3 conditions, the masking probability factors out and
    doesn't affect the MLE.

    Args:
        time_var: Name of failure time variable
        rates: List of rate parameter names (e.g., ["lambda1", "lambda2", "lambda3"])
        candidate_indices: List of indices (0-based) of components in candidate set

    Returns:
        S-expression for log-likelihood contribution
    """
    if rates is None:
        rates = ["lambda1", "lambda2", "lambda3"]
    if candidate_indices is None:
        candidate_indices = list(range(len(rates)))

    # System survival: -t * Σλᵢ (all components)
    rate_sum = rates[0] if len(rates) == 1 else ["+"] + list(rates)
    survival = ["*", -1, ["*", time_var, rate_sum]]

    # Hazard sum over candidate set: log(Σⱼ∈C λⱼ)
    candidate_rates = [rates[i] for i in candidate_indices]
    if len(candidate_rates) == 1:
        hazard_sum = candidate_rates[0]
    else:
        hazard_sum = ["+"] + candidate_rates
    hazard = ["log", hazard_sum]

    return ["+", survival, hazard]


def series_exponential_right_censored(
    time_var: str = "t",
    rates: List[str] = None,
) -> ExprType:
    """
    Log-likelihood for right-censored series system (no failure observed).

    For a series system with m exponential components right-censored at time t:
        log S(t) = -t * Σᵢλᵢ

    Args:
        time_var: Name of censoring time variable
        rates: List of rate parameter names (e.g., ["lambda1", "lambda2", "lambda3"])

    Returns:
        S-expression for log-likelihood contribution
    """
    if rates is None:
        rates = ["lambda1", "lambda2", "lambda3"]

    # System survival: -t * Σλᵢ
    rate_sum = rates[0] if len(rates) == 1 else ["+"] + list(rates)
    return ["*", -1, ["*", time_var, rate_sum]]
