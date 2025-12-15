"""
Convenience constructors for common statistical distributions.

Each function returns a LikelihoodModel with the appropriate log-likelihood
expression for the given distribution family.
"""

from .model import LikelihoodModel
from .evaluate import ExprType


def exponential(data_var: str = "x", param: str = "lambda") -> LikelihoodModel:
    """
    Exponential distribution likelihood model.

    ℓ(λ) = Σᵢ [log(λ) - λxᵢ] = n·log(λ) - λ·Σxᵢ

    MLE: λ̂ = 1/x̄

    Args:
        data_var: Name of data variable
        param: Name of rate parameter

    Returns:
        LikelihoodModel for exponential distribution

    Example:
        >>> model = exponential()
        >>> mle, _ = model.mle(data={'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})
        >>> # mle ≈ {'lambda': 0.333}  (1/mean)
    """
    log_lik = [
        "sum", "i", ["len", data_var],
        ["+",
         ["log", param],
         ["*", -1, ["*", param, ["@", data_var, "i"]]]]
    ]
    return LikelihoodModel(log_lik, [param])


def normal(data_var: str = "x", mean: str = "mu", var: str = "sigma2") -> LikelihoodModel:
    """
    Normal distribution likelihood model (known variance parameterization).

    ℓ(μ, σ²) = -n/2·log(2πσ²) - 1/(2σ²)·Σ(xᵢ - μ)²

    MLE: μ̂ = x̄, σ̂² = Σ(xᵢ - x̄)²/n

    Args:
        data_var: Name of data variable
        mean: Name of mean parameter
        var: Name of variance parameter

    Returns:
        LikelihoodModel for normal distribution

    Example:
        >>> model = normal()
        >>> mle, _ = model.mle(
        ...     data={'x': [1, 2, 3, 4, 5]},
        ...     init={'mu': 0.0, 'sigma2': 1.0},
        ...     bounds={'sigma2': (0.01, None)}
        ... )
    """
    # ℓ = -n/2 * log(2π) - n/2 * log(σ²) - 1/(2σ²) * Σ(x - μ)²
    n = ["len", data_var]
    log_lik = [
        "+",
        ["*", -0.5, ["*", n, ["log", ["*", 2, 3.141592653589793]]]],
        ["+",
         ["*", -0.5, ["*", n, ["log", var]]],
         ["*", -0.5,
          ["*", ["/", 1, var],
           ["sum", "i", n,
            ["^", ["-", ["@", data_var, "i"], mean], 2]]]]]
    ]
    return LikelihoodModel(log_lik, [mean, var])


def normal_mean(data_var: str = "x", mean: str = "mu", known_var: float = 1.0) -> LikelihoodModel:
    """
    Normal distribution with known variance (estimate mean only).

    ℓ(μ) = -1/(2σ²)·Σ(xᵢ - μ)²  (ignoring constants)

    Args:
        data_var: Name of data variable
        mean: Name of mean parameter
        known_var: Known variance value

    Returns:
        LikelihoodModel for normal with known variance
    """
    log_lik = [
        "*", -0.5,
        ["*", ["/", 1, known_var],
         ["sum", "i", ["len", data_var],
          ["^", ["-", ["@", data_var, "i"], mean], 2]]]
    ]
    return LikelihoodModel(log_lik, [mean])


def poisson(data_var: str = "x", param: str = "lambda") -> LikelihoodModel:
    """
    Poisson distribution likelihood model.

    ℓ(λ) = Σᵢ [xᵢ·log(λ) - λ - log(xᵢ!)]
         ≈ Σxᵢ·log(λ) - n·λ  (ignoring factorial)

    MLE: λ̂ = x̄

    Args:
        data_var: Name of data variable
        param: Name of rate parameter

    Returns:
        LikelihoodModel for Poisson distribution

    Example:
        >>> model = poisson()
        >>> mle, _ = model.mle(data={'x': [1, 2, 3, 2, 1]}, init={'lambda': 1.0})
        >>> # mle ≈ {'lambda': 1.8}  (mean)
    """
    # Ignoring log(x!) since it doesn't depend on λ
    log_lik = [
        "+",
        ["*", ["total", data_var], ["log", param]],
        ["*", -1, ["*", ["len", data_var], param]]
    ]
    return LikelihoodModel(log_lik, [param])


def bernoulli(data_var: str = "x", param: str = "p") -> LikelihoodModel:
    """
    Bernoulli distribution likelihood model.

    ℓ(p) = Σᵢ [xᵢ·log(p) + (1-xᵢ)·log(1-p)]
         = k·log(p) + (n-k)·log(1-p)

    where k = Σxᵢ (number of successes)

    MLE: p̂ = k/n

    Args:
        data_var: Name of data variable (0/1 values)
        param: Name of success probability parameter

    Returns:
        LikelihoodModel for Bernoulli distribution
    """
    # k = total(x), n = len(x)
    k = ["total", data_var]
    n = ["len", data_var]
    log_lik = [
        "+",
        ["*", k, ["log", param]],
        ["*", ["-", n, k], ["log", ["-", 1, param]]]
    ]
    return LikelihoodModel(log_lik, [param])


def binomial(successes: str = "k", trials: str = "n", param: str = "p") -> LikelihoodModel:
    """
    Binomial distribution likelihood model (single observation).

    ℓ(p) = k·log(p) + (n-k)·log(1-p)  (ignoring binomial coefficient)

    Args:
        successes: Name of success count variable
        trials: Name of trials count variable
        param: Name of success probability parameter

    Returns:
        LikelihoodModel for binomial distribution
    """
    log_lik = [
        "+",
        ["*", successes, ["log", param]],
        ["*", ["-", trials, successes], ["log", ["-", 1, param]]]
    ]
    return LikelihoodModel(log_lik, [param])


def gamma(data_var: str = "x", shape: str = "alpha", rate: str = "beta") -> LikelihoodModel:
    """
    Gamma distribution likelihood model.

    ℓ(α, β) = n·α·log(β) - n·log(Γ(α)) + (α-1)·Σlog(xᵢ) - β·Σxᵢ

    Args:
        data_var: Name of data variable
        shape: Name of shape parameter (α)
        rate: Name of rate parameter (β)

    Returns:
        LikelihoodModel for gamma distribution

    Note:
        This uses the rate parameterization f(x) = β^α/Γ(α) · x^(α-1) · e^(-βx)
    """
    n = ["len", data_var]
    sum_log_x = ["sum", "i", n, ["log", ["@", data_var, "i"]]]
    sum_x = ["total", data_var]

    log_lik = [
        "+",
        ["*", n, ["*", shape, ["log", rate]]],
        ["+",
         ["*", -1, ["*", n, ["lgamma", shape]]],
         ["+",
          ["*", ["-", shape, 1], sum_log_x],
          ["*", -1, ["*", rate, sum_x]]]]
    ]
    return LikelihoodModel(log_lik, [shape, rate])


def weibull(data_var: str = "x", shape: str = "k", scale: str = "lambda") -> LikelihoodModel:
    """
    Weibull distribution likelihood model.

    ℓ(k, λ) = n·log(k) - n·k·log(λ) + (k-1)·Σlog(xᵢ) - Σ(xᵢ/λ)^k

    Args:
        data_var: Name of data variable
        shape: Name of shape parameter (k)
        scale: Name of scale parameter (λ)

    Returns:
        LikelihoodModel for Weibull distribution
    """
    n = ["len", data_var]
    sum_log_x = ["sum", "i", n, ["log", ["@", data_var, "i"]]]
    sum_xk = ["sum", "i", n, ["^", ["/", ["@", data_var, "i"], scale], shape]]

    log_lik = [
        "+",
        ["*", n, ["log", shape]],
        ["+",
         ["*", -1, ["*", n, ["*", shape, ["log", scale]]]],
         ["+",
          ["*", ["-", shape, 1], sum_log_x],
          ["*", -1, sum_xk]]]
    ]
    return LikelihoodModel(log_lik, [shape, scale])


def beta(data_var: str = "x", alpha: str = "alpha", beta_param: str = "beta") -> LikelihoodModel:
    """
    Beta distribution likelihood model.

    ℓ(α, β) = n·[log(Γ(α+β)) - log(Γ(α)) - log(Γ(β))]
            + (α-1)·Σlog(xᵢ) + (β-1)·Σlog(1-xᵢ)

    Args:
        data_var: Name of data variable (values in (0,1))
        alpha: Name of first shape parameter
        beta_param: Name of second shape parameter

    Returns:
        LikelihoodModel for beta distribution
    """
    n = ["len", data_var]
    sum_log_x = ["sum", "i", n, ["log", ["@", data_var, "i"]]]
    sum_log_1mx = ["sum", "i", n, ["log", ["-", 1, ["@", data_var, "i"]]]]

    log_lik = [
        "+",
        ["*", n,
         ["+",
          ["lgamma", ["+", alpha, beta_param]],
          ["+",
           ["*", -1, ["lgamma", alpha]],
           ["*", -1, ["lgamma", beta_param]]]]],
        ["+",
         ["*", ["-", alpha, 1], sum_log_x],
         ["*", ["-", beta_param, 1], sum_log_1mx]]
    ]
    return LikelihoodModel(log_lik, [alpha, beta_param])
