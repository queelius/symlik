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


# ============================================================
# Additional Distribution Families
# ============================================================


def lognormal(data_var: str = "x", mu: str = "mu", sigma2: str = "sigma2") -> LikelihoodModel:
    """
    Log-normal distribution likelihood model.

    If X ~ LogNormal(μ, σ²), then log(X) ~ Normal(μ, σ²).

    ℓ(μ, σ²) = -n/2·log(2πσ²) - Σlog(xᵢ) - 1/(2σ²)·Σ(log(xᵢ) - μ)²

    MLE: μ̂ = Σlog(xᵢ)/n, σ̂² = Σ(log(xᵢ) - μ̂)²/n

    Args:
        data_var: Name of data variable (positive values)
        mu: Name of log-mean parameter
        sigma2: Name of log-variance parameter

    Returns:
        LikelihoodModel for log-normal distribution

    Example:
        >>> model = lognormal()
        >>> mle, _ = model.mle(
        ...     data={'x': [1.2, 2.5, 1.8, 3.1, 2.0]},
        ...     init={'mu': 0.0, 'sigma2': 1.0},
        ...     bounds={'sigma2': (0.01, None)}
        ... )
    """
    n = ["len", data_var]
    sum_log_x = ["sum", "i", n, ["log", ["@", data_var, "i"]]]
    sum_log_x_minus_mu_sq = ["sum", "i", n,
                             ["^", ["-", ["log", ["@", data_var, "i"]], mu], 2]]

    log_lik = [
        "+",
        ["*", -0.5, ["*", n, ["log", ["*", 2, 3.141592653589793]]]],
        ["+",
         ["*", -0.5, ["*", n, ["log", sigma2]]],
         ["+",
          ["*", -1, sum_log_x],
          ["*", -0.5, ["*", ["/", 1, sigma2], sum_log_x_minus_mu_sq]]]]
    ]
    return LikelihoodModel(log_lik, [mu, sigma2])


def negative_binomial(
    data_var: str = "x",
    r: str = "r",
    p: str = "p"
) -> LikelihoodModel:
    """
    Negative binomial distribution likelihood model.

    Models the number of failures before r successes, with success probability p.
    Useful for overdispersed count data (variance > mean).

    ℓ(r, p) = Σ[log(Γ(xᵢ + r)) - log(Γ(r)) - log(xᵢ!) + r·log(p) + xᵢ·log(1-p)]
            ≈ Σ[lgamma(xᵢ + r) - lgamma(r) + r·log(p) + xᵢ·log(1-p)]

    Mean: r(1-p)/p, Variance: r(1-p)/p²

    Args:
        data_var: Name of data variable (non-negative integers)
        r: Name of number of successes parameter (r > 0)
        p: Name of success probability parameter (0 < p < 1)

    Returns:
        LikelihoodModel for negative binomial distribution

    Example:
        >>> model = negative_binomial()
        >>> mle, _ = model.mle(
        ...     data={'x': [2, 5, 3, 8, 1, 4, 6]},
        ...     init={'r': 2.0, 'p': 0.5},
        ...     bounds={'r': (0.01, None), 'p': (0.01, 0.99)}
        ... )
    """
    n = ["len", data_var]
    # Σ lgamma(xᵢ + r)
    sum_lgamma_xr = ["sum", "i", n, ["lgamma", ["+", ["@", data_var, "i"], r]]]
    # Σ xᵢ
    sum_x = ["total", data_var]

    log_lik = [
        "+",
        sum_lgamma_xr,
        ["+",
         ["*", -1, ["*", n, ["lgamma", r]]],
         ["+",
          ["*", ["*", n, r], ["log", p]],
          ["*", sum_x, ["log", ["-", 1, p]]]]]
    ]
    return LikelihoodModel(log_lik, [r, p])


def student_t(
    data_var: str = "x",
    mu: str = "mu",
    sigma2: str = "sigma2",
    nu: str = "nu"
) -> LikelihoodModel:
    """
    Student's t distribution likelihood model (location-scale parameterization).

    Useful for robust inference with heavy-tailed data.

    ℓ(μ, σ², ν) = Σ[log(Γ((ν+1)/2)) - log(Γ(ν/2)) - 0.5·log(νπσ²)
                   - ((ν+1)/2)·log(1 + (xᵢ-μ)²/(νσ²))]

    As ν → ∞, approaches Normal(μ, σ²).
    ν = 1 gives Cauchy distribution.

    Args:
        data_var: Name of data variable
        mu: Name of location parameter
        sigma2: Name of scale parameter (σ²)
        nu: Name of degrees of freedom parameter (ν > 0)

    Returns:
        LikelihoodModel for Student's t distribution

    Example:
        >>> model = student_t()
        >>> mle, _ = model.mle(
        ...     data={'x': [1.0, 2.5, -0.5, 15.0, 3.0]},  # Note outlier
        ...     init={'mu': 0.0, 'sigma2': 1.0, 'nu': 5.0},
        ...     bounds={'sigma2': (0.01, None), 'nu': (1.0, None)}
        ... )
    """
    n = ["len", data_var]
    nu_plus_1_half = ["*", 0.5, ["+", nu, 1]]
    nu_half = ["*", 0.5, nu]

    # log(Γ((ν+1)/2)) - log(Γ(ν/2))
    lgamma_diff = ["+", ["lgamma", nu_plus_1_half], ["*", -1, ["lgamma", nu_half]]]

    # -0.5·log(νπσ²)
    log_scale = ["*", -0.5, ["log", ["*", nu, ["*", 3.141592653589793, sigma2]]]]

    # Σ -((ν+1)/2)·log(1 + (xᵢ-μ)²/(νσ²))
    sum_log_terms = [
        "sum", "i", n,
        ["*", ["*", -1, nu_plus_1_half],
         ["log", ["+", 1,
                  ["/", ["^", ["-", ["@", data_var, "i"], mu], 2],
                   ["*", nu, sigma2]]]]]
    ]

    log_lik = [
        "+",
        ["*", n, lgamma_diff],
        ["+",
         ["*", n, log_scale],
         sum_log_terms]
    ]
    return LikelihoodModel(log_lik, [mu, sigma2, nu])


def uniform(
    data_var: str = "x",
    a: str = "a",
    b: str = "b"
) -> LikelihoodModel:
    """
    Uniform distribution likelihood model.

    ℓ(a, b) = -n·log(b - a)  if all xᵢ ∈ [a, b], else -∞

    Note: The constraint a ≤ min(x) and b ≥ max(x) must be enforced
    via bounds during optimization. The MLE is â = min(x), b̂ = max(x).

    Args:
        data_var: Name of data variable
        a: Name of lower bound parameter
        b: Name of upper bound parameter

    Returns:
        LikelihoodModel for uniform distribution

    Example:
        >>> model = uniform()
        >>> data = [0.2, 0.5, 0.8, 0.3, 0.9]
        >>> mle, _ = model.mle(
        ...     data={'x': data},
        ...     init={'a': 0.0, 'b': 1.0},
        ...     bounds={'a': (None, min(data)), 'b': (max(data), None)}
        ... )
    """
    n = ["len", data_var]
    log_lik = ["*", -1, ["*", n, ["log", ["-", b, a]]]]
    return LikelihoodModel(log_lik, [a, b])


def laplace(
    data_var: str = "x",
    mu: str = "mu",
    b: str = "b"
) -> LikelihoodModel:
    """
    Laplace (double exponential) distribution likelihood model.

    ℓ(μ, b) = -n·log(2b) - (1/b)·Σ|xᵢ - μ|

    MLE: μ̂ = median(x), b̂ = Σ|xᵢ - μ̂|/n

    The Laplace distribution is equivalent to L1 loss in regression,
    making it robust to outliers.

    Args:
        data_var: Name of data variable
        mu: Name of location parameter (median)
        b: Name of scale parameter (b > 0)

    Returns:
        LikelihoodModel for Laplace distribution

    Example:
        >>> model = laplace()
        >>> mle, _ = model.mle(
        ...     data={'x': [1.0, 2.0, 3.0, 100.0]},  # Note outlier
        ...     init={'mu': 1.0, 'b': 1.0},
        ...     bounds={'b': (0.01, None)}
        ... )
    """
    n = ["len", data_var]
    sum_abs_dev = ["sum", "i", n, ["abs", ["-", ["@", data_var, "i"], mu]]]

    log_lik = [
        "+",
        ["*", -1, ["*", n, ["log", ["*", 2, b]]]],
        ["*", -1, ["*", ["/", 1, b], sum_abs_dev]]
    ]
    return LikelihoodModel(log_lik, [mu, b])


def geometric(data_var: str = "x", p: str = "p") -> LikelihoodModel:
    """
    Geometric distribution likelihood model.

    Models the number of failures before the first success.

    ℓ(p) = n·log(p) + Σxᵢ·log(1-p)

    MLE: p̂ = n / (n + Σxᵢ) = 1 / (1 + x̄)

    Args:
        data_var: Name of data variable (non-negative integers)
        p: Name of success probability parameter (0 < p < 1)

    Returns:
        LikelihoodModel for geometric distribution

    Example:
        >>> model = geometric()
        >>> mle, _ = model.mle(
        ...     data={'x': [0, 2, 1, 0, 3, 1]},
        ...     init={'p': 0.5},
        ...     bounds={'p': (0.01, 0.99)}
        ... )
    """
    n = ["len", data_var]
    sum_x = ["total", data_var]

    log_lik = [
        "+",
        ["*", n, ["log", p]],
        ["*", sum_x, ["log", ["-", 1, p]]]
    ]
    return LikelihoodModel(log_lik, [p])


def pareto(
    data_var: str = "x",
    alpha: str = "alpha",
    x_min: str = "x_min"
) -> LikelihoodModel:
    """
    Pareto distribution likelihood model.

    Models heavy-tailed data (wealth, city sizes, etc.).

    ℓ(α, xₘ) = n·log(α) + n·α·log(xₘ) - (α+1)·Σlog(xᵢ)

    Constraint: xₘ ≤ min(x) must be enforced via bounds.

    MLE: x̂ₘ = min(x), α̂ = n / Σ(log(xᵢ) - log(x̂ₘ))

    Args:
        data_var: Name of data variable (values ≥ xₘ)
        alpha: Name of shape parameter (α > 0)
        x_min: Name of scale/minimum parameter (xₘ > 0)

    Returns:
        LikelihoodModel for Pareto distribution

    Example:
        >>> model = pareto()
        >>> data = [2.5, 3.0, 5.0, 2.1, 4.0]
        >>> mle, _ = model.mle(
        ...     data={'x': data},
        ...     init={'alpha': 2.0, 'x_min': 2.0},
        ...     bounds={'alpha': (0.01, None), 'x_min': (0.01, min(data))}
        ... )
    """
    n = ["len", data_var]
    sum_log_x = ["sum", "i", n, ["log", ["@", data_var, "i"]]]

    log_lik = [
        "+",
        ["*", n, ["log", alpha]],
        ["+",
         ["*", ["*", n, alpha], ["log", x_min]],
         ["*", -1, ["*", ["+", alpha, 1], sum_log_x]]]
    ]
    return LikelihoodModel(log_lik, [alpha, x_min])


def cauchy(
    data_var: str = "x",
    x0: str = "x0",
    gamma_param: str = "gamma"
) -> LikelihoodModel:
    """
    Cauchy distribution likelihood model.

    Heavy-tailed distribution (no finite mean or variance).
    Special case of Student's t with ν = 1.

    ℓ(x₀, γ) = -n·log(πγ) - Σlog(1 + ((xᵢ - x₀)/γ)²)

    Args:
        data_var: Name of data variable
        x0: Name of location parameter (median)
        gamma_param: Name of scale parameter (γ > 0)

    Returns:
        LikelihoodModel for Cauchy distribution

    Example:
        >>> model = cauchy()
        >>> mle, _ = model.mle(
        ...     data={'x': [0.1, -0.5, 1.2, -10.0, 0.8]},
        ...     init={'x0': 0.0, 'gamma': 1.0},
        ...     bounds={'gamma': (0.01, None)}
        ... )
    """
    n = ["len", data_var]
    sum_log_terms = [
        "sum", "i", n,
        ["log", ["+", 1,
                 ["^", ["/", ["-", ["@", data_var, "i"], x0], gamma_param], 2]]]
    ]

    log_lik = [
        "+",
        ["*", -1, ["*", n, ["log", ["*", 3.141592653589793, gamma_param]]]],
        ["*", -1, sum_log_terms]
    ]
    return LikelihoodModel(log_lik, [x0, gamma_param])


def inverse_gaussian(
    data_var: str = "x",
    mu: str = "mu",
    lambda_param: str = "lambda"
) -> LikelihoodModel:
    """
    Inverse Gaussian (Wald) distribution likelihood model.

    Models first-passage times for Brownian motion with drift.

    ℓ(μ, λ) = n/2·log(λ/(2π)) - 3/2·Σlog(xᵢ) - λ/(2μ²)·Σ(xᵢ - μ)²/xᵢ

    MLE: μ̂ = x̄, λ̂ = n / Σ(1/xᵢ - 1/x̄)

    Args:
        data_var: Name of data variable (positive values)
        mu: Name of mean parameter (μ > 0)
        lambda_param: Name of shape parameter (λ > 0)

    Returns:
        LikelihoodModel for inverse Gaussian distribution

    Example:
        >>> model = inverse_gaussian()
        >>> mle, _ = model.mle(
        ...     data={'x': [0.5, 1.2, 0.8, 1.5, 0.9]},
        ...     init={'mu': 1.0, 'lambda': 1.0},
        ...     bounds={'mu': (0.01, None), 'lambda': (0.01, None)}
        ... )
    """
    n = ["len", data_var]
    sum_log_x = ["sum", "i", n, ["log", ["@", data_var, "i"]]]
    # Σ (xᵢ - μ)² / xᵢ
    sum_quad_over_x = [
        "sum", "i", n,
        ["/", ["^", ["-", ["@", data_var, "i"], mu], 2], ["@", data_var, "i"]]
    ]

    log_lik = [
        "+",
        ["*", 0.5, ["*", n, ["log", ["/", lambda_param, ["*", 2, 3.141592653589793]]]]],
        ["+",
         ["*", -1.5, sum_log_x],
         ["*", -0.5, ["*", ["/", lambda_param, ["^", mu, 2]], sum_quad_over_x]]]
    ]
    return LikelihoodModel(log_lik, [mu, lambda_param])
