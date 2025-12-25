"""
Mixture and latent variable models with closed-form marginal likelihoods.

These models involve latent variables that can be marginalized out analytically:
- Zero-inflated models (ZIP, ZINB)
- Hurdle models
- Simple finite mixtures (2-component)

For models requiring numerical integration or EM, see external tools.
"""

from typing import Optional
from .model import LikelihoodModel
from .evaluate import ExprType


def zero_inflated_poisson(
    data_var: str = "y",
    zero_prob: str = "pzero",
    rate: str = "lambda"
) -> LikelihoodModel:
    """
    Zero-inflated Poisson (ZIP) model.

    This model accounts for excess zeros by assuming data comes from a mixture:
    - With probability π, Y = 0 (structural zero)
    - With probability (1-π), Y ~ Poisson(λ)

    P(Y=0) = π + (1-π)·e^(-λ)
    P(Y=k) = (1-π)·λᵏe^(-λ)/k!  for k > 0

    ℓ(π, λ) = Σᵢ [I(yᵢ=0)·log(π + (1-π)e^(-λ))
                 + I(yᵢ>0)·(log(1-π) + yᵢ·log(λ) - λ)]

    Args:
        data_var: Name of count variable (non-negative integers)
        zero_prob: Name of zero-inflation probability parameter (0 < π < 1)
            Default is "pzero" (not "pi" to avoid conflict with math constant)
        rate: Name of Poisson rate parameter (λ > 0)

    Returns:
        LikelihoodModel for zero-inflated Poisson

    Example:
        >>> model = zero_inflated_poisson()
        >>> # Data with excess zeros
        >>> data = {"y": [0, 0, 0, 0, 1, 2, 0, 3, 0, 1]}
        >>> mle, _ = model.mle(
        ...     data=data,
        ...     init={"pzero": 0.3, "lambda": 1.0},
        ...     bounds={"pzero": (0.01, 0.99), "lambda": (0.01, None)}
        ... )
    """
    n = ["len", data_var]

    # Contribution for y=0: log(π + (1-π)·exp(-λ))
    contrib_zero = ["log", ["+", zero_prob,
                            ["*", ["-", 1, zero_prob],
                             ["exp", ["*", -1, rate]]]]]

    # Contribution for y>0: log(1-π) + y·log(λ) - λ
    # (ignoring log(y!) since it doesn't depend on parameters)
    contrib_pos = ["+",
                   ["log", ["-", 1, zero_prob]],
                   ["+",
                    ["*", ["@", data_var, "i"], ["log", rate]],
                    ["*", -1, rate]]]

    # Use 'if' to select: if y[i] is truthy (non-zero) use contrib_pos, else contrib_zero
    contrib_i = ["if", ["@", data_var, "i"], contrib_pos, contrib_zero]

    log_lik = ["sum", "i", n, contrib_i]

    return LikelihoodModel(log_lik, [zero_prob, rate])


def zero_inflated_negative_binomial(
    data_var: str = "y",
    zero_prob: str = "pzero",
    r: str = "r",
    p: str = "p"
) -> LikelihoodModel:
    """
    Zero-inflated Negative Binomial (ZINB) model.

    Accounts for excess zeros in overdispersed count data:
    - With probability π, Y = 0 (structural zero)
    - With probability (1-π), Y ~ NegBin(r, p)

    P(Y=0) = π + (1-π)·p^r
    P(Y=k) = (1-π)·C(k+r-1, k)·p^r·(1-p)^k  for k > 0

    Args:
        data_var: Name of count variable
        zero_prob: Name of zero-inflation probability (0 < π < 1)
            Default is "pzero" (not "pi" to avoid conflict with math constant)
        r: Name of NegBin size parameter (r > 0)
        p: Name of NegBin probability parameter (0 < p < 1)

    Returns:
        LikelihoodModel for zero-inflated negative binomial

    Example:
        >>> model = zero_inflated_negative_binomial()
        >>> data = {"y": [0, 0, 0, 1, 5, 0, 2, 0, 8, 0]}
        >>> mle, _ = model.mle(
        ...     data=data,
        ...     init={"pzero": 0.3, "r": 2.0, "p": 0.5},
        ...     bounds={"pzero": (0.01, 0.99), "r": (0.1, None), "p": (0.01, 0.99)}
        ... )
    """
    n = ["len", data_var]

    # P(Y=0|NB) = p^r
    nb_zero_prob = ["^", p, r]

    # Contribution for y=0: log(π + (1-π)·p^r)
    contrib_zero = ["log", ["+", zero_prob,
                            ["*", ["-", 1, zero_prob], nb_zero_prob]]]

    # Contribution for y>0 from NegBin:
    # log(1-π) + lgamma(y+r) - lgamma(r) - lgamma(y+1) + r·log(p) + y·log(1-p)
    # Ignoring lgamma(y+1) since it doesn't depend on parameters
    y_i = ["@", data_var, "i"]
    contrib_pos = ["+",
                   ["log", ["-", 1, zero_prob]],
                   ["+",
                    ["lgamma", ["+", y_i, r]],
                    ["+",
                     ["*", -1, ["lgamma", r]],
                     ["+",
                      ["*", r, ["log", p]],
                      ["*", y_i, ["log", ["-", 1, p]]]]]]]

    contrib_i = ["if", y_i, contrib_pos, contrib_zero]

    log_lik = ["sum", "i", n, contrib_i]

    return LikelihoodModel(log_lik, [zero_prob, r, p])


def hurdle_poisson(
    data_var: str = "y",
    zero_prob: str = "pzero",
    rate: str = "lambda"
) -> LikelihoodModel:
    """
    Hurdle Poisson model.

    Unlike ZIP, the hurdle model treats zeros and positives as separate processes:
    - P(Y=0) = π
    - P(Y=k|Y>0) ∝ Poisson(λ) for k > 0

    This is equivalent to:
    - P(Y=0) = π
    - P(Y=k) = (1-π) · λᵏe^(-λ) / (1 - e^(-λ)) for k > 0

    The key difference from ZIP: in hurdle models, zeros only come from the
    "hurdle" process, not from the count process.

    Args:
        data_var: Name of count variable
        zero_prob: Name of hurdle probability (0 < π < 1)
            Default is "pzero" (not "pi" to avoid conflict with math constant)
        rate: Name of truncated Poisson rate (λ > 0)

    Returns:
        LikelihoodModel for hurdle Poisson
    """
    n = ["len", data_var]

    # Contribution for y=0: log(π)
    contrib_zero = ["log", zero_prob]

    # Contribution for y>0: log(1-π) + y·log(λ) - λ - log(1 - e^(-λ))
    # The last term normalizes the truncated Poisson
    y_i = ["@", data_var, "i"]
    contrib_pos = ["+",
                   ["log", ["-", 1, zero_prob]],
                   ["+",
                    ["*", y_i, ["log", rate]],
                    ["+",
                     ["*", -1, rate],
                     ["*", -1, ["log", ["-", 1, ["exp", ["*", -1, rate]]]]]]]]

    contrib_i = ["if", y_i, contrib_pos, contrib_zero]

    log_lik = ["sum", "i", n, contrib_i]

    return LikelihoodModel(log_lik, [zero_prob, rate])


def hurdle_negative_binomial(
    data_var: str = "y",
    zero_prob: str = "pzero",
    r: str = "r",
    p: str = "p"
) -> LikelihoodModel:
    """
    Hurdle Negative Binomial model.

    - P(Y=0) = π
    - P(Y=k) = (1-π) · NB(k|r,p) / (1 - p^r) for k > 0

    Args:
        data_var: Name of count variable
        zero_prob: Name of hurdle probability
            Default is "pzero" (not "pi" to avoid conflict with math constant)
        r: Name of NegBin size parameter
        p: Name of NegBin probability parameter

    Returns:
        LikelihoodModel for hurdle negative binomial
    """
    n = ["len", data_var]

    # P(Y=0|NB) = p^r
    nb_zero_prob = ["^", p, r]

    # Contribution for y=0: log(π)
    contrib_zero = ["log", zero_prob]

    # Contribution for y>0:
    # log(1-π) + lgamma(y+r) - lgamma(r) + r·log(p) + y·log(1-p) - log(1 - p^r)
    y_i = ["@", data_var, "i"]
    contrib_pos = ["+",
                   ["log", ["-", 1, zero_prob]],
                   ["+",
                    ["lgamma", ["+", y_i, r]],
                    ["+",
                     ["*", -1, ["lgamma", r]],
                     ["+",
                      ["*", r, ["log", p]],
                      ["+",
                       ["*", y_i, ["log", ["-", 1, p]]],
                       ["*", -1, ["log", ["-", 1, nb_zero_prob]]]]]]]]

    contrib_i = ["if", y_i, contrib_pos, contrib_zero]

    log_lik = ["sum", "i", n, contrib_i]

    return LikelihoodModel(log_lik, [zero_prob, r, p])


def mixture_exponential(
    data_var: str = "x",
    mixing_prob: str = "omega",
    rate1: str = "lambda1",
    rate2: str = "lambda2"
) -> LikelihoodModel:
    """
    Two-component exponential mixture model.

    f(x) = ω·λ₁·e^(-λ₁x) + (1-ω)·λ₂·e^(-λ₂x)

    ℓ(ω, λ₁, λ₂) = Σᵢ log(ω·λ₁·e^(-λ₁xᵢ) + (1-ω)·λ₂·e^(-λ₂xᵢ))

    Note: This involves log-sum which can be numerically unstable.
    For better stability with extreme values, consider using log-sum-exp tricks.

    Args:
        data_var: Name of data variable (positive values)
        mixing_prob: Name of mixing probability (0 < ω < 1)
            Default is "omega" (not "pi" to avoid conflict with math constant)
        rate1: Name of first component rate
        rate2: Name of second component rate

    Returns:
        LikelihoodModel for 2-component exponential mixture
    """
    n = ["len", data_var]
    x_i = ["@", data_var, "i"]

    # f₁(x) = λ₁·exp(-λ₁x)
    f1 = ["*", rate1, ["exp", ["*", -1, ["*", rate1, x_i]]]]

    # f₂(x) = λ₂·exp(-λ₂x)
    f2 = ["*", rate2, ["exp", ["*", -1, ["*", rate2, x_i]]]]

    # log(π·f₁ + (1-π)·f₂)
    contrib_i = ["log", ["+",
                         ["*", mixing_prob, f1],
                         ["*", ["-", 1, mixing_prob], f2]]]

    log_lik = ["sum", "i", n, contrib_i]

    return LikelihoodModel(log_lik, [mixing_prob, rate1, rate2])


def mixture_normal(
    data_var: str = "x",
    mixing_prob: str = "omega",
    mu1: str = "mu1",
    mu2: str = "mu2",
    sigma2: str = "sigma2"
) -> LikelihoodModel:
    """
    Two-component normal mixture with equal variances.

    f(x) = ω·φ(x|μ₁,σ²) + (1-ω)·φ(x|μ₂,σ²)

    where φ(x|μ,σ²) = (2πσ²)^(-1/2) exp(-(x-μ)²/(2σ²))

    Note: Equal variance assumption simplifies computation and identifiability.

    Args:
        data_var: Name of data variable
        mixing_prob: Name of mixing probability (0 < ω < 1)
            Default is "omega" (not "pi" to avoid conflict with math constant)
        mu1: Name of first component mean
        mu2: Name of second component mean
        sigma2: Name of common variance

    Returns:
        LikelihoodModel for 2-component normal mixture (equal variance)

    Example:
        >>> model = mixture_normal()
        >>> # Bimodal data
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> data = np.concatenate([np.random.normal(0, 1, 50),
        ...                        np.random.normal(5, 1, 50)])
        >>> mle, _ = model.mle(
        ...     data={"x": data.tolist()},
        ...     init={"omega": 0.5, "mu1": -1, "mu2": 6, "sigma2": 2},
        ...     bounds={"omega": (0.1, 0.9), "sigma2": (0.1, None)}
        ... )
    """
    n = ["len", data_var]
    x_i = ["@", data_var, "i"]

    # φ(x|μ,σ²) = (2πσ²)^(-1/2) exp(-(x-μ)²/(2σ²))
    # = exp(-0.5*log(2πσ²) - (x-μ)²/(2σ²))
    # For numerical stability, we compute log(π·φ₁ + (1-π)·φ₂) directly

    # Components (unnormalized for numerical stability)
    # log(φᵢ) = -0.5·log(2πσ²) - (x-μᵢ)²/(2σ²)
    inv_2sigma2 = ["/", 1, ["*", 2, sigma2]]
    log_norm_const = ["*", -0.5, ["log", ["*", 2, ["*", 3.141592653589793, sigma2]]]]

    # φ₁(x) and φ₂(x) - the actual densities
    phi1 = ["exp", ["+", log_norm_const,
                    ["*", -1, ["*", inv_2sigma2,
                               ["^", ["-", x_i, mu1], 2]]]]]
    phi2 = ["exp", ["+", log_norm_const,
                    ["*", -1, ["*", inv_2sigma2,
                               ["^", ["-", x_i, mu2], 2]]]]]

    # log(π·φ₁ + (1-π)·φ₂)
    contrib_i = ["log", ["+",
                         ["*", mixing_prob, phi1],
                         ["*", ["-", 1, mixing_prob], phi2]]]

    log_lik = ["sum", "i", n, contrib_i]

    return LikelihoodModel(log_lik, [mixing_prob, mu1, mu2, sigma2])
