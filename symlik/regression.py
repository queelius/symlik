"""
Generalized Linear Models (GLM) for regression analysis.

Each function returns a LikelihoodModel with the appropriate log-likelihood
for the given GLM family. The linear predictor η = β₀ + Σβⱼxⱼ is specified
via predictors (data variable names) and coefficients (parameter names).

Supported models:
- linear_regression: Normal with identity link
- logistic_regression: Bernoulli with logit link
- probit_regression: Bernoulli with probit link
- poisson_regression: Poisson with log link
- gamma_regression: Gamma with log link
- negative_binomial_regression: Negative binomial with log link
"""

from typing import List, Optional
from .model import LikelihoodModel
from .evaluate import ExprType


def _build_linear_predictor(
    intercept: str,
    coefficients: List[str],
    predictors: List[str],
    index_var: str = "i"
) -> ExprType:
    """
    Build the linear predictor expression: η = β₀ + Σβⱼxⱼ[i]

    Args:
        intercept: Name of intercept parameter
        coefficients: Names of coefficient parameters
        predictors: Names of predictor data variables
        index_var: Index variable for summation context

    Returns:
        S-expression for the linear predictor at observation i
    """
    if not coefficients:
        return intercept

    # Build: intercept + coef1*pred1[i] + coef2*pred2[i] + ...
    result = intercept
    for coef, pred in zip(coefficients, predictors):
        term = ["*", coef, ["@", pred, index_var]]
        result = ["+", result, term]

    return result


def linear_regression(
    response: str = "y",
    predictors: Optional[List[str]] = None,
    intercept: str = "beta0",
    coefficients: Optional[List[str]] = None,
    var: str = "sigma2"
) -> LikelihoodModel:
    """
    Linear regression (Normal GLM with identity link).

    Model: Y ~ Normal(η, σ²) where η = β₀ + Σβⱼxⱼ

    ℓ(β, σ²) = -n/2·log(2πσ²) - 1/(2σ²)·Σ(yᵢ - ηᵢ)²

    Args:
        response: Name of response variable
        predictors: Names of predictor variables (data arrays)
        intercept: Name of intercept parameter
        coefficients: Names of coefficient parameters (auto-generated if None)
        var: Name of variance parameter

    Returns:
        LikelihoodModel for linear regression

    Example:
        >>> model = linear_regression(response="y", predictors=["x1", "x2"])
        >>> # Parameters: beta0, beta1, beta2, sigma2
        >>> mle, _ = model.mle(
        ...     data={"y": [1, 2, 3], "x1": [0.1, 0.2, 0.3], "x2": [1, 2, 3]},
        ...     init={"beta0": 0, "beta1": 0, "beta2": 0, "sigma2": 1},
        ...     bounds={"sigma2": (0.01, None)}
        ... )
    """
    if predictors is None:
        predictors = []
    if coefficients is None:
        coefficients = [f"beta{i+1}" for i in range(len(predictors))]

    if len(coefficients) != len(predictors):
        raise ValueError(f"Number of coefficients ({len(coefficients)}) must match "
                        f"number of predictors ({len(predictors)})")

    n = ["len", response]

    # Linear predictor for observation i
    eta_i = _build_linear_predictor(intercept, coefficients, predictors, "i")

    # Sum of squared residuals: Σ(yᵢ - ηᵢ)²
    ssr = ["sum", "i", n, ["^", ["-", ["@", response, "i"], eta_i], 2]]

    # Log-likelihood: -n/2·log(2πσ²) - SSR/(2σ²)
    log_lik = [
        "+",
        ["*", -0.5, ["*", n, ["log", ["*", 2, 3.141592653589793]]]],
        ["+",
         ["*", -0.5, ["*", n, ["log", var]]],
         ["*", -0.5, ["*", ["/", 1, var], ssr]]]
    ]

    params = [intercept] + coefficients + [var]
    return LikelihoodModel(log_lik, params)


def logistic_regression(
    response: str = "y",
    predictors: Optional[List[str]] = None,
    intercept: str = "beta0",
    coefficients: Optional[List[str]] = None
) -> LikelihoodModel:
    """
    Logistic regression (Bernoulli GLM with logit link).

    Model: Y ~ Bernoulli(p) where logit(p) = η = β₀ + Σβⱼxⱼ
           p = 1/(1 + exp(-η)) = exp(η)/(1 + exp(η))

    ℓ(β) = Σ[yᵢ·ηᵢ - log(1 + exp(ηᵢ))]

    Args:
        response: Name of response variable (0/1 values)
        predictors: Names of predictor variables
        intercept: Name of intercept parameter
        coefficients: Names of coefficient parameters

    Returns:
        LikelihoodModel for logistic regression

    Example:
        >>> model = logistic_regression(response="y", predictors=["x1"])
        >>> mle, _ = model.mle(
        ...     data={"y": [0, 0, 1, 1], "x1": [-1, -0.5, 0.5, 1]},
        ...     init={"beta0": 0, "beta1": 0}
        ... )
    """
    if predictors is None:
        predictors = []
    if coefficients is None:
        coefficients = [f"beta{i+1}" for i in range(len(predictors))]

    if len(coefficients) != len(predictors):
        raise ValueError(f"Number of coefficients ({len(coefficients)}) must match "
                        f"number of predictors ({len(predictors)})")

    n = ["len", response]

    # Linear predictor for observation i
    eta_i = _build_linear_predictor(intercept, coefficients, predictors, "i")

    # Log-likelihood contribution: y*η - log(1 + exp(η))
    # Note: log(1 + exp(η)) = log1p(exp(η)) for numerical stability,
    # but we use the direct form here
    contrib_i = ["-",
                 ["*", ["@", response, "i"], eta_i],
                 ["log", ["+", 1, ["exp", eta_i]]]]

    log_lik = ["sum", "i", n, contrib_i]

    params = [intercept] + coefficients
    return LikelihoodModel(log_lik, params)


def probit_regression(
    response: str = "y",
    predictors: Optional[List[str]] = None,
    intercept: str = "beta0",
    coefficients: Optional[List[str]] = None
) -> LikelihoodModel:
    """
    Probit regression (Bernoulli GLM with probit link).

    Model: Y ~ Bernoulli(p) where Φ⁻¹(p) = η = β₀ + Σβⱼxⱼ
           p = Φ(η) where Φ is the standard normal CDF

    ℓ(β) = Σ[yᵢ·log(Φ(ηᵢ)) + (1-yᵢ)·log(1-Φ(ηᵢ))]

    Note: Uses the error function approximation: Φ(x) ≈ 0.5*(1 + erf(x/√2))

    Args:
        response: Name of response variable (0/1 values)
        predictors: Names of predictor variables
        intercept: Name of intercept parameter
        coefficients: Names of coefficient parameters

    Returns:
        LikelihoodModel for probit regression
    """
    if predictors is None:
        predictors = []
    if coefficients is None:
        coefficients = [f"beta{i+1}" for i in range(len(predictors))]

    if len(coefficients) != len(predictors):
        raise ValueError(f"Number of coefficients ({len(coefficients)}) must match "
                        f"number of predictors ({len(predictors)})")

    n = ["len", response]

    # Linear predictor for observation i
    eta_i = _build_linear_predictor(intercept, coefficients, predictors, "i")

    # Φ(η) = 0.5 * (1 + erf(η / sqrt(2)))
    # Using sqrt(2) ≈ 1.4142135623730951
    phi_eta = ["*", 0.5, ["+", 1, ["erf", ["/", eta_i, 1.4142135623730951]]]]

    # Log-likelihood contribution: y*log(Φ(η)) + (1-y)*log(1-Φ(η))
    contrib_i = ["+",
                 ["*", ["@", response, "i"], ["log", phi_eta]],
                 ["*", ["-", 1, ["@", response, "i"]], ["log", ["-", 1, phi_eta]]]]

    log_lik = ["sum", "i", n, contrib_i]

    params = [intercept] + coefficients
    return LikelihoodModel(log_lik, params)


def poisson_regression(
    response: str = "y",
    predictors: Optional[List[str]] = None,
    intercept: str = "beta0",
    coefficients: Optional[List[str]] = None
) -> LikelihoodModel:
    """
    Poisson regression (Poisson GLM with log link).

    Model: Y ~ Poisson(λ) where log(λ) = η = β₀ + Σβⱼxⱼ
           λ = exp(η)

    ℓ(β) = Σ[yᵢ·ηᵢ - exp(ηᵢ)]  (ignoring log(y!))

    Args:
        response: Name of response variable (non-negative integers)
        predictors: Names of predictor variables
        intercept: Name of intercept parameter
        coefficients: Names of coefficient parameters

    Returns:
        LikelihoodModel for Poisson regression

    Example:
        >>> model = poisson_regression(response="counts", predictors=["x1", "x2"])
        >>> mle, _ = model.mle(
        ...     data={"counts": [1, 3, 5, 8], "x1": [0, 1, 2, 3], "x2": [1, 1, 2, 2]},
        ...     init={"beta0": 0, "beta1": 0, "beta2": 0}
        ... )
    """
    if predictors is None:
        predictors = []
    if coefficients is None:
        coefficients = [f"beta{i+1}" for i in range(len(predictors))]

    if len(coefficients) != len(predictors):
        raise ValueError(f"Number of coefficients ({len(coefficients)}) must match "
                        f"number of predictors ({len(predictors)})")

    n = ["len", response]

    # Linear predictor for observation i
    eta_i = _build_linear_predictor(intercept, coefficients, predictors, "i")

    # Log-likelihood contribution: y*η - exp(η)
    contrib_i = ["-",
                 ["*", ["@", response, "i"], eta_i],
                 ["exp", eta_i]]

    log_lik = ["sum", "i", n, contrib_i]

    params = [intercept] + coefficients
    return LikelihoodModel(log_lik, params)


def gamma_regression(
    response: str = "y",
    predictors: Optional[List[str]] = None,
    intercept: str = "beta0",
    coefficients: Optional[List[str]] = None,
    shape: str = "alpha"
) -> LikelihoodModel:
    """
    Gamma regression (Gamma GLM with log link).

    Model: Y ~ Gamma(α, β) where log(μ) = η = β₀ + Σβⱼxⱼ
           μ = α/β = exp(η), so β = α/exp(η) = α·exp(-η)

    ℓ(β, α) = Σ[α·log(α) - log(Γ(α)) + (α-1)·log(yᵢ) - α·ηᵢ - α·yᵢ·exp(-ηᵢ)]

    This is the canonical GLM form with log link for the mean.

    Args:
        response: Name of response variable (positive values)
        predictors: Names of predictor variables
        intercept: Name of intercept parameter
        coefficients: Names of coefficient parameters
        shape: Name of shape parameter (α)

    Returns:
        LikelihoodModel for Gamma regression
    """
    if predictors is None:
        predictors = []
    if coefficients is None:
        coefficients = [f"beta{i+1}" for i in range(len(predictors))]

    if len(coefficients) != len(predictors):
        raise ValueError(f"Number of coefficients ({len(coefficients)}) must match "
                        f"number of predictors ({len(predictors)})")

    n = ["len", response]

    # Linear predictor for observation i
    eta_i = _build_linear_predictor(intercept, coefficients, predictors, "i")

    # Log-likelihood contribution for observation i:
    # α·log(α) - lgamma(α) + (α-1)·log(yᵢ) - α·ηᵢ - α·yᵢ·exp(-ηᵢ)
    contrib_i = ["+",
                 ["*", shape, ["log", shape]],
                 ["+",
                  ["*", -1, ["lgamma", shape]],
                  ["+",
                   ["*", ["-", shape, 1], ["log", ["@", response, "i"]]],
                   ["+",
                    ["*", -1, ["*", shape, eta_i]],
                    ["*", -1, ["*", shape, ["*", ["@", response, "i"],
                                            ["exp", ["*", -1, eta_i]]]]]]]]]

    log_lik = ["sum", "i", n, contrib_i]

    params = [intercept] + coefficients + [shape]
    return LikelihoodModel(log_lik, params)


def negative_binomial_regression(
    response: str = "y",
    predictors: Optional[List[str]] = None,
    intercept: str = "beta0",
    coefficients: Optional[List[str]] = None,
    dispersion: str = "alpha"
) -> LikelihoodModel:
    """
    Negative binomial regression with log link (NB2 parameterization).

    Model: Y ~ NegBin(μ, α) where log(μ) = η = β₀ + Σβⱼxⱼ
           Var(Y) = μ + α·μ² (overdispersion controlled by α)

    ℓ(β, α) = Σ[lgamma(y+1/α) - lgamma(1/α) - lgamma(y+1)
              + (1/α)·log(1/(1+α·μ)) + y·log(α·μ/(1+α·μ))]

    As α → 0, this approaches Poisson regression.

    Args:
        response: Name of response variable (non-negative integers)
        predictors: Names of predictor variables
        intercept: Name of intercept parameter
        coefficients: Names of coefficient parameters
        dispersion: Name of dispersion parameter (α > 0)

    Returns:
        LikelihoodModel for negative binomial regression

    Example:
        >>> model = negative_binomial_regression(response="counts", predictors=["x"])
        >>> mle, _ = model.mle(
        ...     data={"counts": [0, 1, 5, 3, 12, 8], "x": [1, 2, 3, 4, 5, 6]},
        ...     init={"beta0": 0, "beta1": 0.1, "alpha": 1.0},
        ...     bounds={"alpha": (0.01, None)}
        ... )
    """
    if predictors is None:
        predictors = []
    if coefficients is None:
        coefficients = [f"beta{i+1}" for i in range(len(predictors))]

    if len(coefficients) != len(predictors):
        raise ValueError(f"Number of coefficients ({len(coefficients)}) must match "
                        f"number of predictors ({len(predictors)})")

    n = ["len", response]

    # Linear predictor for observation i
    eta_i = _build_linear_predictor(intercept, coefficients, predictors, "i")

    # μ = exp(η)
    mu_i = ["exp", eta_i]

    # r = 1/α (size parameter in standard NB parameterization)
    r = ["/", 1, dispersion]

    # Log-likelihood contribution for observation i:
    # lgamma(y + r) - lgamma(r) + r·log(r/(r+μ)) + y·log(μ/(r+μ))
    # Ignoring lgamma(y+1) since it doesn't depend on parameters
    y_i = ["@", response, "i"]
    r_plus_mu = ["+", r, mu_i]

    contrib_i = ["+",
                 ["lgamma", ["+", y_i, r]],
                 ["+",
                  ["*", -1, ["lgamma", r]],
                  ["+",
                   ["*", r, ["log", ["/", r, r_plus_mu]]],
                   ["*", y_i, ["log", ["/", mu_i, r_plus_mu]]]]]]

    log_lik = ["sum", "i", n, contrib_i]

    params = [intercept] + coefficients + [dispersion]
    return LikelihoodModel(log_lik, params)
