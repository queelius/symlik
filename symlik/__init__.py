"""
symlik: Symbolic Likelihood Models

A clean library for symbolic statistical inference, combining:
- Symbolic differentiation via rerum
- Numerical evaluation with extensible operators
- Likelihood-based inference (MLE, standard errors)

Quick Start:
    >>> from symlik import LikelihoodModel
    >>> from symlik.distributions import exponential
    >>>
    >>> # Use pre-built distribution
    >>> model = exponential()
    >>> mle, _ = model.mle(data={'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})
    >>>
    >>> # Or build custom log-likelihood
    >>> log_lik = ['sum', 'i', ['len', 'x'],
    ...            ['+', ['log', 'lambda'],
    ...             ['*', -1, ['*', 'lambda', ['@', 'x', 'i']]]]]
    >>> model = LikelihoodModel(log_lik, params=['lambda'])
    >>> score = model.score()  # Symbolic gradient

For heterogeneous data with different observation types:
    >>> from symlik import ContributionModel
    >>> from symlik.contributions import complete_exponential, right_censored_exponential
    >>>
    >>> model = ContributionModel(
    ...     params=["lambda"],
    ...     type_column="obs_type",
    ...     contributions={
    ...         "complete": complete_exponential(),
    ...         "censored": right_censored_exponential(),
    ...     }
    ... )
"""

__version__ = "0.4.0"

# Core model classes
from .model import LikelihoodModel
from .contribution import ContributionModel
from .fitted import FittedLikelihoodModel

# Evaluation
from .evaluate import evaluate, STANDARD_OPS, ExprType

# Calculus operations
from .calculus import (
    simplify,
    diff,
    integrate,
    gradient,
    hessian,
    jacobian,
    laplacian,
    diff_at,
    gradient_at,
    hessian_at,
)

# Rule engine access
from .rules import (
    get_default_engine,
    create_derivative_engine,
    create_algebra_engine,
    create_integral_engine,
    create_calculus_engine,
    create_full_engine,
)

__all__ = [
    # Core
    "LikelihoodModel",
    "ContributionModel",
    "FittedLikelihoodModel",
    "ExprType",
    # Evaluation
    "evaluate",
    "STANDARD_OPS",
    # Calculus (symbolic)
    "simplify",
    "diff",
    "integrate",
    "gradient",
    "hessian",
    "jacobian",
    "laplacian",
    # Calculus (numerical)
    "diff_at",
    "gradient_at",
    "hessian_at",
    # Engines
    "get_default_engine",
    "create_derivative_engine",
    "create_algebra_engine",
    "create_integral_engine",
    "create_calculus_engine",
    "create_full_engine",
]

# Additional modules available as symlik.<module>
from . import series      # Series system reliability
from . import regression  # GLM/regression models
from . import mixture     # Latent variable/mixture models
