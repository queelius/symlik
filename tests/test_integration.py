"""Integration tests for complete statistical inference workflows.

These tests verify end-to-end workflows combining:
- Model creation via symlik.distributions
- Maximum likelihood estimation
- Standard error computation
- Confidence interval coverage
- Symbolic differentiation producing correct numerical results
"""

import math
import pytest
import numpy as np
from symlik.distributions import (
    exponential,
    normal,
    normal_mean,
    poisson,
    bernoulli,
    gamma,
)
from symlik.model import LikelihoodModel
from symlik.calculus import diff, gradient, hessian
from symlik import evaluate


class TestEndToEndStatisticalInference:
    """Integration tests for complete inference workflows."""

    def test_exponential_inference_pipeline(self):
        """Test complete workflow: create model -> fit -> inference."""
        # Given: Exponential data with known true parameter
        np.random.seed(42)
        true_lambda = 2.0
        data = {"x": np.random.exponential(1/true_lambda, size=100).tolist()}

        # When: Full inference pipeline
        model = exponential()
        fit = model.fit(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})
        # se available via fit.se

        # Then: Confidence interval should contain true value
        ci_lower = fit.params["lambda"] - 1.96 * fit.se["lambda"]
        ci_upper = fit.params["lambda"] + 1.96 * fit.se["lambda"]
        assert ci_lower < true_lambda < ci_upper

    def test_normal_mean_inference_pipeline(self):
        """Test normal distribution inference with known variance."""
        # Given: Normal data
        np.random.seed(123)
        true_mu = 5.0
        true_sigma = 2.0
        data = {"x": np.random.normal(true_mu, true_sigma, size=100).tolist()}

        # When: Fit normal model with known variance
        model = normal_mean(known_var=true_sigma**2)
        fit = model.fit(data=data, init={"mu": 0})
        # se available via fit.se

        # Then: 95% CI should contain true mean
        ci_lower = fit.params["mu"] - 1.96 * fit.se["mu"]
        ci_upper = fit.params["mu"] + 1.96 * fit.se["mu"]
        assert ci_lower < true_mu < ci_upper

    def test_poisson_inference_pipeline(self):
        """Test Poisson distribution inference."""
        # Given: Poisson data
        np.random.seed(456)
        true_lambda = 5.0
        data = {"x": np.random.poisson(true_lambda, size=100).tolist()}

        # When: Fit Poisson model
        model = poisson()
        fit = model.fit(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 100)})
        # se available via fit.se

        # Then: 95% CI should contain true parameter
        ci_lower = fit.params["lambda"] - 1.96 * fit.se["lambda"]
        ci_upper = fit.params["lambda"] + 1.96 * fit.se["lambda"]
        assert ci_lower < true_lambda < ci_upper

    def test_bernoulli_inference_pipeline(self):
        """Test Bernoulli distribution inference."""
        # Given: Bernoulli data
        np.random.seed(789)
        true_p = 0.3
        data = {"x": np.random.binomial(1, true_p, size=200).tolist()}

        # When: Fit Bernoulli model
        model = bernoulli()
        fit = model.fit(data=data, init={"p": 0.5}, bounds={"p": (0.01, 0.99)})
        # se available via fit.se

        # Then: 95% CI should contain true parameter
        ci_lower = fit.params["p"] - 1.96 * fit.se["p"]
        ci_upper = fit.params["p"] + 1.96 * fit.se["p"]
        assert ci_lower < true_p < ci_upper


class TestSymbolicToNumericalPipeline:
    """Test that symbolic operations produce correct numerical results."""

    def test_symbolic_diff_matches_numerical_diff(self):
        """Symbolic derivative should match numerical derivative."""
        # Given: Expression f(x) = x^3 + 2x^2 + x
        expr = ["+", ["+", ["^", "x", 3], ["*", 2, ["^", "x", 2]]], "x"]

        # When: Compute symbolic derivative
        deriv = diff(expr, "x")

        # Then: At various points, symbolic derivative matches numerical
        for x_val in [0.5, 1.0, 2.0, -1.0]:
            # Symbolic: f'(x) = 3x^2 + 4x + 1
            symbolic_result = evaluate(deriv, {"x": x_val})
            expected = 3 * x_val**2 + 4 * x_val + 1
            assert symbolic_result == pytest.approx(expected, rel=1e-6)

    def test_symbolic_gradient_matches_numerical(self):
        """Symbolic gradient should match expected numerical gradient."""
        # Given: f(x,y) = x^2*y + x*y^2
        expr = ["+", ["*", ["^", "x", 2], "y"], ["*", "x", ["^", "y", 2]]]

        # When: Compute symbolic gradient
        grad = gradient(expr, ["x", "y"])

        # Then: At point (2, 3), gradients match expected
        env = {"x": 2, "y": 3}
        # df/dx = 2xy + y^2 = 2*2*3 + 9 = 21
        # df/dy = x^2 + 2xy = 4 + 2*2*3 = 16
        grad_x = evaluate(grad[0], env)
        grad_y = evaluate(grad[1], env)
        assert grad_x == pytest.approx(21.0, rel=1e-6)
        assert grad_y == pytest.approx(16.0, rel=1e-6)

    def test_symbolic_hessian_matches_numerical(self):
        """Symbolic Hessian should match expected numerical Hessian."""
        # Given: f(x,y) = x^3 + x*y + y^2
        expr = ["+", ["+", ["^", "x", 3], ["*", "x", "y"]], ["^", "y", 2]]

        # When: Compute symbolic Hessian
        hess = hessian(expr, ["x", "y"])

        # Then: At point (1, 2), Hessian matches expected
        env = {"x": 1, "y": 2}
        # d2f/dx2 = 6x = 6
        # d2f/dxdy = 1
        # d2f/dydx = 1
        # d2f/dy2 = 2
        assert evaluate(hess[0][0], env) == pytest.approx(6.0, rel=1e-6)
        assert evaluate(hess[0][1], env) == pytest.approx(1.0, rel=1e-6)
        assert evaluate(hess[1][0], env) == pytest.approx(1.0, rel=1e-6)
        assert evaluate(hess[1][1], env) == pytest.approx(2.0, rel=1e-6)


class TestModelConstructionToInference:
    """Test building custom models and performing inference."""

    def test_custom_likelihood_model_workflow(self):
        """Build a custom model from scratch and fit it."""
        # Given: Custom likelihood for exponential-like model
        # log L = sum(log(lambda) - lambda*x_i) = n*log(lambda) - lambda*sum(x)
        log_lik = ["-",
                   ["*", ["len", "x"], ["log", "lambda"]],
                   ["*", "lambda", ["total", "x"]]]
        model = LikelihoodModel(log_lik, ["lambda"])

        # When: Generate data and fit
        np.random.seed(42)
        true_lambda = 2.0
        data = {"x": np.random.exponential(1/true_lambda, size=100).tolist()}
        fit = model.fit(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})

        # Then: MLE should be close to true parameter
        assert fit.params["lambda"] == pytest.approx(true_lambda, rel=0.2)

    def test_custom_model_score_evaluated_correctly(self):
        """Score function should evaluate to zero at MLE (numerically)."""
        # Given: Simple quadratic log-likelihood around mu=0
        # log L = -sum((x_i - mu)^2) = -n*mu^2 + 2*mu*sum(x) - sum(x^2)
        # We'll use a simple form: -sum((x - mu)^2)
        log_lik = ["*", -1,
                   ["sum", "i", ["len", "x"],
                    ["^", ["-", ["@", "x", "i"], "mu"], 2]]]
        model = LikelihoodModel(log_lik, ["mu"])

        # When: Fit to data centered at mu=3
        data = {"x": [2.0, 3.0, 4.0]}  # mean = 3
        fit = model.fit(data=data, init={"mu": 0})

        # Then: MLE should be at the mean (which is 3)
        assert fit.params["mu"] == pytest.approx(3.0, rel=1e-4)

        # And: Score at MLE should be approximately zero
        score_expr = model.score()[0]
        score_at_mle = evaluate(score_expr, {**data, **fit.params})
        assert score_at_mle == pytest.approx(0.0, abs=1e-3)


class TestCoveragePropertiesAcrossDistributions:
    """Test that confidence interval coverage is correct across runs."""

    @pytest.mark.parametrize("seed", range(10))
    def test_exponential_coverage_simulation(self, seed):
        """Repeated simulations should show ~95% coverage for 95% CI."""
        np.random.seed(seed * 100)
        true_lambda = 1.5
        n = 50

        data = {"x": np.random.exponential(1/true_lambda, size=n).tolist()}
        model = exponential()

        fit = model.fit(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})
        # se available via fit.se

        ci_lower = fit.params["lambda"] - 1.96 * fit.se["lambda"]
        ci_upper = fit.params["lambda"] + 1.96 * fit.se["lambda"]

        # This individual test just checks the CI is valid (positive, etc.)
        assert ci_lower > 0, "Lower CI bound should be positive"
        assert ci_upper > ci_lower, "Upper CI should be greater than lower"

    def test_exponential_coverage_aggregated(self):
        """Aggregate coverage should be approximately 95%."""
        true_lambda = 1.5
        n = 100
        n_simulations = 100
        coverage_count = 0

        for seed in range(n_simulations):
            np.random.seed(seed)
            data = {"x": np.random.exponential(1/true_lambda, size=n).tolist()}
            model = exponential()

            fit = model.fit(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})
            # se available via fit.se

            ci_lower = fit.params["lambda"] - 1.96 * fit.se["lambda"]
            ci_upper = fit.params["lambda"] + 1.96 * fit.se["lambda"]

            if ci_lower < true_lambda < ci_upper:
                coverage_count += 1

        coverage_rate = coverage_count / n_simulations
        # With n=100 samples, coverage should be close to 95% (allow 85-100%)
        assert 0.85 <= coverage_rate <= 1.0, f"Coverage {coverage_rate} not in expected range"


class TestDistributionEvaluationConsistency:
    """Test that distribution evaluation is consistent across methods."""

    def test_exponential_evaluate_vs_manual(self):
        """Model evaluation should match manual log-likelihood calculation."""
        model = exponential()
        data = {"x": [1.0, 2.0, 3.0, 4.0, 5.0]}
        lam = 0.5

        # Via model
        ll_model = model.evaluate({**data, "lambda": lam})

        # Manual: sum(log(lambda) - lambda*x_i) = n*log(lambda) - lambda*sum(x)
        n = len(data["x"])
        sum_x = sum(data["x"])
        ll_manual = n * math.log(lam) - lam * sum_x

        assert ll_model == pytest.approx(ll_manual, rel=1e-6)

    def test_poisson_evaluate_vs_manual(self):
        """Poisson model evaluation should match manual log-likelihood."""
        model = poisson()
        data = {"x": [1, 2, 3, 2, 1]}
        lam = 2.0

        # Via model (note: symlik uses sum(x)*log(lambda) - n*lambda, ignoring -log(x!))
        ll_model = model.evaluate({**data, "lambda": lam})

        # Manual (same form used by symlik)
        sum_x = sum(data["x"])
        n = len(data["x"])
        ll_manual = sum_x * math.log(lam) - n * lam

        assert ll_model == pytest.approx(ll_manual, rel=1e-6)

    def test_bernoulli_evaluate_vs_manual(self):
        """Bernoulli model evaluation should match manual log-likelihood."""
        model = bernoulli()
        data = {"x": [1, 0, 1, 1, 0]}
        p = 0.6

        # Via model
        ll_model = model.evaluate({**data, "p": p})

        # Manual: sum(x*log(p) + (1-x)*log(1-p))
        successes = sum(data["x"])
        failures = len(data["x"]) - successes
        ll_manual = successes * math.log(p) + failures * math.log(1 - p)

        assert ll_model == pytest.approx(ll_manual, rel=1e-6)


class TestMultiParameterInference:
    """Test inference for models with multiple parameters."""

    def test_gamma_parameter_estimation(self):
        """Test that gamma distribution fits produce reasonable estimates."""
        model = gamma()

        # Given: Gamma data with shape=2, rate=1 (mean=2)
        np.random.seed(42)
        true_alpha = 2.0
        true_beta = 1.0
        data = {"x": np.random.gamma(true_alpha, 1/true_beta, size=200).tolist()}

        # When: Fit model
        fit = model.fit(
            data=data,
            init={"alpha": 1.0, "beta": 1.0},
            bounds={"alpha": (0.1, 10), "beta": (0.1, 10)}
        )

        # Then: Parameters should be reasonably close to true values
        assert fit.params["alpha"] == pytest.approx(true_alpha, rel=0.3)
        assert fit.params["beta"] == pytest.approx(true_beta, rel=0.3)

    def test_normal_two_parameter_inference(self):
        """Test normal distribution with both mu and sigma2 estimated."""
        model = normal()

        # Given: Normal data
        np.random.seed(42)
        true_mu = 5.0
        true_sigma2 = 4.0
        data = {"x": np.random.normal(true_mu, math.sqrt(true_sigma2), size=200).tolist()}

        # When: Fit model
        fit = model.fit(
            data=data,
            init={"mu": 0, "sigma2": 1.0},
            bounds={"sigma2": (0.1, 100)}
        )

        # Then: Both parameters should be reasonably estimated
        assert fit.params["mu"] == pytest.approx(true_mu, rel=0.15)
        assert fit.params["sigma2"] == pytest.approx(true_sigma2, rel=0.25)
