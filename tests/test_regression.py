"""Tests for symlik.regression module (GLM models)."""

import math
import pytest
import numpy as np
from symlik.regression import (
    linear_regression,
    logistic_regression,
    probit_regression,
    poisson_regression,
    gamma_regression,
    negative_binomial_regression,
)


class TestLinearRegression:
    """Test linear regression model."""

    def test_params_no_predictors(self):
        """With no predictors, just intercept and variance."""
        model = linear_regression(predictors=[])
        assert model.params == ["beta0", "sigma2"]

    def test_params_with_predictors(self):
        model = linear_regression(predictors=["x1", "x2"])
        assert model.params == ["beta0", "beta1", "beta2", "sigma2"]

    def test_custom_names(self):
        model = linear_regression(
            response="outcome",
            predictors=["age", "income"],
            intercept="alpha",
            coefficients=["b_age", "b_income"],
            var="s2"
        )
        assert model.params == ["alpha", "b_age", "b_income", "s2"]

    def test_mle_intercept_only(self):
        """Test intercept-only model (equivalent to estimating mean)."""
        model = linear_regression(predictors=[])
        np.random.seed(42)
        y = np.random.normal(5.0, 2.0, 100).tolist()

        fit = model.fit(
            data={"y": y},
            init={"beta0": 0.0, "sigma2": 1.0},
            bounds={"sigma2": (0.01, None)}
        )

        assert fit.params["beta0"] == pytest.approx(np.mean(y), rel=0.01)
        assert fit.params["sigma2"] == pytest.approx(np.var(y), rel=0.1)

    def test_mle_simple_regression(self):
        """Test simple linear regression y = 2 + 3*x + noise."""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 10, n)
        y = 2 + 3 * x + np.random.normal(0, 1, n)

        model = linear_regression(predictors=["x"])
        fit = model.fit(
            data={"y": y.tolist(), "x": x.tolist()},
            init={"beta0": 0, "beta1": 0, "sigma2": 1},
            bounds={"sigma2": (0.01, None)}
        )

        assert fit.params["beta0"] == pytest.approx(2.0, rel=0.2)
        assert fit.params["beta1"] == pytest.approx(3.0, rel=0.1)
        assert fit.params["sigma2"] == pytest.approx(1.0, rel=0.3)

    def test_linear_predictor_structure(self):
        """Test that multiple predictors build correct linear predictor."""
        # This tests the model structure without running full MLE optimization
        model = linear_regression(predictors=["x1", "x2"])

        # Verify the params list is correct
        assert model.params == ["beta0", "beta1", "beta2", "sigma2"]

        # Verify score has correct length
        score = model.score()
        assert len(score) == 4  # One per parameter

        # Test that log-likelihood can be evaluated
        data = {
            "y": [1.0, 2.0, 3.0],
            "x1": [0.5, 1.0, 1.5],
            "x2": [1.0, 2.0, 3.0],
            "beta0": 1.0,
            "beta1": 1.0,
            "beta2": 0.5,
            "sigma2": 1.0
        }
        ll = model.evaluate(data)
        assert np.isfinite(ll)

    def test_score_and_hessian(self):
        """Test that score and Hessian have correct dimensions."""
        model = linear_regression(predictors=["x1", "x2"])

        score = model.score()
        assert len(score) == 4  # beta0, beta1, beta2, sigma2

        hess = model.hessian()
        assert len(hess) == 4
        assert all(len(row) == 4 for row in hess)


class TestLogisticRegression:
    """Test logistic regression model."""

    def test_params(self):
        model = logistic_regression(predictors=["x1"])
        assert model.params == ["beta0", "beta1"]

    def test_mle_separable_data(self):
        """Test with clearly separable data."""
        np.random.seed(42)

        # Generate separable data
        x = np.array([-3, -2, -1, 1, 2, 3]).astype(float)
        y = np.array([0, 0, 0, 1, 1, 1]).astype(float)

        model = logistic_regression(predictors=["x"])
        fit = model.fit(
            data={"y": y.tolist(), "x": x.tolist()},
            init={"beta0": 0, "beta1": 0},
            bounds={"beta0": (-10, 10), "beta1": (-10, 10)}
        )

        # With separable data, beta1 should be positive (higher x -> higher prob)
        assert fit.params["beta1"] > 0

    def test_mle_probability_estimation(self):
        """Test that predicted probabilities make sense."""
        from symlik import evaluate

        np.random.seed(42)
        n = 200
        x = np.random.uniform(-3, 3, n)
        # True model: logit(p) = -0.5 + 1.5*x
        true_prob = 1 / (1 + np.exp(-(-0.5 + 1.5 * x)))
        y = (np.random.uniform(0, 1, n) < true_prob).astype(float)

        model = logistic_regression(predictors=["x"])
        fit = model.fit(
            data={"y": y.tolist(), "x": x.tolist()},
            init={"beta0": 0, "beta1": 0}
        )

        # Coefficients should be in the right ballpark
        assert fit.params["beta0"] == pytest.approx(-0.5, abs=0.5)
        assert fit.params["beta1"] == pytest.approx(1.5, abs=0.5)

    def test_score(self):
        model = logistic_regression(predictors=["x"])
        score = model.score()
        assert len(score) == 2


class TestProbitRegression:
    """Test probit regression model."""

    def test_params(self):
        model = probit_regression(predictors=["x"])
        assert model.params == ["beta0", "beta1"]

    def test_evaluate(self):
        """Test log-likelihood evaluation."""
        model = probit_regression(predictors=["x"])

        # At x=0 with beta0=0, beta1=1:
        # eta = 0, Phi(0) = 0.5
        # For y=1: log(0.5), for y=0: log(0.5)
        data = {"y": [1.0, 0.0], "x": [0.0, 0.0], "beta0": 0.0, "beta1": 1.0}
        ll = model.evaluate(data)
        expected = 2 * math.log(0.5)
        assert ll == pytest.approx(expected, rel=0.01)


class TestPoissonRegression:
    """Test Poisson regression model."""

    def test_params(self):
        model = poisson_regression(predictors=["x"])
        assert model.params == ["beta0", "beta1"]

    def test_mle(self):
        """Test MLE with simulated count data."""
        np.random.seed(42)
        n = 200
        x = np.random.uniform(0, 2, n)
        # True model: log(λ) = 0.5 + 0.8*x, so λ = exp(0.5 + 0.8*x)
        true_lambda = np.exp(0.5 + 0.8 * x)
        y = np.random.poisson(true_lambda)

        model = poisson_regression(predictors=["x"])
        fit = model.fit(
            data={"y": y.tolist(), "x": x.tolist()},
            init={"beta0": 0, "beta1": 0}
        )

        assert fit.params["beta0"] == pytest.approx(0.5, abs=0.3)
        assert fit.params["beta1"] == pytest.approx(0.8, abs=0.3)

    def test_score(self):
        model = poisson_regression(predictors=["x1", "x2"])
        score = model.score()
        assert len(score) == 3


class TestGammaRegression:
    """Test Gamma regression model."""

    def test_params(self):
        model = gamma_regression(predictors=["x"])
        assert model.params == ["beta0", "beta1", "alpha"]

    def test_evaluate(self):
        """Test that log-likelihood can be evaluated."""
        model = gamma_regression(predictors=["x"])

        # Simple evaluation test
        data = {
            "y": [1.0, 2.0, 3.0],
            "x": [0.0, 0.0, 0.0],
            "beta0": 0.0,
            "beta1": 0.0,
            "alpha": 1.0
        }
        ll = model.evaluate(data)
        # Just check it returns a reasonable number
        assert np.isfinite(ll)

    def test_score(self):
        model = gamma_regression(predictors=["x"])
        score = model.score()
        assert len(score) == 3


class TestNegativeBinomialRegression:
    """Test negative binomial regression model."""

    def test_params(self):
        model = negative_binomial_regression(predictors=["x"])
        assert model.params == ["beta0", "beta1", "alpha"]

    def test_negative_binomial_evaluate(self):
        """Test negative binomial regression can evaluate log-likelihood."""
        model = negative_binomial_regression(predictors=["x"])

        # Simple evaluation test - no MLE optimization needed
        data = {
            "y": [1, 2, 3, 5, 8],
            "x": [0.0, 0.5, 1.0, 1.5, 2.0],
            "beta0": 1.0,
            "beta1": 0.5,
            "alpha": 1.0
        }
        ll = model.evaluate(data)
        assert np.isfinite(ll)

    def test_score(self):
        model = negative_binomial_regression(predictors=["x"])
        score = model.score()
        assert len(score) == 3


class TestRegressionValidation:
    """Test input validation for regression models."""

    def test_mismatched_coefficients_predictors(self):
        with pytest.raises(ValueError):
            linear_regression(
                predictors=["x1", "x2"],
                coefficients=["b1"]  # Only 1 coefficient for 2 predictors
            )

    def test_empty_predictors_works(self):
        """Empty predictors should work (intercept-only model)."""
        model = linear_regression(predictors=[])
        assert model.params == ["beta0", "sigma2"]


class TestRegressionScoreHessian:
    """Test score and Hessian for all regression models."""

    def test_linear_regression_hessian(self):
        model = linear_regression(predictors=["x"])
        hess = model.hessian()
        assert len(hess) == 3
        assert all(len(row) == 3 for row in hess)

    def test_logistic_regression_hessian(self):
        model = logistic_regression(predictors=["x1", "x2"])
        hess = model.hessian()
        assert len(hess) == 3
        assert all(len(row) == 3 for row in hess)

    def test_poisson_regression_hessian(self):
        model = poisson_regression(predictors=["x"])
        hess = model.hessian()
        assert len(hess) == 2
        assert all(len(row) == 2 for row in hess)

    def test_negative_binomial_regression_score_length(self):
        """Test negative binomial regression score has correct length."""
        # Note: Hessian computation is slow for complex models.
        # We just test the score structure here.
        model = negative_binomial_regression(predictors=["x"])
        score = model.score()
        assert len(score) == 3
