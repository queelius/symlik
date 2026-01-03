"""Tests for symlik.model module."""

import math
import pytest
import numpy as np
from symlik.model import LikelihoodModel


class TestLikelihoodModelBasics:
    """Test basic LikelihoodModel functionality."""

    def test_init(self):
        log_lik = ["+", "x", "y"]
        model = LikelihoodModel(log_lik, ["x"])
        assert model.params == ["x"]
        assert model.log_lik == ["+", "x", "y"]

    def test_repr(self):
        model = LikelihoodModel("x", ["x"])
        assert "LikelihoodModel" in repr(model)
        assert "x" in repr(model)

    def test_multiple_params(self):
        model = LikelihoodModel("x", ["mu", "sigma"])
        assert model.params == ["mu", "sigma"]


class TestScore:
    """Test score (gradient) computation."""

    def test_score_linear(self):
        # ℓ(x) = 2x => score = 2
        model = LikelihoodModel(["*", 2, "x"], ["x"])
        score = model.score()
        assert len(score) == 1
        assert score[0] == 2

    def test_score_quadratic(self):
        # ℓ(x) = -x^2 => score = -2x
        model = LikelihoodModel(["*", -1, ["^", "x", 2]], ["x"])
        score = model.score()
        from symlik import evaluate
        # At x=3: score = -6
        assert evaluate(score[0], {"x": 3}) == pytest.approx(-6)

    def test_score_multiple_params(self):
        # ℓ(x, y) = x + 2y => score = [1, 2]
        model = LikelihoodModel(["+", "x", ["*", 2, "y"]], ["x", "y"])
        score = model.score()
        assert len(score) == 2
        assert score[0] == 1
        assert score[1] == 2

    def test_score_caching(self):
        model = LikelihoodModel(["^", "x", 2], ["x"])
        score1 = model.score()
        score2 = model.score()
        # Should return the same cached object
        assert score1 is score2


class TestHessian:
    """Test Hessian computation."""

    def test_hessian_quadratic(self):
        # ℓ(x) = -x^2 => H = -2
        model = LikelihoodModel(["*", -1, ["^", "x", 2]], ["x"])
        hess = model.hessian()
        assert len(hess) == 1
        assert len(hess[0]) == 1
        assert hess[0][0] == -2

    def test_hessian_two_params(self):
        # ℓ(x, y) = -(x^2 + y^2) => H = [[-2, 0], [0, -2]]
        model = LikelihoodModel(
            ["*", -1, ["+", ["^", "x", 2], ["^", "y", 2]]],
            ["x", "y"]
        )
        hess = model.hessian()
        assert hess == [[-2, 0], [0, -2]]

    def test_hessian_mixed_terms(self):
        # ℓ(x, y) = -(x^2 + xy + y^2) => H = [[-2, -1], [-1, -2]]
        model = LikelihoodModel(
            ["*", -1, ["+", ["+", ["^", "x", 2], ["*", "x", "y"]], ["^", "y", 2]]],
            ["x", "y"]
        )
        hess = model.hessian()
        from symlik import evaluate
        env = {"x": 0, "y": 0}
        H = [[evaluate(hess[i][j], env) for j in range(2)] for i in range(2)]
        assert H == [[-2, -1], [-1, -2]]

    def test_hessian_caching(self):
        model = LikelihoodModel(["^", "x", 2], ["x"])
        hess1 = model.hessian()
        hess2 = model.hessian()
        assert hess1 is hess2


class TestInformation:
    """Test Fisher information computation."""

    def test_information_negative_hessian(self):
        # ℓ(x) = -x^2 => H = -2 => I = 2
        model = LikelihoodModel(["*", -1, ["^", "x", 2]], ["x"])
        info = model.information()
        assert info[0][0] == 2

    def test_information_two_params(self):
        # ℓ(x, y) = -(x^2 + y^2) => I = [[2, 0], [0, 2]]
        model = LikelihoodModel(
            ["*", -1, ["+", ["^", "x", 2], ["^", "y", 2]]],
            ["x", "y"]
        )
        info = model.information()
        from symlik import evaluate
        env = {"x": 0, "y": 0}
        I_mat = [[evaluate(info[i][j], env) for j in range(2)] for i in range(2)]
        assert I_mat == [[2, 0], [0, 2]]


class TestNumericalEvaluation:
    """Test numerical evaluation methods."""

    def test_evaluate_simple(self):
        model = LikelihoodModel(["+", "x", "y"], ["x"])
        result = model.evaluate({"x": 3, "y": 4})
        assert result == pytest.approx(7.0)

    def test_score_at(self):
        # ℓ(x) = -x^2 => score = -2x
        model = LikelihoodModel(["*", -1, ["^", "x", 2]], ["x"])
        score = model.score_at({"x": 5})
        np.testing.assert_array_almost_equal(score, [-10])

    def test_score_at_multiple(self):
        model = LikelihoodModel(
            ["+", ["*", 2, "x"], ["*", 3, "y"]],
            ["x", "y"]
        )
        score = model.score_at({"x": 1, "y": 2})
        np.testing.assert_array_almost_equal(score, [2, 3])

    def test_hessian_at(self):
        model = LikelihoodModel(
            ["*", -1, ["+", ["^", "x", 2], ["^", "y", 2]]],
            ["x", "y"]
        )
        H = model.hessian_at({"x": 0, "y": 0})
        np.testing.assert_array_almost_equal(H, [[-2, 0], [0, -2]])

    def test_information_at(self):
        model = LikelihoodModel(
            ["*", -1, ["+", ["^", "x", 2], ["^", "y", 2]]],
            ["x", "y"]
        )
        I = model.information_at({"x": 0, "y": 0})
        np.testing.assert_array_almost_equal(I, [[2, 0], [0, 2]])


class TestFit:
    """Test model fitting with fit() API."""

    def test_fit_quadratic(self):
        # ℓ(x) = -(x - 3)^2 = -x^2 + 6x - 9
        # Maximum at x = 3
        model = LikelihoodModel(
            ["+", ["*", -1, ["^", "x", 2]], ["*", 6, "x"]],
            ["x"]
        )
        fit = model.fit(data={}, init={"x": 0})
        assert fit.params["x"] == pytest.approx(3.0, abs=1e-6)

    def test_fit_two_params(self):
        # ℓ(x, y) = -(x - 2)^2 - (y - 3)^2
        # Maximum at (2, 3)
        model = LikelihoodModel(
            ["+",
             ["*", -1, ["^", ["-", "x", 2], 2]],
             ["*", -1, ["^", ["-", "y", 3], 2]]],
            ["x", "y"]
        )
        fit = model.fit(data={}, init={"x": 0, "y": 0})
        assert fit.params["x"] == pytest.approx(2.0, abs=1e-5)
        assert fit.params["y"] == pytest.approx(3.0, abs=1e-5)

    def test_fit_with_bounds(self):
        # ℓ(x) = -(x - 5)^2, but bounded to [0, 3]
        # Should converge to 3 (upper bound)
        model = LikelihoodModel(
            ["*", -1, ["^", ["-", "x", 5], 2]],
            ["x"]
        )
        fit = model.fit(
            data={},
            init={"x": 1},
            bounds={"x": (0, 3)}
        )
        assert fit.params["x"] == pytest.approx(3.0, abs=1e-3)

    def test_fit_returns_fitted_model(self):
        model = LikelihoodModel(["*", -1, ["^", "x", 2]], ["x"])
        fit = model.fit(data={}, init={"x": 5})
        # Should have all FittedLikelihoodModel properties
        assert hasattr(fit, 'params')
        assert hasattr(fit, 'se')
        assert hasattr(fit, 'llf')
        assert hasattr(fit, 'n_iter')

    def test_fit_convergence_tolerance(self):
        model = LikelihoodModel(
            ["*", -1, ["^", ["-", "x", 10], 2]],
            ["x"]
        )
        # With tight tolerance
        fit = model.fit(data={}, init={"x": 0}, tol=1e-10)
        assert fit.params["x"] == pytest.approx(10.0, abs=1e-8)


class TestFitSE:
    """Test standard error from fitted model."""

    def test_se_simple(self):
        # ℓ(x) = -x^2 => I = 2 => SE = 1/sqrt(2)
        model = LikelihoodModel(["*", -1, ["^", "x", 2]], ["x"])
        fit = model.fit(data={}, init={"x": 0})
        assert fit.se["x"] == pytest.approx(1 / math.sqrt(2))

    def test_se_two_params_independent(self):
        # ℓ(x, y) = -x^2 - 2y^2 => I = [[2, 0], [0, 4]]
        # SE(x) = 1/sqrt(2), SE(y) = 1/sqrt(4) = 0.5
        model = LikelihoodModel(
            ["+", ["*", -1, ["^", "x", 2]], ["*", -2, ["^", "y", 2]]],
            ["x", "y"]
        )
        fit = model.fit(data={}, init={"x": 0, "y": 0})
        assert fit.se["x"] == pytest.approx(1 / math.sqrt(2))
        assert fit.se["y"] == pytest.approx(0.5)


class TestExponentialLikelihood:
    """Test with actual exponential likelihood."""

    @pytest.fixture
    def exponential_model(self):
        # ℓ(λ) = Σᵢ [log(λ) - λxᵢ] = n·log(λ) - λ·Σxᵢ
        log_lik = [
            "sum", "i", ["len", "x"],
            ["+",
             ["log", "lambda"],
             ["*", -1, ["*", "lambda", ["@", "x", "i"]]]]
        ]
        return LikelihoodModel(log_lik, ["lambda"])

    def test_exponential_score(self, exponential_model):
        # Score: n/λ - Σxᵢ
        score = exponential_model.score()
        assert len(score) == 1

    def test_exponential_fit(self, exponential_model):
        # MLE for exponential: λ̂ = 1/x̄
        data = {"x": [1, 2, 3, 4, 5]}  # mean = 3
        fit = exponential_model.fit(
            data=data,
            init={"lambda": 1.0},
            bounds={"lambda": (0.01, 10)}
        )
        expected = 1.0 / 3.0  # 1/mean
        assert fit.params["lambda"] == pytest.approx(expected, rel=1e-5)

    def test_exponential_se(self, exponential_model):
        data = {"x": [1, 2, 3, 4, 5]}
        fit = exponential_model.fit(
            data=data,
            init={"lambda": 1.0},
            bounds={"lambda": (0.01, 10)}
        )
        # SE(λ̂) = λ̂/sqrt(n) for exponential
        n = len(data["x"])
        expected_se = fit.params["lambda"] / math.sqrt(n)
        assert fit.se["lambda"] == pytest.approx(expected_se, rel=1e-3)


class TestNormalLikelihood:
    """Test with normal likelihood (known variance)."""

    @pytest.fixture
    def normal_model(self):
        # ℓ(μ) = -1/(2σ²) Σ(xᵢ - μ)² with σ² = 1
        log_lik = [
            "*", -0.5,
            ["sum", "i", ["len", "x"],
             ["^", ["-", ["@", "x", "i"], "mu"], 2]]
        ]
        return LikelihoodModel(log_lik, ["mu"])

    def test_normal_fit(self, normal_model):
        # MLE for μ: μ̂ = x̄
        data = {"x": [1, 2, 3, 4, 5]}  # mean = 3
        fit = normal_model.fit(data=data, init={"mu": 0})
        assert fit.params["mu"] == pytest.approx(3.0, abs=1e-5)

    def test_normal_se(self, normal_model):
        # SE(μ̂) = σ/sqrt(n) = 1/sqrt(5) for σ²=1
        data = {"x": [1, 2, 3, 4, 5]}
        fit = normal_model.fit(data=data, init={"mu": 0})
        expected_se = 1.0 / math.sqrt(5)
        assert fit.se["mu"] == pytest.approx(expected_se, rel=1e-3)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_fit_already_converged(self):
        # Start at the optimum
        model = LikelihoodModel(
            ["*", -1, ["^", "x", 2]],
            ["x"]
        )
        fit = model.fit(data={}, init={"x": 0}, tol=1e-8)
        assert fit.params["x"] == pytest.approx(0.0, abs=1e-7)
        assert fit.n_iter == 1  # Should converge immediately

    def test_single_data_point(self):
        log_lik = [
            "sum", "i", ["len", "x"],
            ["+", ["log", "lambda"], ["*", -1, ["*", "lambda", ["@", "x", "i"]]]]
        ]
        model = LikelihoodModel(log_lik, ["lambda"])
        data = {"x": [2.0]}
        fit = model.fit(
            data=data,
            init={"lambda": 1.0},
            bounds={"lambda": (0.01, 10)}
        )
        assert fit.params["lambda"] == pytest.approx(0.5, rel=1e-3)


class TestFitRobustness:
    """Test fit robustness to numerical difficulties."""

    def test_fit_singular_hessian_uses_gradient(self):
        # Constant likelihood has zero Hessian (singular)
        # Should fall back to gradient ascent without crashing
        log_lik = 5  # constant likelihood
        model = LikelihoodModel(log_lik, ["x"])
        fit = model.fit(data={}, init={"x": 3}, max_iter=5)
        # Should not crash even though Hessian is 0
        assert "x" in fit.params

    def test_fit_handles_nonfinite_gracefully(self):
        # log(x) has issues near x=0
        log_lik = ["log", "x"]
        model = LikelihoodModel(log_lik, ["x"])
        # Starting very small may produce non-finite gradients
        fit = model.fit(data={}, init={"x": 0.001}, max_iter=5, bounds={"x": (0.0001, 10)})
        # Should exit without crashing
        assert "x" in fit.params

    def test_fit_very_tight_bounds(self):
        # MLE should respect very tight bounds
        model = LikelihoodModel(["*", -1, ["^", ["-", "x", 5], 2]], ["x"])
        fit = model.fit(data={}, init={"x": 2}, bounds={"x": (2.0, 2.5)})
        assert 2.0 <= fit.params["x"] <= 2.5


class TestSERobustness:
    """Test standard error robustness."""

    def test_se_singular_information_returns_nan(self):
        # Constant likelihood has zero information (singular)
        log_lik = 5
        model = LikelihoodModel(log_lik, ["x"])
        fit = model.fit(data={}, init={"x": 0})
        assert math.isnan(fit.se["x"])

    def test_se_near_singular_information(self):
        # Very flat likelihood has near-singular information
        # ℓ(x) = -0.0001 * x^2 => I = 0.0002 => SE = sqrt(1/0.0002) ≈ 70.7
        model = LikelihoodModel(["*", -0.0001, ["^", "x", 2]], ["x"])
        fit = model.fit(data={}, init={"x": 0})
        expected = 1 / math.sqrt(0.0002)
        assert fit.se["x"] == pytest.approx(expected, rel=0.01)
