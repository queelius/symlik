"""Tests for symlik.distributions module."""

import math
import pytest
import numpy as np
from symlik.distributions import (
    exponential,
    normal,
    normal_mean,
    poisson,
    bernoulli,
    binomial,
    gamma,
    weibull,
    beta,
)


class TestExponential:
    """Test exponential distribution."""

    def test_params(self):
        model = exponential()
        assert model.params == ["lambda"]

    def test_custom_names(self):
        model = exponential(data_var="y", param="rate")
        assert model.params == ["rate"]

    def test_mle(self):
        model = exponential()
        data = {"x": [1, 2, 3, 4, 5]}
        mle, _ = model.mle(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})
        # MLE: λ̂ = 1/x̄ = 1/3
        assert mle["lambda"] == pytest.approx(1 / 3, rel=1e-5)

    def test_mle_large_sample(self):
        model = exponential()
        np.random.seed(42)
        samples = np.random.exponential(scale=2.0, size=1000)  # λ = 0.5
        data = {"x": samples.tolist()}
        mle, _ = model.mle(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})
        assert mle["lambda"] == pytest.approx(0.5, rel=0.1)

    def test_se(self):
        model = exponential()
        data = {"x": [1, 2, 3, 4, 5]}
        mle, _ = model.mle(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})
        se = model.se(mle, data)
        # SE(λ̂) = λ̂/√n
        expected_se = mle["lambda"] / math.sqrt(5)
        assert se["lambda"] == pytest.approx(expected_se, rel=0.01)


class TestNormal:
    """Test normal distribution (full parameterization)."""

    def test_params(self):
        model = normal()
        assert model.params == ["mu", "sigma2"]

    def test_custom_names(self):
        model = normal(data_var="y", mean="m", var="v")
        assert model.params == ["m", "v"]


class TestNormalMean:
    """Test normal distribution with known variance."""

    def test_params(self):
        model = normal_mean()
        assert model.params == ["mu"]

    def test_mle(self):
        model = normal_mean(known_var=1.0)
        data = {"x": [1, 2, 3, 4, 5]}
        mle, _ = model.mle(data=data, init={"mu": 0})
        # MLE: μ̂ = x̄ = 3
        assert mle["mu"] == pytest.approx(3.0, abs=1e-5)

    def test_mle_different_variance(self):
        model = normal_mean(known_var=4.0)
        data = {"x": [0, 2, 4, 6, 8]}
        mle, _ = model.mle(data=data, init={"mu": 0})
        assert mle["mu"] == pytest.approx(4.0, abs=1e-5)

    def test_se(self):
        model = normal_mean(known_var=1.0)
        data = {"x": [1, 2, 3, 4, 5]}
        mle, _ = model.mle(data=data, init={"mu": 0})
        se = model.se(mle, data)
        # SE(μ̂) = σ/√n = 1/√5
        expected_se = 1 / math.sqrt(5)
        assert se["mu"] == pytest.approx(expected_se, rel=0.01)


class TestPoisson:
    """Test Poisson distribution."""

    def test_params(self):
        model = poisson()
        assert model.params == ["lambda"]

    def test_mle(self):
        model = poisson()
        data = {"x": [2, 3, 1, 4, 2, 3]}  # mean = 2.5
        mle, _ = model.mle(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 100)})
        # MLE: λ̂ = x̄ = 2.5
        assert mle["lambda"] == pytest.approx(2.5, rel=1e-5)

    def test_mle_large_sample(self):
        model = poisson()
        np.random.seed(42)
        samples = np.random.poisson(lam=5.0, size=500)
        data = {"x": samples.tolist()}
        mle, _ = model.mle(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 100)})
        assert mle["lambda"] == pytest.approx(5.0, rel=0.1)


class TestBernoulli:
    """Test Bernoulli distribution."""

    def test_params(self):
        model = bernoulli()
        assert model.params == ["p"]

    def test_mle(self):
        model = bernoulli()
        data = {"x": [1, 0, 1, 1, 0, 1, 0, 1]}  # 5 successes, 8 trials
        mle, _ = model.mle(data=data, init={"p": 0.5}, bounds={"p": (0.01, 0.99)})
        # MLE: p̂ = 5/8 = 0.625
        assert mle["p"] == pytest.approx(0.625, rel=1e-5)

    def test_mle_all_success(self):
        model = bernoulli()
        data = {"x": [1, 1, 1, 1, 1]}
        mle, _ = model.mle(data=data, init={"p": 0.5}, bounds={"p": (0.01, 0.99)})
        assert mle["p"] == pytest.approx(0.99, abs=0.01)  # Bounded at 0.99

    def test_mle_all_failure(self):
        model = bernoulli()
        data = {"x": [0, 0, 0, 0, 0]}
        mle, _ = model.mle(data=data, init={"p": 0.5}, bounds={"p": (0.01, 0.99)})
        assert mle["p"] == pytest.approx(0.01, abs=0.01)  # Bounded at 0.01


class TestBinomial:
    """Test binomial distribution (single observation)."""

    def test_params(self):
        model = binomial()
        assert model.params == ["p"]

    def test_mle(self):
        model = binomial()
        data = {"k": 7, "n": 10}  # 7 successes in 10 trials
        mle, _ = model.mle(data=data, init={"p": 0.5}, bounds={"p": (0.01, 0.99)})
        # MLE: p̂ = k/n = 0.7
        assert mle["p"] == pytest.approx(0.7, rel=1e-3)


class TestGamma:
    """Test gamma distribution."""

    def test_params(self):
        model = gamma()
        assert model.params == ["alpha", "beta"]

    def test_custom_names(self):
        model = gamma(data_var="y", shape="k", rate="r")
        assert model.params == ["k", "r"]


class TestWeibull:
    """Test Weibull distribution."""

    def test_params(self):
        model = weibull()
        assert model.params == ["k", "lambda"]

    def test_custom_names(self):
        model = weibull(data_var="t", shape="shape", scale="scale")
        assert model.params == ["shape", "scale"]


class TestBeta:
    """Test beta distribution."""

    def test_params(self):
        model = beta()
        assert model.params == ["alpha", "beta"]

    def test_custom_names(self):
        model = beta(data_var="p", alpha="a", beta_param="b")
        assert model.params == ["a", "b"]


class TestDistributionScores:
    """Test that distributions have well-defined score functions."""

    def test_exponential_score(self):
        model = exponential()
        score = model.score()
        assert len(score) == 1

    def test_normal_score(self):
        model = normal()
        score = model.score()
        assert len(score) == 2

    def test_poisson_score(self):
        model = poisson()
        score = model.score()
        assert len(score) == 1

    def test_bernoulli_score(self):
        model = bernoulli()
        score = model.score()
        assert len(score) == 1

    def test_gamma_score(self):
        model = gamma()
        score = model.score()
        assert len(score) == 2

    def test_weibull_score(self):
        model = weibull()
        score = model.score()
        assert len(score) == 2

    def test_beta_score(self):
        model = beta()
        score = model.score()
        assert len(score) == 2


class TestDistributionHessians:
    """Test that distributions have well-defined Hessian matrices."""

    def test_exponential_hessian(self):
        model = exponential()
        hess = model.hessian()
        assert len(hess) == 1
        assert len(hess[0]) == 1

    def test_normal_hessian(self):
        model = normal()
        hess = model.hessian()
        assert len(hess) == 2
        assert len(hess[0]) == 2

    def test_poisson_hessian(self):
        model = poisson()
        hess = model.hessian()
        assert len(hess) == 1
        assert len(hess[0]) == 1


class TestDistributionEvaluation:
    """Test that distributions can be evaluated numerically."""

    def test_exponential_evaluate(self):
        model = exponential()
        # ℓ(λ=0.5 | x=[1,2]) = 2*log(0.5) - 0.5*(1+2) = 2*(-0.693) - 1.5 ≈ -2.886
        ll = model.evaluate({"x": [1, 2], "lambda": 0.5})
        expected = 2 * math.log(0.5) - 0.5 * 3
        assert ll == pytest.approx(expected, rel=1e-5)

    def test_poisson_evaluate(self):
        model = poisson()
        # ℓ(λ=2 | x=[1,2,3]) = sum(x)*log(λ) - n*λ = 6*log(2) - 3*2 = 6*0.693 - 6 ≈ -1.84
        ll = model.evaluate({"x": [1, 2, 3], "lambda": 2.0})
        expected = 6 * math.log(2) - 3 * 2
        assert ll == pytest.approx(expected, rel=1e-5)

    def test_bernoulli_evaluate(self):
        model = bernoulli()
        # ℓ(p=0.6 | x=[1,0,1]) = 2*log(0.6) + 1*log(0.4)
        ll = model.evaluate({"x": [1, 0, 1], "p": 0.6})
        expected = 2 * math.log(0.6) + 1 * math.log(0.4)
        assert ll == pytest.approx(expected, rel=1e-5)


class TestConvergenceProperties:
    """Test convergence properties of MLE estimates."""

    def test_exponential_consistency(self):
        """MLE should converge to true parameter as n increases."""
        model = exponential()
        true_lambda = 2.0

        np.random.seed(123)
        for n in [50, 200, 1000]:
            samples = np.random.exponential(scale=1/true_lambda, size=n)
            data = {"x": samples.tolist()}
            mle, _ = model.mle(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})
            # Larger samples should give estimates closer to true value
            if n >= 200:
                assert abs(mle["lambda"] - true_lambda) < 0.3

    def test_normal_consistency(self):
        """MLE for normal mean should converge to true parameter."""
        model = normal_mean(known_var=1.0)
        true_mu = 5.0

        np.random.seed(456)
        for n in [50, 200, 1000]:
            samples = np.random.normal(loc=true_mu, scale=1.0, size=n)
            data = {"x": samples.tolist()}
            mle, _ = model.mle(data=data, init={"mu": 0})
            if n >= 200:
                assert abs(mle["mu"] - true_mu) < 0.2


class TestGammaMLE:
    """Test gamma distribution MLE estimation."""

    def test_gamma_mle_basic(self):
        model = gamma()
        np.random.seed(42)
        # shape=2, rate=1
        samples = np.random.gamma(2.0, 1.0, size=200)
        data = {"x": samples.tolist()}
        mle, _ = model.mle(
            data=data,
            init={"alpha": 1.0, "beta": 1.0},
            bounds={"alpha": (0.1, 10), "beta": (0.1, 10)}
        )
        # Alpha should be close to 2, beta close to 1
        assert mle["alpha"] == pytest.approx(2.0, rel=0.3)
        assert mle["beta"] == pytest.approx(1.0, rel=0.3)

    def test_gamma_mle_different_params(self):
        model = gamma()
        np.random.seed(123)
        # shape=3, rate=2 (scale=0.5)
        samples = np.random.gamma(3.0, 0.5, size=300)
        data = {"x": samples.tolist()}
        mle, _ = model.mle(
            data=data,
            init={"alpha": 1.0, "beta": 1.0},
            bounds={"alpha": (0.1, 20), "beta": (0.1, 10)}
        )
        assert mle["alpha"] == pytest.approx(3.0, rel=0.3)
        # Rate beta = 1/scale = 2
        assert mle["beta"] == pytest.approx(2.0, rel=0.3)


class TestWeibullMLE:
    """Test Weibull distribution MLE estimation."""

    def test_weibull_mle_basic(self):
        model = weibull()
        np.random.seed(42)
        # k=2, lambda=1
        samples = np.random.weibull(2.0, size=200)
        data = {"x": samples.tolist()}
        mle, _ = model.mle(
            data=data,
            init={"k": 1.0, "lambda": 1.0},
            bounds={"k": (0.1, 10), "lambda": (0.1, 10)}
        )
        # Shape k should be close to 2
        assert mle["k"] == pytest.approx(2.0, rel=0.3)

    def test_weibull_mle_scale(self):
        model = weibull()
        np.random.seed(456)
        # k=1.5, lambda=2
        samples = 2.0 * np.random.weibull(1.5, size=300)
        data = {"x": samples.tolist()}
        mle, _ = model.mle(
            data=data,
            init={"k": 1.0, "lambda": 1.0},
            bounds={"k": (0.1, 10), "lambda": (0.1, 10)}
        )
        assert mle["k"] == pytest.approx(1.5, rel=0.3)
        assert mle["lambda"] == pytest.approx(2.0, rel=0.3)


class TestBetaMLE:
    """Test beta distribution MLE estimation."""

    def test_beta_mle_symmetric(self):
        model = beta()
        np.random.seed(42)
        # alpha=2, beta=2 (symmetric around 0.5)
        samples = np.random.beta(2.0, 2.0, size=200)
        data = {"x": samples.tolist()}
        mle, _ = model.mle(
            data=data,
            init={"alpha": 1.0, "beta": 1.0},
            bounds={"alpha": (0.1, 10), "beta": (0.1, 10)}
        )
        assert mle["alpha"] == pytest.approx(2.0, rel=0.4)
        assert mle["beta"] == pytest.approx(2.0, rel=0.4)

    def test_beta_mle_asymmetric(self):
        model = beta()
        np.random.seed(123)
        # alpha=5, beta=2 (skewed right)
        samples = np.random.beta(5.0, 2.0, size=300)
        data = {"x": samples.tolist()}
        mle, _ = model.mle(
            data=data,
            init={"alpha": 1.0, "beta": 1.0},
            bounds={"alpha": (0.1, 20), "beta": (0.1, 20)}
        )
        assert mle["alpha"] == pytest.approx(5.0, rel=0.4)
        assert mle["beta"] == pytest.approx(2.0, rel=0.4)


class TestNestedDataIndexing:
    """Test nested and complex data indexing scenarios."""

    def test_exponential_with_nested_sum(self):
        """Test that nested summation over data works correctly."""
        from symlik import evaluate

        # Construct expression: sum_{i=1}^n log(lambda) - lambda*x[i]
        expr = ["sum", "i", ["len", "x"],
                ["-", ["log", "lambda"], ["*", "lambda", ["@", "x", "i"]]]]

        data = {"x": [1.0, 2.0, 3.0], "lambda": 0.5}
        result = evaluate(expr, data)

        # Expected: 3*log(0.5) - 0.5*(1+2+3) = 3*(-0.693) - 3 = -5.08
        expected = 3 * math.log(0.5) - 0.5 * 6
        assert result == pytest.approx(expected, rel=1e-5)

    def test_double_indexing_pattern(self):
        """Test indexing with computed index."""
        from symlik import evaluate

        # Sum of x[i] for i from 1 to 3
        expr = ["sum", "i", 3, ["@", "x", "i"]]
        data = {"x": [10.0, 20.0, 30.0]}

        result = evaluate(expr, data)
        assert result == pytest.approx(60.0)

    def test_product_over_indexed_data(self):
        """Test product over indexed elements."""
        from symlik import evaluate

        # Product of x[i] for i from 1 to n
        expr = ["prod", "i", ["len", "x"], ["@", "x", "i"]]
        data = {"x": [2.0, 3.0, 4.0]}

        result = evaluate(expr, data)
        assert result == pytest.approx(24.0)

    def test_conditional_on_indexed_data(self):
        """Test if-then-else with indexed data."""
        from symlik import evaluate

        # if x[1] > 0 then x[2] else x[3] (using non-zero as truthy)
        expr = ["if", ["@", "x", 1], ["@", "x", 2], ["@", "x", 3]]
        data = {"x": [1.0, 20.0, 30.0]}

        result = evaluate(expr, data)
        assert result == pytest.approx(20.0)

        # With zero in first position
        data2 = {"x": [0.0, 20.0, 30.0]}
        result2 = evaluate(expr, data2)
        assert result2 == pytest.approx(30.0)
