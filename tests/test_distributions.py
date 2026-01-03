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
    lognormal,
    negative_binomial,
    student_t,
    uniform,
    laplace,
    geometric,
    pareto,
    cauchy,
    inverse_gaussian,
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
        fit = model.fit(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})
        # MLE: λ̂ = 1/x̄ = 1/3
        assert fit.params["lambda"] == pytest.approx(1 / 3, rel=1e-5)

    def test_mle_large_sample(self):
        model = exponential()
        np.random.seed(42)
        samples = np.random.exponential(scale=2.0, size=1000)  # λ = 0.5
        data = {"x": samples.tolist()}
        fit = model.fit(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})
        assert fit.params["lambda"] == pytest.approx(0.5, rel=0.1)

    def test_se(self):
        model = exponential()
        data = {"x": [1, 2, 3, 4, 5]}
        fit = model.fit(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})
        # se available via fit.se
        # SE(λ̂) = λ̂/√n
        expected_se = fit.params["lambda"] / math.sqrt(5)
        assert fit.se["lambda"] == pytest.approx(expected_se, rel=0.01)


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
        fit = model.fit(data=data, init={"mu": 0})
        # MLE: μ̂ = x̄ = 3
        assert fit.params["mu"] == pytest.approx(3.0, abs=1e-5)

    def test_mle_different_variance(self):
        model = normal_mean(known_var=4.0)
        data = {"x": [0, 2, 4, 6, 8]}
        fit = model.fit(data=data, init={"mu": 0})
        assert fit.params["mu"] == pytest.approx(4.0, abs=1e-5)

    def test_se(self):
        model = normal_mean(known_var=1.0)
        data = {"x": [1, 2, 3, 4, 5]}
        fit = model.fit(data=data, init={"mu": 0})
        # se available via fit.se
        # SE(μ̂) = σ/√n = 1/√5
        expected_se = 1 / math.sqrt(5)
        assert fit.se["mu"] == pytest.approx(expected_se, rel=0.01)


class TestPoisson:
    """Test Poisson distribution."""

    def test_params(self):
        model = poisson()
        assert model.params == ["lambda"]

    def test_mle(self):
        model = poisson()
        data = {"x": [2, 3, 1, 4, 2, 3]}  # mean = 2.5
        fit = model.fit(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 100)})
        # MLE: λ̂ = x̄ = 2.5
        assert fit.params["lambda"] == pytest.approx(2.5, rel=1e-5)

    def test_mle_large_sample(self):
        model = poisson()
        np.random.seed(42)
        samples = np.random.poisson(lam=5.0, size=500)
        data = {"x": samples.tolist()}
        fit = model.fit(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 100)})
        assert fit.params["lambda"] == pytest.approx(5.0, rel=0.1)


class TestBernoulli:
    """Test Bernoulli distribution."""

    def test_params(self):
        model = bernoulli()
        assert model.params == ["p"]

    def test_mle(self):
        model = bernoulli()
        data = {"x": [1, 0, 1, 1, 0, 1, 0, 1]}  # 5 successes, 8 trials
        fit = model.fit(data=data, init={"p": 0.5}, bounds={"p": (0.01, 0.99)})
        # MLE: p̂ = 5/8 = 0.625
        assert fit.params["p"] == pytest.approx(0.625, rel=1e-5)

    def test_mle_all_success(self):
        model = bernoulli()
        data = {"x": [1, 1, 1, 1, 1]}
        fit = model.fit(data=data, init={"p": 0.5}, bounds={"p": (0.01, 0.99)})
        assert fit.params["p"] == pytest.approx(0.99, abs=0.01)  # Bounded at 0.99

    def test_mle_all_failure(self):
        model = bernoulli()
        data = {"x": [0, 0, 0, 0, 0]}
        fit = model.fit(data=data, init={"p": 0.5}, bounds={"p": (0.01, 0.99)})
        assert fit.params["p"] == pytest.approx(0.01, abs=0.01)  # Bounded at 0.01


class TestBinomial:
    """Test binomial distribution (single observation)."""

    def test_params(self):
        model = binomial()
        assert model.params == ["p"]

    def test_mle(self):
        model = binomial()
        data = {"k": 7, "n": 10}  # 7 successes in 10 trials
        fit = model.fit(data=data, init={"p": 0.5}, bounds={"p": (0.01, 0.99)})
        # MLE: p̂ = k/n = 0.7
        assert fit.params["p"] == pytest.approx(0.7, rel=1e-3)


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
            fit = model.fit(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})
            # Larger samples should give estimates closer to true value
            if n >= 200:
                assert abs(fit.params["lambda"] - true_lambda) < 0.3

    def test_normal_consistency(self):
        """MLE for normal mean should converge to true parameter."""
        model = normal_mean(known_var=1.0)
        true_mu = 5.0

        np.random.seed(456)
        for n in [50, 200, 1000]:
            samples = np.random.normal(loc=true_mu, scale=1.0, size=n)
            data = {"x": samples.tolist()}
            fit = model.fit(data=data, init={"mu": 0})
            if n >= 200:
                assert abs(fit.params["mu"] - true_mu) < 0.2


class TestGammaMLE:
    """Test gamma distribution MLE estimation."""

    def test_gamma_mle_basic(self):
        model = gamma()
        np.random.seed(42)
        # shape=2, rate=1
        samples = np.random.gamma(2.0, 1.0, size=200)
        data = {"x": samples.tolist()}
        fit = model.fit(
            data=data,
            init={"alpha": 1.0, "beta": 1.0},
            bounds={"alpha": (0.1, 10), "beta": (0.1, 10)}
        )
        # Alpha should be close to 2, beta close to 1
        assert fit.params["alpha"] == pytest.approx(2.0, rel=0.3)
        assert fit.params["beta"] == pytest.approx(1.0, rel=0.3)

    def test_gamma_mle_different_params(self):
        model = gamma()
        np.random.seed(123)
        # shape=3, rate=2 (scale=0.5)
        samples = np.random.gamma(3.0, 0.5, size=300)
        data = {"x": samples.tolist()}
        fit = model.fit(
            data=data,
            init={"alpha": 1.0, "beta": 1.0},
            bounds={"alpha": (0.1, 20), "beta": (0.1, 10)}
        )
        assert fit.params["alpha"] == pytest.approx(3.0, rel=0.3)
        # Rate beta = 1/scale = 2
        assert fit.params["beta"] == pytest.approx(2.0, rel=0.3)


class TestWeibullMLE:
    """Test Weibull distribution MLE estimation."""

    def test_weibull_mle_basic(self):
        model = weibull()
        np.random.seed(42)
        # k=2, lambda=1
        samples = np.random.weibull(2.0, size=200)
        data = {"x": samples.tolist()}
        fit = model.fit(
            data=data,
            init={"k": 1.0, "lambda": 1.0},
            bounds={"k": (0.1, 10), "lambda": (0.1, 10)}
        )
        # Shape k should be close to 2
        assert fit.params["k"] == pytest.approx(2.0, rel=0.3)

    def test_weibull_mle_scale(self):
        model = weibull()
        np.random.seed(456)
        # k=1.5, lambda=2
        samples = 2.0 * np.random.weibull(1.5, size=300)
        data = {"x": samples.tolist()}
        fit = model.fit(
            data=data,
            init={"k": 1.0, "lambda": 1.0},
            bounds={"k": (0.1, 10), "lambda": (0.1, 10)}
        )
        assert fit.params["k"] == pytest.approx(1.5, rel=0.3)
        assert fit.params["lambda"] == pytest.approx(2.0, rel=0.3)


class TestBetaMLE:
    """Test beta distribution MLE estimation."""

    def test_beta_mle_symmetric(self):
        model = beta()
        np.random.seed(42)
        # alpha=2, beta=2 (symmetric around 0.5)
        samples = np.random.beta(2.0, 2.0, size=200)
        data = {"x": samples.tolist()}
        fit = model.fit(
            data=data,
            init={"alpha": 1.0, "beta": 1.0},
            bounds={"alpha": (0.1, 10), "beta": (0.1, 10)}
        )
        assert fit.params["alpha"] == pytest.approx(2.0, rel=0.4)
        assert fit.params["beta"] == pytest.approx(2.0, rel=0.4)

    def test_beta_mle_asymmetric(self):
        model = beta()
        np.random.seed(123)
        # alpha=5, beta=2 (skewed right)
        samples = np.random.beta(5.0, 2.0, size=300)
        data = {"x": samples.tolist()}
        fit = model.fit(
            data=data,
            init={"alpha": 1.0, "beta": 1.0},
            bounds={"alpha": (0.1, 20), "beta": (0.1, 20)}
        )
        assert fit.params["alpha"] == pytest.approx(5.0, rel=0.4)
        assert fit.params["beta"] == pytest.approx(2.0, rel=0.4)


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


# ============================================================
# Tests for New Distribution Families
# ============================================================


class TestLognormal:
    """Test log-normal distribution."""

    def test_params(self):
        model = lognormal()
        assert model.params == ["mu", "sigma2"]

    def test_custom_names(self):
        model = lognormal(data_var="y", mu="m", sigma2="s2")
        assert model.params == ["m", "s2"]

    def test_mle(self):
        model = lognormal()
        np.random.seed(42)
        # mu=1.0, sigma=0.5
        samples = np.random.lognormal(mean=1.0, sigma=0.5, size=500)
        data = {"x": samples.tolist()}
        # Use sample statistics as initial values
        log_samples = np.log(samples)
        fit = model.fit(
            data=data,
            init={"mu": float(np.mean(log_samples)), "sigma2": float(np.var(log_samples))},
            bounds={"mu": (-5, 5), "sigma2": (0.01, 2.0)}
        )
        # MLE: mu = mean(log(x)), sigma2 = var(log(x))
        assert fit.params["mu"] == pytest.approx(1.0, rel=0.1)
        assert fit.params["sigma2"] == pytest.approx(0.25, rel=0.2)  # sigma^2 = 0.5^2 = 0.25

    def test_score(self):
        model = lognormal()
        score = model.score()
        assert len(score) == 2


class TestNegativeBinomial:
    """Test negative binomial distribution."""

    def test_params(self):
        model = negative_binomial()
        assert model.params == ["r", "p"]

    def test_custom_names(self):
        model = negative_binomial(data_var="counts", r="size", p="prob")
        assert model.params == ["size", "prob"]

    def test_mle(self):
        model = negative_binomial()
        np.random.seed(42)
        # r=5, p=0.6 -> mean = r(1-p)/p = 5*0.4/0.6 ≈ 3.33
        samples = np.random.negative_binomial(n=5, p=0.6, size=300)
        data = {"x": samples.tolist()}
        fit = model.fit(
            data=data,
            init={"r": 3.0, "p": 0.5},
            bounds={"r": (0.5, 20), "p": (0.1, 0.9)}
        )
        assert fit.params["r"] == pytest.approx(5.0, rel=0.3)
        assert fit.params["p"] == pytest.approx(0.6, rel=0.2)

    def test_score(self):
        model = negative_binomial()
        score = model.score()
        assert len(score) == 2


class TestStudentT:
    """Test Student's t distribution."""

    def test_params(self):
        model = student_t()
        assert model.params == ["mu", "sigma2", "nu"]

    def test_custom_names(self):
        model = student_t(data_var="y", mu="loc", sigma2="scale", nu="df")
        assert model.params == ["loc", "scale", "df"]

    def test_mle_approaches_normal(self):
        """With high df, should behave like normal."""
        model = student_t()
        np.random.seed(42)
        # Large nu -> approaches normal
        samples = np.random.normal(loc=3.0, scale=2.0, size=200)
        data = {"x": samples.tolist()}
        # Use sample statistics as initial values
        fit = model.fit(
            data=data,
            init={"mu": float(np.mean(samples)), "sigma2": float(np.var(samples)), "nu": 30.0},
            bounds={"sigma2": (0.01, None), "nu": (2.0, 100.0)}
        )
        assert fit.params["mu"] == pytest.approx(3.0, rel=0.2)
        # For t-distribution, relationship between sigma2 and sample variance is complex
        # Just check it's in a reasonable range
        assert 1.0 < fit.params["sigma2"] < 10.0

    def test_score(self):
        model = student_t()
        score = model.score()
        assert len(score) == 3


class TestUniform:
    """Test uniform distribution."""

    def test_params(self):
        model = uniform()
        assert model.params == ["a", "b"]

    def test_custom_names(self):
        model = uniform(data_var="y", a="lower", b="upper")
        assert model.params == ["lower", "upper"]

    def test_evaluate(self):
        model = uniform()
        # ℓ(a=0, b=1 | x=[0.2, 0.5, 0.8]) = -3*log(1-0) = 0
        ll = model.evaluate({"x": [0.2, 0.5, 0.8], "a": 0.0, "b": 1.0})
        assert ll == pytest.approx(0.0)

        # ℓ(a=0, b=2 | x=[0.2, 0.5, 0.8]) = -3*log(2) ≈ -2.08
        ll2 = model.evaluate({"x": [0.2, 0.5, 0.8], "a": 0.0, "b": 2.0})
        assert ll2 == pytest.approx(-3 * math.log(2))

    def test_score(self):
        model = uniform()
        score = model.score()
        assert len(score) == 2


class TestLaplace:
    """Test Laplace distribution."""

    def test_params(self):
        model = laplace()
        assert model.params == ["mu", "b"]

    def test_custom_names(self):
        model = laplace(data_var="y", mu="loc", b="scale")
        assert model.params == ["loc", "scale"]

    def test_mle(self):
        model = laplace()
        np.random.seed(42)
        # mu=2.0, b=1.0
        samples = np.random.laplace(loc=2.0, scale=1.0, size=500)
        data = {"x": samples.tolist()}
        # Use median as initial value for mu (close to MLE)
        fit = model.fit(
            data=data,
            init={"mu": float(np.median(samples)), "b": 1.0},
            bounds={"b": (0.01, None)}
        )
        # MLE: mu ≈ median, b ≈ mean absolute deviation
        assert fit.params["mu"] == pytest.approx(2.0, rel=0.1)
        assert fit.params["b"] == pytest.approx(1.0, rel=0.2)

    def test_robust_to_outliers(self):
        """Laplace should be more robust than normal to outliers."""
        model = laplace()
        # Data with an outlier
        data = {"x": [1.0, 2.0, 2.5, 3.0, 100.0]}
        # Use median as initial value
        fit = model.fit(
            data=data,
            init={"mu": 2.5, "b": 1.0},
            bounds={"b": (0.01, None)}
        )
        # Mu should be close to the median (2.5), not pulled by outlier
        assert fit.params["mu"] == pytest.approx(2.5, rel=0.5)

    def test_score(self):
        model = laplace()
        score = model.score()
        assert len(score) == 2


class TestGeometric:
    """Test geometric distribution."""

    def test_params(self):
        model = geometric()
        assert model.params == ["p"]

    def test_custom_names(self):
        model = geometric(data_var="failures", p="prob")
        assert model.params == ["prob"]

    def test_mle(self):
        model = geometric()
        np.random.seed(42)
        # p=0.3 -> mean = (1-p)/p = 0.7/0.3 ≈ 2.33
        samples = np.random.geometric(p=0.3, size=500) - 1  # numpy counts trials, we want failures
        data = {"x": samples.tolist()}
        fit = model.fit(
            data=data,
            init={"p": 0.5},
            bounds={"p": (0.01, 0.99)}
        )
        # MLE: p = n/(n + sum(x)) = 1/(1 + mean(x))
        assert fit.params["p"] == pytest.approx(0.3, rel=0.15)

    def test_score(self):
        model = geometric()
        score = model.score()
        assert len(score) == 1


class TestPareto:
    """Test Pareto distribution."""

    def test_params(self):
        model = pareto()
        assert model.params == ["alpha", "x_min"]

    def test_custom_names(self):
        model = pareto(data_var="income", alpha="shape", x_min="scale")
        assert model.params == ["shape", "scale"]

    def test_evaluate(self):
        """Test log-likelihood evaluation for Pareto distribution."""
        model = pareto()
        # For Pareto: ℓ = n*log(α) + n*α*log(xₘ) - (α+1)*Σlog(xᵢ)
        # With x=[2,3,4], alpha=2, x_min=1:
        # ℓ = 3*log(2) + 3*2*log(1) - 3*Σlog([2,3,4])
        #   = 3*0.693 + 0 - 3*(0.693+1.099+1.386)
        #   = 2.079 - 3*3.178 = 2.079 - 9.534 = -7.455
        ll = model.evaluate({"x": [2.0, 3.0, 4.0], "alpha": 2.0, "x_min": 1.0})
        expected = 3 * math.log(2) - 3 * (math.log(2) + math.log(3) + math.log(4))
        assert ll == pytest.approx(expected, rel=1e-5)

    def test_score_at_mle(self):
        """Test that the score is zero at the analytic MLE."""
        from symlik import evaluate

        np.random.seed(42)
        u = np.random.uniform(0, 1, size=500)
        samples = 1.0 * (1 - u) ** (-1 / 2.5)
        x_min_data = min(samples)

        # Analytic MLE for alpha given x_min = min(x)
        alpha_analytic = len(samples) / np.sum(np.log(samples) - np.log(x_min_data))

        # Verify score wrt alpha is zero at the analytic MLE
        model = pareto()
        score_exprs = model.score()
        data = {"x": samples.tolist(), "alpha": alpha_analytic, "x_min": x_min_data}
        grad_alpha = evaluate(score_exprs[0], data)

        # Score should be (approximately) zero at MLE
        assert grad_alpha == pytest.approx(0.0, abs=1e-6)
        # And the analytic MLE should be close to true value
        assert alpha_analytic == pytest.approx(2.5, rel=0.05)

    def test_score(self):
        model = pareto()
        score = model.score()
        assert len(score) == 2


class TestCauchy:
    """Test Cauchy distribution."""

    def test_params(self):
        model = cauchy()
        assert model.params == ["x0", "gamma"]

    def test_custom_names(self):
        model = cauchy(data_var="y", x0="loc", gamma_param="scale")
        assert model.params == ["loc", "scale"]

    def test_evaluate(self):
        model = cauchy()
        # Single point at the mode
        ll = model.evaluate({"x": [0.0], "x0": 0.0, "gamma": 1.0})
        # f(0|0,1) = 1/(π*1*(1+0)) = 1/π
        expected = math.log(1 / math.pi)
        assert ll == pytest.approx(expected, rel=1e-5)

    def test_mle(self):
        """Basic Cauchy MLE test - note Cauchy is notoriously hard to estimate."""
        model = cauchy()
        np.random.seed(42)
        # Standard Cauchy (x0=0, gamma=1)
        samples = np.random.standard_cauchy(size=200)
        # Trim extreme outliers for stability
        samples = samples[(samples > -50) & (samples < 50)]
        data = {"x": samples.tolist()}
        fit = model.fit(
            data=data,
            init={"x0": 0.0, "gamma": 1.0},
            bounds={"gamma": (0.1, 10)}
        )
        # Location should be near 0 (median)
        assert abs(fit.params["x0"]) < 1.0
        # Scale should be near 1
        assert fit.params["gamma"] == pytest.approx(1.0, rel=0.5)

    def test_score(self):
        model = cauchy()
        score = model.score()
        assert len(score) == 2


class TestInverseGaussian:
    """Test inverse Gaussian (Wald) distribution."""

    def test_params(self):
        model = inverse_gaussian()
        assert model.params == ["mu", "lambda"]

    def test_custom_names(self):
        model = inverse_gaussian(data_var="t", mu="mean", lambda_param="shape")
        assert model.params == ["mean", "shape"]

    def test_mle(self):
        model = inverse_gaussian()
        np.random.seed(42)
        # mu=2.0, lambda=3.0
        samples = np.random.wald(mean=2.0, scale=3.0, size=500)
        data = {"x": samples.tolist()}
        fit = model.fit(
            data=data,
            init={"mu": 1.0, "lambda": 1.0},
            bounds={"mu": (0.1, 10), "lambda": (0.1, 20)}
        )
        # MLE: mu = mean(x)
        assert fit.params["mu"] == pytest.approx(2.0, rel=0.15)
        assert fit.params["lambda"] == pytest.approx(3.0, rel=0.3)

    def test_score(self):
        model = inverse_gaussian()
        score = model.score()
        assert len(score) == 2


class TestNewDistributionScores:
    """Test that new distributions have well-defined score functions."""

    def test_lognormal_score(self):
        model = lognormal()
        score = model.score()
        assert len(score) == 2

    def test_negative_binomial_score(self):
        model = negative_binomial()
        score = model.score()
        assert len(score) == 2

    def test_student_t_score(self):
        model = student_t()
        score = model.score()
        assert len(score) == 3

    def test_uniform_score(self):
        model = uniform()
        score = model.score()
        assert len(score) == 2

    def test_laplace_score(self):
        model = laplace()
        score = model.score()
        assert len(score) == 2

    def test_geometric_score(self):
        model = geometric()
        score = model.score()
        assert len(score) == 1

    def test_pareto_score(self):
        model = pareto()
        score = model.score()
        assert len(score) == 2

    def test_cauchy_score(self):
        model = cauchy()
        score = model.score()
        assert len(score) == 2

    def test_inverse_gaussian_score(self):
        model = inverse_gaussian()
        score = model.score()
        assert len(score) == 2


class TestNewDistributionHessians:
    """Test that new distributions have well-defined Hessian matrices."""

    def test_lognormal_hessian(self):
        model = lognormal()
        hess = model.hessian()
        assert len(hess) == 2
        assert len(hess[0]) == 2

    def test_negative_binomial_hessian(self):
        model = negative_binomial()
        hess = model.hessian()
        assert len(hess) == 2
        assert len(hess[0]) == 2

    def test_student_t_hessian(self):
        model = student_t()
        hess = model.hessian()
        assert len(hess) == 3
        assert len(hess[0]) == 3

    def test_laplace_hessian(self):
        model = laplace()
        hess = model.hessian()
        assert len(hess) == 2
        assert len(hess[0]) == 2

    def test_geometric_hessian(self):
        model = geometric()
        hess = model.hessian()
        assert len(hess) == 1
        assert len(hess[0]) == 1

    def test_pareto_hessian(self):
        model = pareto()
        hess = model.hessian()
        assert len(hess) == 2
        assert len(hess[0]) == 2

    def test_inverse_gaussian_hessian(self):
        model = inverse_gaussian()
        hess = model.hessian()
        assert len(hess) == 2
        assert len(hess[0]) == 2
