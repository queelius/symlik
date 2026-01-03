"""Tests for symlik.contribution module."""

import math
import pytest
import numpy as np
from symlik import ContributionModel
from symlik.contributions import (
    complete_exponential,
    right_censored_exponential,
    left_censored_exponential,
    interval_censored_exponential,
    complete_weibull,
    right_censored_weibull,
    complete_normal,
    complete_poisson,
    complete_bernoulli,
    series_exponential_known_cause,
    series_exponential_masked_cause,
    series_exponential_right_censored,
)


class TestContributionModelBasics:
    """Test basic ContributionModel functionality."""

    def test_init(self):
        model = ContributionModel(
            params=["lambda"],
            type_column="obs_type",
            contributions={
                "complete": complete_exponential(),
            }
        )
        assert model.params == ["lambda"]
        assert model.type_column == "obs_type"

    def test_repr(self):
        model = ContributionModel(
            params=["lambda"],
            type_column="type",
            contributions={
                "A": complete_exponential(),
                "B": right_censored_exponential(),
            }
        )
        repr_str = repr(model)
        assert "ContributionModel" in repr_str
        assert "lambda" in repr_str

    def test_single_type_matches_standard_model(self):
        """Single contribution type should behave like standard exponential model."""
        from symlik.distributions import exponential

        # Standard model
        std_model = exponential()

        # Contribution model with single type
        contrib_model = ContributionModel(
            params=["lambda"],
            type_column="obs_type",
            contributions={"complete": complete_exponential()},
        )

        data_std = {"x": [1.0, 2.0, 3.0]}
        data_contrib = {
            "obs_type": ["complete", "complete", "complete"],
            "t": [1.0, 2.0, 3.0],
        }

        # MLEs should match
        fit_std = std_model.fit(data=data_std, init={"lambda": 1.0})
        fit_contrib = contrib_model.fit(data=data_contrib, init={"lambda": 1.0})

        assert fit_std.params["lambda"] == pytest.approx(fit_contrib.params["lambda"], rel=1e-4)


class TestContributionModelMLE:
    """Test MLE estimation with mixed observation types."""

    def test_mixed_complete_and_censored_exponential(self):
        """Test MLE with mixed complete and right-censored observations."""
        model = ContributionModel(
            params=["lambda"],
            type_column="obs_type",
            contributions={
                "complete": complete_exponential(),
                "censored": right_censored_exponential(),
            },
        )

        # Generate data: some complete, some censored
        # For exponential, MLE with censoring is n_complete / (sum of all times)
        data = {
            "obs_type": ["complete", "complete", "censored", "censored"],
            "t": [1.0, 2.0, 3.0, 4.0],
        }

        fit = model.fit(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})

        # With 2 complete and 2 censored, MLE = 2 / (1+2+3+4) = 0.2
        assert fit.params["lambda"] == pytest.approx(0.2, rel=0.05)

    def test_all_censored_gives_boundary_rate(self):
        """All censored observations - likelihood is monotonic, hits boundary."""
        model = ContributionModel(
            params=["lambda"],
            type_column="obs_type",
            contributions={
                "complete": complete_exponential(),
                "censored": right_censored_exponential(),
            },
        )

        data = {
            "obs_type": ["censored", "censored", "censored"],
            "t": [1.0, 2.0, 3.0],
        }

        # With all censored, log L = -lambda * sum(t) is monotonically increasing as lambda -> 0
        # The optimizer should hit the lower bound
        fit = model.fit(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.001, 10)})

        # Rate should be at or near the lower bound (likelihood maximized at lambda=0)
        assert fit.params["lambda"] <= 0.1  # Should be small

    def test_mle_with_larger_sample(self):
        """Test MLE convergence with larger sample."""
        np.random.seed(42)

        model = ContributionModel(
            params=["lambda"],
            type_column="obs_type",
            contributions={
                "complete": complete_exponential(),
                "censored": right_censored_exponential(),
            },
        )

        true_lambda = 0.5
        n = 100

        # Generate exponential data
        times = np.random.exponential(1/true_lambda, n)

        # Randomly censor some (20%)
        censor_time = 3.0
        obs_types = ["complete" if t < censor_time else "censored" for t in times]
        observed_times = [min(t, censor_time) for t in times]

        data = {
            "obs_type": obs_types,
            "t": observed_times,
        }

        fit = model.fit(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})

        # MLE should be reasonably close to true value
        assert fit.params["lambda"] == pytest.approx(true_lambda, rel=0.3)


class TestContributionModelEvaluation:
    """Test log-likelihood evaluation."""

    def test_evaluate_matches_manual(self):
        """Evaluate should match manual calculation."""
        model = ContributionModel(
            params=["lambda"],
            type_column="obs_type",
            contributions={
                "complete": complete_exponential(),
                "censored": right_censored_exponential(),
            },
        )

        data = {
            "obs_type": ["complete", "censored"],
            "t": [1.0, 2.0],
            "lambda": 0.5,
        }

        ll = model.evaluate(data)

        # Manual: log(0.5) - 0.5*1 + (-0.5*2) = -0.693 - 0.5 - 1 = -2.193
        expected = math.log(0.5) - 0.5 * 1.0 - 0.5 * 2.0
        assert ll == pytest.approx(expected, rel=1e-4)


class TestContributionModelSE:
    """Test standard error computation."""

    def test_se_computed(self):
        """Standard errors should be computable."""
        model = ContributionModel(
            params=["lambda"],
            type_column="obs_type",
            contributions={
                "complete": complete_exponential(),
                "censored": right_censored_exponential(),
            },
        )

        data = {
            "obs_type": ["complete", "complete", "complete", "censored", "censored"],
            "t": [1.0, 2.0, 1.5, 3.0, 4.0],
        }

        fit = model.fit(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})

        assert "lambda" in fit.se
        assert fit.se["lambda"] > 0
        assert np.isfinite(fit.se["lambda"])


class TestContributionModelErrors:
    """Test error handling."""

    def test_missing_type_column_raises(self):
        model = ContributionModel(
            params=["lambda"],
            type_column="obs_type",
            contributions={"complete": complete_exponential()},
        )

        data = {"t": [1.0, 2.0]}  # Missing obs_type

        with pytest.raises(ValueError, match="Type column"):
            model.fit(data=data, init={"lambda": 1.0})

    def test_unknown_type_raises(self):
        model = ContributionModel(
            params=["lambda"],
            type_column="obs_type",
            contributions={"complete": complete_exponential()},
        )

        data = {
            "obs_type": ["complete", "unknown"],  # "unknown" not defined
            "t": [1.0, 2.0],
        }

        with pytest.raises(ValueError, match="Unknown observation types"):
            model.fit(data=data, init={"lambda": 1.0})


class TestContributionModelMultipleParams:
    """Test models with multiple parameters."""

    def test_weibull_two_params(self):
        """Test Weibull model with shape and scale parameters."""
        model = ContributionModel(
            params=["k", "lambda"],
            type_column="obs_type",
            contributions={
                "complete": complete_weibull(),
                "censored": right_censored_weibull(),
            },
        )

        np.random.seed(42)
        true_k = 2.0
        true_lambda = 1.5

        # Generate Weibull data
        n = 100
        times = true_lambda * np.random.weibull(true_k, n)

        # Censor some
        censor_time = 2.0
        obs_types = ["complete" if t < censor_time else "censored" for t in times]
        observed_times = [min(t, censor_time) for t in times]

        data = {
            "obs_type": obs_types,
            "t": observed_times,
        }

        fit = model.fit(
            data=data,
            init={"k": 1.0, "lambda": 1.0},
            bounds={"k": (0.1, 10), "lambda": (0.1, 10)},
        )

        # Parameters should be in reasonable range
        assert 1.0 < fit.params["k"] < 4.0
        assert 0.5 < fit.params["lambda"] < 3.0


class TestContributionConstructors:
    """Test the contribution constructor functions."""

    def test_complete_exponential(self):
        contrib = complete_exponential()
        assert isinstance(contrib, list)
        # Should contain log and multiplication
        assert "log" in str(contrib)

    def test_right_censored_exponential(self):
        contrib = right_censored_exponential()
        assert isinstance(contrib, list)
        # Should be simpler (no log term)

    def test_left_censored_exponential(self):
        contrib = left_censored_exponential()
        assert isinstance(contrib, list)
        # Should contain log and exp

    def test_interval_censored_exponential(self):
        contrib = interval_censored_exponential()
        assert isinstance(contrib, list)

    def test_complete_weibull(self):
        contrib = complete_weibull()
        assert isinstance(contrib, list)

    def test_right_censored_weibull(self):
        contrib = right_censored_weibull()
        assert isinstance(contrib, list)

    def test_complete_normal(self):
        contrib = complete_normal()
        assert isinstance(contrib, list)

    def test_complete_poisson(self):
        contrib = complete_poisson()
        assert isinstance(contrib, list)

    def test_complete_bernoulli(self):
        contrib = complete_bernoulli()
        assert isinstance(contrib, list)


class TestContributionConstructorCustomNames:
    """Test contribution constructors with custom variable names."""

    def test_exponential_custom_names(self):
        contrib = complete_exponential(time_var="duration", rate="rate")
        # The expression should reference "duration" and "rate"
        assert "duration" in str(contrib)
        assert "rate" in str(contrib)

    def test_weibull_custom_names(self):
        contrib = complete_weibull(time_var="t", shape="alpha", scale="beta")
        assert "alpha" in str(contrib)
        assert "beta" in str(contrib)


class TestContributionModelScore:
    """Test score (gradient) computation."""

    def test_score_exists(self):
        model = ContributionModel(
            params=["lambda"],
            type_column="obs_type",
            contributions={
                "complete": complete_exponential(),
            },
        )

        score = model.score()
        assert len(score) == 1
        assert isinstance(score[0], (list, int, float, str))


class TestContributionModelHessian:
    """Test Hessian computation."""

    def test_hessian_exists(self):
        model = ContributionModel(
            params=["lambda"],
            type_column="obs_type",
            contributions={
                "complete": complete_exponential(),
            },
        )

        hess = model.hessian()
        assert len(hess) == 1
        assert len(hess[0]) == 1

    def test_hessian_two_params(self):
        model = ContributionModel(
            params=["k", "lambda"],
            type_column="obs_type",
            contributions={
                "complete": complete_weibull(),
            },
        )

        hess = model.hessian()
        assert len(hess) == 2
        assert len(hess[0]) == 2


class TestEmptyTypeCategory:
    """Test handling of empty type categories."""

    def test_empty_category_handled(self):
        """Model should handle case where one type has no observations."""
        model = ContributionModel(
            params=["lambda"],
            type_column="obs_type",
            contributions={
                "complete": complete_exponential(),
                "censored": right_censored_exponential(),
            },
        )

        # All complete, no censored
        data = {
            "obs_type": ["complete", "complete", "complete"],
            "t": [1.0, 2.0, 3.0],
        }

        fit = model.fit(data=data, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})

        # Should work and give MLE = 1/mean = 1/2 = 0.5
        assert fit.params["lambda"] == pytest.approx(0.5, rel=0.05)


class TestContributionModelNumericalMethods:
    """Test numerical evaluation methods."""

    def test_score_at(self):
        model = ContributionModel(
            params=["lambda"],
            type_column="obs_type",
            contributions={"complete": complete_exponential()},
        )

        data = {
            "obs_type": ["complete", "complete"],
            "t": [1.0, 2.0],
            "lambda": 0.5,
        }

        score = model.score_at(data)
        assert len(score) == 1
        assert np.isfinite(score[0])

    def test_hessian_at(self):
        model = ContributionModel(
            params=["lambda"],
            type_column="obs_type",
            contributions={"complete": complete_exponential()},
        )

        data = {
            "obs_type": ["complete", "complete"],
            "t": [1.0, 2.0],
            "lambda": 0.5,
        }

        hess = model.hessian_at(data)
        assert hess.shape == (1, 1)
        assert np.isfinite(hess[0, 0])

    def test_information_at(self):
        model = ContributionModel(
            params=["lambda"],
            type_column="obs_type",
            contributions={"complete": complete_exponential()},
        )

        data = {
            "obs_type": ["complete", "complete"],
            "t": [1.0, 2.0],
            "lambda": 0.5,
        }

        info = model.information_at(data)
        # Information should be positive (for well-behaved likelihood)
        assert info[0, 0] > 0


# ============================================================
# Series System Contribution Tests
# ============================================================

class TestSeriesExponentialContributors:
    """Test series system contribution constructors."""

    def test_known_cause_constructor(self):
        contrib = series_exponential_known_cause()
        assert isinstance(contrib, list)
        # Should contain log for hazard term
        assert "log" in str(contrib)

    def test_masked_cause_constructor(self):
        contrib = series_exponential_masked_cause()
        assert isinstance(contrib, list)
        assert "log" in str(contrib)

    def test_right_censored_constructor(self):
        contrib = series_exponential_right_censored()
        assert isinstance(contrib, list)
        # Survival term only, no log

    def test_custom_rates(self):
        rates = ["r1", "r2"]
        contrib = series_exponential_known_cause(rates=rates, cause_index=1)
        assert "r1" in str(contrib)
        assert "r2" in str(contrib)

    def test_custom_candidate_indices(self):
        rates = ["lambda1", "lambda2", "lambda3"]
        # Candidate set {1, 2} means indices [0, 1]
        contrib = series_exponential_masked_cause(rates=rates, candidate_indices=[0, 1])
        assert "lambda1" in str(contrib)
        assert "lambda2" in str(contrib)


class TestSeriesSystemMLE:
    """Test MLE for series system with masked cause."""

    def test_known_cause_mle(self):
        """Test MLE with only known-cause observations."""
        model = ContributionModel(
            params=["lambda1", "lambda2"],
            type_column="obs_type",
            contributions={
                "known_1": series_exponential_known_cause(
                    rates=["lambda1", "lambda2"], cause_index=0
                ),
                "known_2": series_exponential_known_cause(
                    rates=["lambda1", "lambda2"], cause_index=1
                ),
            },
        )

        # Simple data: 2 failures from component 1, 2 from component 2
        data = {
            "obs_type": ["known_1", "known_1", "known_2", "known_2"],
            "t": [1.0, 1.0, 1.0, 1.0],
        }

        fit = model.fit(
            data=data,
            init={"lambda1": 0.5, "lambda2": 0.5},
            bounds={"lambda1": (0.01, 10), "lambda2": (0.01, 10)},
        )

        # With equal failures and equal times, rates should be similar
        assert 0.1 < fit.params["lambda1"] < 2.0
        assert 0.1 < fit.params["lambda2"] < 2.0

    def test_mixed_known_and_censored(self):
        """Test MLE with known cause and right-censored observations."""
        model = ContributionModel(
            params=["lambda1", "lambda2"],
            type_column="obs_type",
            contributions={
                "known_1": series_exponential_known_cause(
                    rates=["lambda1", "lambda2"], cause_index=0
                ),
                "known_2": series_exponential_known_cause(
                    rates=["lambda1", "lambda2"], cause_index=1
                ),
                "censored": series_exponential_right_censored(
                    rates=["lambda1", "lambda2"]
                ),
            },
        )

        data = {
            "obs_type": ["known_1", "known_2", "censored", "censored"],
            "t": [1.0, 1.0, 2.0, 2.0],
        }

        fit = model.fit(
            data=data,
            init={"lambda1": 0.5, "lambda2": 0.5},
            bounds={"lambda1": (0.01, 10), "lambda2": (0.01, 10)},
        )

        assert fit.params["lambda1"] > 0
        assert fit.params["lambda2"] > 0

    def test_masked_cause_mle(self):
        """Test MLE with masked cause observations."""
        model = ContributionModel(
            params=["lambda1", "lambda2"],
            type_column="obs_type",
            contributions={
                "masked_12": series_exponential_masked_cause(
                    rates=["lambda1", "lambda2"], candidate_indices=[0, 1]
                ),
            },
        )

        # All observations have both components as candidates
        data = {
            "obs_type": ["masked_12"] * 10,
            "t": [1.0] * 10,
        }

        fit = model.fit(
            data=data,
            init={"lambda1": 0.5, "lambda2": 0.5},
            bounds={"lambda1": (0.01, 10), "lambda2": (0.01, 10)},
        )

        # With symmetric data and equal masking, rates should be similar
        assert 0.1 < fit.params["lambda1"] < 5.0
        assert 0.1 < fit.params["lambda2"] < 5.0

    def test_three_component_series(self):
        """Test 3-component series system."""
        np.random.seed(42)

        rates = ["lambda1", "lambda2", "lambda3"]
        model = ContributionModel(
            params=rates,
            type_column="obs_type",
            contributions={
                "known_1": series_exponential_known_cause(rates=rates, cause_index=0),
                "known_2": series_exponential_known_cause(rates=rates, cause_index=1),
                "known_3": series_exponential_known_cause(rates=rates, cause_index=2),
                "censored": series_exponential_right_censored(rates=rates),
            },
        )

        # Generate synthetic data
        true_rates = [0.3, 0.5, 0.2]
        n = 50
        obs_types = []
        times = []

        for _ in range(n):
            lifetimes = [np.random.exponential(1/r) for r in true_rates]
            sys_time = min(lifetimes)
            cause = lifetimes.index(sys_time)

            if sys_time > 5.0:
                obs_types.append("censored")
                times.append(5.0)
            else:
                obs_types.append(f"known_{cause+1}")
                times.append(sys_time)

        data = {"obs_type": obs_types, "t": times}

        fit = model.fit(
            data=data,
            init={r: 0.5 for r in rates},
            bounds={r: (0.01, 5) for r in rates},
        )

        # MLEs should be in reasonable range
        for r in rates:
            assert 0.05 < fit.params[r] < 2.0


class TestSeriesSystemEvaluation:
    """Test log-likelihood evaluation for series systems."""

    def test_known_cause_evaluation(self):
        """Evaluate known-cause log-likelihood manually."""
        from symlik.evaluate import evaluate

        contrib = series_exponential_known_cause(
            rates=["lambda1", "lambda2"], cause_index=0
        )

        env = {"t": 1.0, "lambda1": 0.5, "lambda2": 0.3}
        ll = evaluate(contrib, env)

        # Manual: -t*(λ₁+λ₂) + log(λ₁) = -1*(0.5+0.3) + log(0.5) = -0.8 - 0.693 = -1.493
        expected = -1.0 * (0.5 + 0.3) + math.log(0.5)
        assert ll == pytest.approx(expected, rel=1e-4)

    def test_masked_cause_evaluation(self):
        """Evaluate masked-cause log-likelihood manually."""
        from symlik.evaluate import evaluate

        contrib = series_exponential_masked_cause(
            rates=["lambda1", "lambda2"], candidate_indices=[0, 1]
        )

        env = {"t": 1.0, "lambda1": 0.5, "lambda2": 0.3}
        ll = evaluate(contrib, env)

        # Manual: -t*(λ₁+λ₂) + log(λ₁+λ₂) = -1*(0.8) + log(0.8) = -0.8 - 0.223 = -1.023
        expected = -1.0 * (0.5 + 0.3) + math.log(0.5 + 0.3)
        assert ll == pytest.approx(expected, rel=1e-4)

    def test_right_censored_evaluation(self):
        """Evaluate right-censored log-likelihood manually."""
        from symlik.evaluate import evaluate

        contrib = series_exponential_right_censored(rates=["lambda1", "lambda2"])

        env = {"t": 2.0, "lambda1": 0.5, "lambda2": 0.3}
        ll = evaluate(contrib, env)

        # Manual: -t*(λ₁+λ₂) = -2*(0.8) = -1.6
        expected = -2.0 * (0.5 + 0.3)
        assert ll == pytest.approx(expected, rel=1e-4)
