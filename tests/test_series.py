"""Tests for symlik.series module - Series System Likelihood Contributions."""

import math
import pytest
import numpy as np

from symlik import ContributionModel, evaluate
from symlik.series import (
    # Component hazards
    ComponentHazard,
    exponential_component,
    weibull_component,
    custom_component,
    exponential_ph_component,
    weibull_ph_component,
    exponential_log_linear_component,
    # Series system compositions
    series_known_cause,
    series_masked_cause,
    series_right_censored,
    series_left_censored,
    # Exponential series convenience
    exponential_series_known_cause,
    exponential_series_masked_cause,
    exponential_series_right_censored,
    exponential_series_left_censored,
    exponential_series_interval_censored,
    # Weibull series convenience
    weibull_series_known_cause,
    weibull_series_masked_cause,
    weibull_series_right_censored,
    weibull_series_left_censored,
    # Mixed systems
    mixed_series_known_cause,
    mixed_series_masked_cause,
    mixed_series_right_censored,
    mixed_series_left_censored,
    # Factory functions
    build_exponential_series_contributions,
    build_weibull_series_contributions,
)


class TestComponentHazard:
    """Test ComponentHazard building blocks."""

    def test_exponential_component(self):
        comp = exponential_component("lambda")
        assert comp.hazard == "lambda"
        assert comp.cumulative_hazard == ["*", "lambda", "t"]
        assert comp.params == ["lambda"]

    def test_exponential_component_custom_time(self):
        comp = exponential_component("rate", time_var="time")
        assert comp.cumulative_hazard == ["*", "rate", "time"]

    def test_weibull_component(self):
        comp = weibull_component("k", "theta")
        assert "k" in str(comp.hazard)
        assert "theta" in str(comp.hazard)
        assert comp.params == ["k", "theta"]

    def test_weibull_component_custom_time(self):
        comp = weibull_component("shape", "scale", time_var="duration")
        assert "duration" in str(comp.hazard)


class TestExponentialComponentEvaluation:
    """Test exponential component hazard evaluation."""

    def test_hazard_value(self):
        comp = exponential_component("lambda")
        env = {"lambda": 0.5, "t": 2.0}
        h = evaluate(comp.hazard, env)
        assert h == 0.5

    def test_cumulative_hazard_value(self):
        comp = exponential_component("lambda")
        env = {"lambda": 0.5, "t": 2.0}
        H = evaluate(comp.cumulative_hazard, env)
        assert H == pytest.approx(1.0)  # 0.5 * 2


class TestWeibullComponentEvaluation:
    """Test Weibull component hazard evaluation."""

    def test_hazard_value(self):
        comp = weibull_component("k", "theta")
        env = {"k": 2.0, "theta": 1.0, "t": 1.0}
        h = evaluate(comp.hazard, env)
        # h(t) = (k/theta) * (t/theta)^(k-1) = 2/1 * 1^1 = 2
        assert h == pytest.approx(2.0)

    def test_cumulative_hazard_value(self):
        comp = weibull_component("k", "theta")
        env = {"k": 2.0, "theta": 1.0, "t": 1.0}
        H = evaluate(comp.cumulative_hazard, env)
        # H(t) = (t/theta)^k = 1^2 = 1
        assert H == pytest.approx(1.0)

    def test_weibull_reduces_to_exponential(self):
        """Weibull with k=1 should equal exponential."""
        exp_comp = exponential_component("lambda")
        wei_comp = weibull_component("k", "theta")

        env = {"lambda": 0.5, "k": 1.0, "theta": 2.0, "t": 3.0}
        # Exponential: lambda=0.5 means scale=2
        # Weibull k=1, theta=2 should match

        H_exp = evaluate(exp_comp.cumulative_hazard, env)  # 0.5 * 3 = 1.5
        H_wei = evaluate(wei_comp.cumulative_hazard, env)  # (3/2)^1 = 1.5
        assert H_exp == pytest.approx(H_wei)


class TestSeriesKnownCause:
    """Test series_known_cause contribution."""

    def test_two_exponential_components(self):
        components = [
            exponential_component("lambda1"),
            exponential_component("lambda2"),
        ]
        contrib = series_known_cause(components, cause_index=0)

        env = {"lambda1": 0.3, "lambda2": 0.5, "t": 1.0}
        ll = evaluate(contrib, env)

        # log L = log(λ₁) - t*(λ₁+λ₂) = log(0.3) - 1*(0.8) = -1.204 - 0.8 = -2.004
        expected = math.log(0.3) - 1.0 * (0.3 + 0.5)
        assert ll == pytest.approx(expected, rel=1e-4)

    def test_three_components_cause_2(self):
        components = [
            exponential_component("l1"),
            exponential_component("l2"),
            exponential_component("l3"),
        ]
        contrib = series_known_cause(components, cause_index=1)

        env = {"l1": 0.2, "l2": 0.4, "l3": 0.1, "t": 2.0}
        ll = evaluate(contrib, env)

        # log L = log(λ₂) - t*(λ₁+λ₂+λ₃)
        expected = math.log(0.4) - 2.0 * (0.2 + 0.4 + 0.1)
        assert ll == pytest.approx(expected, rel=1e-4)


class TestSeriesMaskedCause:
    """Test series_masked_cause contribution."""

    def test_two_components_both_masked(self):
        components = [
            exponential_component("lambda1"),
            exponential_component("lambda2"),
        ]
        contrib = series_masked_cause(components, candidate_indices=[0, 1])

        env = {"lambda1": 0.3, "lambda2": 0.5, "t": 1.0}
        ll = evaluate(contrib, env)

        # log L = log(λ₁+λ₂) - t*(λ₁+λ₂)
        expected = math.log(0.3 + 0.5) - 1.0 * (0.3 + 0.5)
        assert ll == pytest.approx(expected, rel=1e-4)

    def test_three_components_partial_mask(self):
        components = [
            exponential_component("l1"),
            exponential_component("l2"),
            exponential_component("l3"),
        ]
        # Candidate set {1, 3} = indices [0, 2]
        contrib = series_masked_cause(components, candidate_indices=[0, 2])

        env = {"l1": 0.2, "l2": 0.4, "l3": 0.1, "t": 2.0}
        ll = evaluate(contrib, env)

        # log L = log(λ₁+λ₃) - t*(λ₁+λ₂+λ₃)
        expected = math.log(0.2 + 0.1) - 2.0 * (0.2 + 0.4 + 0.1)
        assert ll == pytest.approx(expected, rel=1e-4)


class TestSeriesRightCensored:
    """Test series_right_censored contribution."""

    def test_two_components(self):
        components = [
            exponential_component("lambda1"),
            exponential_component("lambda2"),
        ]
        contrib = series_right_censored(components)

        env = {"lambda1": 0.3, "lambda2": 0.5, "t": 2.0}
        ll = evaluate(contrib, env)

        # log L = -t*(λ₁+λ₂) = -2*(0.8) = -1.6
        expected = -2.0 * (0.3 + 0.5)
        assert ll == pytest.approx(expected, rel=1e-4)


class TestSeriesLeftCensored:
    """Test series_left_censored contribution."""

    def test_two_components(self):
        components = [
            exponential_component("lambda1"),
            exponential_component("lambda2"),
        ]
        contrib = series_left_censored(components)

        env = {"lambda1": 0.3, "lambda2": 0.5, "t": 2.0}
        ll = evaluate(contrib, env)

        # log L = log(1 - exp(-t*(λ₁+λ₂))) = log(1 - exp(-1.6))
        expected = math.log(1 - math.exp(-2.0 * (0.3 + 0.5)))
        assert ll == pytest.approx(expected, rel=1e-4)


class TestExponentialSeriesConvenience:
    """Test exponential series convenience functions."""

    def test_known_cause_matches_composition(self):
        # Convenience
        contrib1 = exponential_series_known_cause(["l1", "l2"], cause_index=0)

        # Composition
        components = [exponential_component("l1"), exponential_component("l2")]
        contrib2 = series_known_cause(components, cause_index=0)

        env = {"l1": 0.3, "l2": 0.5, "t": 1.5}
        assert evaluate(contrib1, env) == pytest.approx(evaluate(contrib2, env))

    def test_masked_cause_matches_composition(self):
        contrib1 = exponential_series_masked_cause(["l1", "l2", "l3"], [0, 2])
        components = [
            exponential_component("l1"),
            exponential_component("l2"),
            exponential_component("l3"),
        ]
        contrib2 = series_masked_cause(components, [0, 2])

        env = {"l1": 0.2, "l2": 0.4, "l3": 0.1, "t": 2.0}
        assert evaluate(contrib1, env) == pytest.approx(evaluate(contrib2, env))

    def test_right_censored_matches_composition(self):
        contrib1 = exponential_series_right_censored(["l1", "l2"])
        components = [exponential_component("l1"), exponential_component("l2")]
        contrib2 = series_right_censored(components)

        env = {"l1": 0.3, "l2": 0.5, "t": 2.0}
        assert evaluate(contrib1, env) == pytest.approx(evaluate(contrib2, env))

    def test_left_censored_matches_composition(self):
        contrib1 = exponential_series_left_censored(["l1", "l2"])
        components = [exponential_component("l1"), exponential_component("l2")]
        contrib2 = series_left_censored(components)

        env = {"l1": 0.3, "l2": 0.5, "t": 2.0}
        assert evaluate(contrib1, env) == pytest.approx(evaluate(contrib2, env))

    def test_interval_censored(self):
        contrib = exponential_series_interval_censored(["l1", "l2"])

        env = {"l1": 0.3, "l2": 0.5, "t_lower": 1.0, "t_upper": 2.0}
        ll = evaluate(contrib, env)

        # log L = log(exp(-0.8*1) - exp(-0.8*2)) = log(0.449 - 0.202)
        expected = math.log(math.exp(-0.8) - math.exp(-1.6))
        assert ll == pytest.approx(expected, rel=1e-4)


class TestWeibullSeriesConvenience:
    """Test Weibull series convenience functions."""

    def test_known_cause(self):
        contrib = weibull_series_known_cause(["k1", "k2"], ["th1", "th2"], cause_index=0)

        env = {"k1": 2.0, "k2": 1.5, "th1": 1.0, "th2": 2.0, "t": 1.0}
        ll = evaluate(contrib, env)

        # Verify it evaluates to a finite number
        assert np.isfinite(ll)

    def test_masked_cause(self):
        contrib = weibull_series_masked_cause(["k1", "k2"], ["th1", "th2"], [0, 1])

        env = {"k1": 2.0, "k2": 1.5, "th1": 1.0, "th2": 2.0, "t": 1.0}
        ll = evaluate(contrib, env)
        assert np.isfinite(ll)

    def test_right_censored(self):
        contrib = weibull_series_right_censored(["k1", "k2"], ["th1", "th2"])

        env = {"k1": 2.0, "k2": 1.5, "th1": 1.0, "th2": 2.0, "t": 1.0}
        ll = evaluate(contrib, env)

        # H(t) = (t/th1)^k1 + (t/th2)^k2 = 1^2 + 0.5^1.5 = 1 + 0.354 = 1.354
        # log L = -H(t) = -1.354
        expected = -(1.0**2.0 + 0.5**1.5)
        assert ll == pytest.approx(expected, rel=1e-3)

    def test_left_censored(self):
        contrib = weibull_series_left_censored(["k1", "k2"], ["th1", "th2"])

        env = {"k1": 2.0, "k2": 1.5, "th1": 1.0, "th2": 2.0, "t": 1.0}
        ll = evaluate(contrib, env)
        assert np.isfinite(ll)

    def test_shape_scale_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            weibull_series_known_cause(["k1", "k2"], ["th1"], cause_index=0)


class TestMixedSeriesSystems:
    """Test series systems with mixed component types."""

    def test_exponential_and_weibull(self):
        """System with one exponential and one Weibull component."""
        components = [
            exponential_component("lambda"),  # Component 1
            weibull_component("k", "theta"),   # Component 2
        ]

        contrib = mixed_series_known_cause(components, cause_index=0)

        env = {"lambda": 0.5, "k": 2.0, "theta": 1.0, "t": 1.0}
        ll = evaluate(contrib, env)

        # log(λ) - λt - (t/θ)^k = log(0.5) - 0.5 - 1 = -0.693 - 0.5 - 1 = -2.193
        expected = math.log(0.5) - 0.5 - 1.0
        assert ll == pytest.approx(expected, rel=1e-3)

    def test_mixed_masked_cause(self):
        components = [
            exponential_component("l1"),
            weibull_component("k", "theta"),
        ]
        contrib = mixed_series_masked_cause(components, [0, 1])

        env = {"l1": 0.5, "k": 2.0, "theta": 1.0, "t": 1.0}
        ll = evaluate(contrib, env)
        assert np.isfinite(ll)


class TestBuildExponentialSeriesContributions:
    """Test the factory function for exponential series."""

    def test_two_component_system(self):
        contribs = build_exponential_series_contributions(2)

        # Should have: known_1, known_2, masked_12, right_censored
        assert "known_1" in contribs
        assert "known_2" in contribs
        assert "masked_12" in contribs
        assert "right_censored" in contribs
        assert len(contribs) == 4

    def test_three_component_system(self):
        contribs = build_exponential_series_contributions(3)

        # Known: 3, Masked: C(3,2) + C(3,3) = 3 + 1 = 4, Right-censored: 1
        # Total: 8
        assert "known_1" in contribs
        assert "known_2" in contribs
        assert "known_3" in contribs
        assert "masked_12" in contribs
        assert "masked_13" in contribs
        assert "masked_23" in contribs
        assert "masked_123" in contribs
        assert "right_censored" in contribs
        assert len(contribs) == 8

    def test_custom_rate_names(self):
        contribs = build_exponential_series_contributions(
            2, rate_names=["r1", "r2"]
        )
        assert "r1" in str(contribs["known_1"])
        assert "r2" in str(contribs["known_1"])

    def test_include_left_censored(self):
        contribs = build_exponential_series_contributions(2, include_left_censored=True)
        assert "left_censored" in contribs

    def test_include_interval_censored(self):
        contribs = build_exponential_series_contributions(2, include_interval_censored=True)
        assert "interval_censored" in contribs

    def test_rate_names_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            build_exponential_series_contributions(3, rate_names=["l1", "l2"])


class TestBuildWeibullSeriesContributions:
    """Test the factory function for Weibull series."""

    def test_two_component_system(self):
        contribs = build_weibull_series_contributions(2)

        assert "known_1" in contribs
        assert "known_2" in contribs
        assert "masked_12" in contribs
        assert "right_censored" in contribs

    def test_custom_param_names(self):
        contribs = build_weibull_series_contributions(
            2, shape_names=["a1", "a2"], scale_names=["b1", "b2"]
        )
        assert "a1" in str(contribs["known_1"])
        assert "b1" in str(contribs["known_1"])

    def test_include_left_censored(self):
        contribs = build_weibull_series_contributions(2, include_left_censored=True)
        assert "left_censored" in contribs


class TestSeriesMLEIntegration:
    """Integration tests: MLE estimation for series systems."""

    def test_exponential_series_mle_known_cause(self):
        """Test MLE recovery with known-cause exponential series."""
        np.random.seed(42)

        # True parameters
        true_rates = [0.3, 0.5]
        n = 100

        # Generate data
        obs_types = []
        times = []
        for _ in range(n):
            lifetimes = [np.random.exponential(1/r) for r in true_rates]
            sys_time = min(lifetimes)
            cause = lifetimes.index(sys_time)
            obs_types.append(f"known_{cause+1}")
            times.append(sys_time)

        # Build model
        contribs = build_exponential_series_contributions(2)
        model = ContributionModel(
            params=["lambda1", "lambda2"],
            type_column="obs_type",
            contributions=contribs,
        )

        data = {"obs_type": obs_types, "t": times}
        fit = model.fit(
            data=data,
            init={"lambda1": 0.5, "lambda2": 0.5},
            bounds={"lambda1": (0.01, 5), "lambda2": (0.01, 5)},
        )

        # MLEs should be reasonably close to true values
        assert fit.params["lambda1"] == pytest.approx(0.3, rel=0.4)
        assert fit.params["lambda2"] == pytest.approx(0.5, rel=0.4)

    def test_exponential_series_mle_mixed_observations(self):
        """Test MLE with mixed known, masked, and censored observations."""
        np.random.seed(123)

        true_rates = [0.3, 0.5, 0.2]
        censor_time = 4.0
        mask_prob = 0.3
        n = 150

        obs_types = []
        times = []

        for _ in range(n):
            lifetimes = [np.random.exponential(1/r) for r in true_rates]
            sys_time = min(lifetimes)
            cause = lifetimes.index(sys_time)

            if sys_time > censor_time:
                obs_types.append("right_censored")
                times.append(censor_time)
            elif np.random.random() < mask_prob:
                # Mask to candidate set containing true cause
                others = [i for i in range(3) if i != cause]
                partner = np.random.choice(others)
                candidates = sorted([cause, partner])
                key = "masked_" + "".join(str(c+1) for c in candidates)
                obs_types.append(key)
                times.append(sys_time)
            else:
                obs_types.append(f"known_{cause+1}")
                times.append(sys_time)

        # Build model
        contribs = build_exponential_series_contributions(3)
        model = ContributionModel(
            params=["lambda1", "lambda2", "lambda3"],
            type_column="obs_type",
            contributions=contribs,
        )

        data = {"obs_type": obs_types, "t": times}
        fit = model.fit(
            data=data,
            init={"lambda1": 0.5, "lambda2": 0.5, "lambda3": 0.5},
            bounds={f"lambda{i}": (0.01, 5) for i in range(1, 4)},
        )

        # All MLEs should be positive and finite
        for p in ["lambda1", "lambda2", "lambda3"]:
            assert fit.params[p] > 0
            assert np.isfinite(fit.params[p])

    def test_weibull_series_mle(self):
        """Test MLE for Weibull series system."""
        np.random.seed(456)

        # True parameters (shape=2, different scales)
        true_shapes = [2.0, 2.0]
        true_scales = [1.5, 2.0]
        n = 100

        obs_types = []
        times = []

        for _ in range(n):
            lifetimes = [
                true_scales[i] * np.random.weibull(true_shapes[i])
                for i in range(2)
            ]
            sys_time = min(lifetimes)
            cause = lifetimes.index(sys_time)
            obs_types.append(f"known_{cause+1}")
            times.append(sys_time)

        # Build model
        contribs = build_weibull_series_contributions(2)
        model = ContributionModel(
            params=["k1", "k2", "theta1", "theta2"],
            type_column="obs_type",
            contributions=contribs,
        )

        data = {"obs_type": obs_types, "t": times}
        fit = model.fit(
            data=data,
            init={"k1": 1.5, "k2": 1.5, "theta1": 1.0, "theta2": 1.0},
            bounds={
                "k1": (0.5, 5), "k2": (0.5, 5),
                "theta1": (0.1, 10), "theta2": (0.1, 10),
            },
        )

        # Parameters should be in reasonable range
        assert 1.0 < fit.params["k1"] < 4.0
        assert 1.0 < fit.params["k2"] < 4.0
        assert 0.5 < fit.params["theta1"] < 5.0
        assert 0.5 < fit.params["theta2"] < 5.0


class TestCustomComponent:
    """Test custom_component for user-defined hazards."""

    def test_custom_exponential(self):
        """Custom component matching exponential."""
        comp = custom_component(
            hazard="lambda",
            cumulative_hazard=["*", "lambda", "t"],
            params=["lambda"]
        )

        env = {"lambda": 0.5, "t": 2.0}
        h = evaluate(comp.hazard, env)
        H = evaluate(comp.cumulative_hazard, env)

        assert h == 0.5
        assert H == pytest.approx(1.0)

    def test_custom_time_varying_hazard(self):
        """Custom component with time-varying hazard."""
        # h(t) = α * t (linearly increasing hazard)
        # H(t) = α * t² / 2
        comp = custom_component(
            hazard=["*", "alpha", "t"],
            cumulative_hazard=["*", 0.5, ["*", "alpha", ["^", "t", 2]]],
            params=["alpha"]
        )

        env = {"alpha": 2.0, "t": 3.0}
        h = evaluate(comp.hazard, env)
        H = evaluate(comp.cumulative_hazard, env)

        assert h == pytest.approx(6.0)  # 2 * 3
        assert H == pytest.approx(9.0)  # 0.5 * 2 * 9


class TestExponentialPHComponent:
    """Test exponential proportional hazards component."""

    def test_no_covariates_matches_exponential(self):
        """PH with no covariates should equal standard exponential."""
        comp_ph = exponential_ph_component(
            baseline_rate="lambda",
            coefficients=[],
            covariates=[],
        )
        comp_exp = exponential_component("lambda")

        env = {"lambda": 0.5, "t": 2.0}

        h_ph = evaluate(comp_ph.hazard, env)
        h_exp = evaluate(comp_exp.hazard, env)
        assert h_ph == pytest.approx(h_exp)

        H_ph = evaluate(comp_ph.cumulative_hazard, env)
        H_exp = evaluate(comp_exp.cumulative_hazard, env)
        assert H_ph == pytest.approx(H_exp)

    def test_single_covariate(self):
        """PH with single covariate."""
        comp = exponential_ph_component(
            baseline_rate="lambda0",
            coefficients=["beta"],
            covariates=["x"],
        )

        env = {"lambda0": 0.5, "beta": 1.0, "x": 0.5, "t": 2.0}
        h = evaluate(comp.hazard, env)
        H = evaluate(comp.cumulative_hazard, env)

        # h = 0.5 * exp(1.0 * 0.5) = 0.5 * exp(0.5) ≈ 0.824
        expected_h = 0.5 * math.exp(0.5)
        assert h == pytest.approx(expected_h, rel=1e-4)

        # H = 0.5 * 2 * exp(0.5) ≈ 1.649
        expected_H = 0.5 * 2.0 * math.exp(0.5)
        assert H == pytest.approx(expected_H, rel=1e-4)

    def test_multiple_covariates(self):
        """PH with multiple covariates."""
        comp = exponential_ph_component(
            baseline_rate="lambda0",
            coefficients=["beta1", "beta2"],
            covariates=["x1", "x2"],
        )

        env = {
            "lambda0": 0.3,
            "beta1": 0.5, "x1": 2.0,
            "beta2": -0.3, "x2": 1.0,
            "t": 1.5
        }
        h = evaluate(comp.hazard, env)

        # h = 0.3 * exp(0.5*2 + (-0.3)*1) = 0.3 * exp(0.7) ≈ 0.604
        expected_h = 0.3 * math.exp(0.5 * 2.0 + (-0.3) * 1.0)
        assert h == pytest.approx(expected_h, rel=1e-4)

    def test_covariate_coefficient_mismatch_raises(self):
        with pytest.raises(ValueError):
            exponential_ph_component(
                baseline_rate="lambda",
                coefficients=["beta1", "beta2"],
                covariates=["x1"],
            )


class TestWeibullPHComponent:
    """Test Weibull proportional hazards component."""

    def test_no_covariates_matches_weibull(self):
        """PH with no covariates should equal standard Weibull."""
        comp_ph = weibull_ph_component(
            shape="k",
            baseline_scale="theta",
            coefficients=[],
            covariates=[],
        )
        comp_wei = weibull_component("k", "theta")

        env = {"k": 2.0, "theta": 1.5, "t": 1.0}

        h_ph = evaluate(comp_ph.hazard, env)
        h_wei = evaluate(comp_wei.hazard, env)
        assert h_ph == pytest.approx(h_wei, rel=1e-4)

        H_ph = evaluate(comp_ph.cumulative_hazard, env)
        H_wei = evaluate(comp_wei.cumulative_hazard, env)
        assert H_ph == pytest.approx(H_wei, rel=1e-4)

    def test_with_covariate(self):
        """Weibull PH with covariate."""
        comp = weibull_ph_component(
            shape="k",
            baseline_scale="theta",
            coefficients=["beta"],
            covariates=["x"],
        )

        env = {"k": 2.0, "theta": 1.0, "beta": 0.5, "x": 1.0, "t": 1.0}
        H = evaluate(comp.cumulative_hazard, env)

        # H = (t/θ)^k * exp(k * β * x) = 1^2 * exp(2 * 0.5 * 1) = exp(1) ≈ 2.718
        expected_H = 1.0 * math.exp(2.0 * 0.5 * 1.0)
        assert H == pytest.approx(expected_H, rel=1e-4)


class TestExponentialLogLinearComponent:
    """Test exponential log-linear component."""

    def test_basic(self):
        """Log-linear component evaluation."""
        comp = exponential_log_linear_component(
            log_rate_intercept="alpha",
            coefficients=["beta"],
            covariates=["x"],
        )

        env = {"alpha": -1.0, "beta": 0.5, "x": 2.0, "t": 1.5}
        h = evaluate(comp.hazard, env)
        H = evaluate(comp.cumulative_hazard, env)

        # λ = exp(-1 + 0.5*2) = exp(0) = 1
        expected_rate = math.exp(-1.0 + 0.5 * 2.0)
        assert h == pytest.approx(expected_rate, rel=1e-4)
        assert H == pytest.approx(1.5 * expected_rate, rel=1e-4)


class TestCovariateSeriesMLE:
    """Integration tests for covariate-dependent series systems."""

    def test_exponential_ph_series_mle(self):
        """MLE recovery for exponential PH series system."""
        np.random.seed(789)

        # True parameters
        true_lambda0 = [0.3, 0.5]  # Baseline rates
        true_beta = 0.5  # Covariate effect (shared)
        n = 100

        # Generate data with covariate
        obs_types = []
        times = []
        x_vals = []

        for _ in range(n):
            x = np.random.normal(0, 1)  # Covariate
            x_vals.append(x)

            # Hazards depend on covariate
            rates = [lam * math.exp(true_beta * x) for lam in true_lambda0]
            lifetimes = [np.random.exponential(1/r) for r in rates]
            sys_time = min(lifetimes)
            cause = lifetimes.index(sys_time)

            obs_types.append(f"known_{cause+1}")
            times.append(sys_time)

        # Build model with covariate-dependent components
        comp1 = exponential_ph_component("l1", ["beta"], ["x"])
        comp2 = exponential_ph_component("l2", ["beta"], ["x"])

        from symlik.series import series_known_cause, series_right_censored
        contribs = {
            "known_1": series_known_cause([comp1, comp2], cause_index=0),
            "known_2": series_known_cause([comp1, comp2], cause_index=1),
        }

        model = ContributionModel(
            params=["l1", "l2", "beta"],
            type_column="obs_type",
            contributions=contribs,
        )

        data = {"obs_type": obs_types, "t": times, "x": x_vals}
        fit = model.fit(
            data=data,
            init={"l1": 0.5, "l2": 0.5, "beta": 0.0},
            bounds={"l1": (0.01, 5), "l2": (0.01, 5), "beta": (-3, 3)},
        )

        # MLEs should be in reasonable range
        assert 0.1 < fit.params["l1"] < 1.0
        assert 0.2 < fit.params["l2"] < 1.0
        # Beta should be positive (correct direction)
        assert fit.params["beta"] > -1.0
