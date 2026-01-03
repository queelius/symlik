"""Tests for FittedLikelihoodModel inference operations."""

import math
import pytest
import numpy as np
from symlik import LikelihoodModel, FittedLikelihoodModel
from symlik.distributions import exponential, normal, poisson


class TestFittedModelBasics:
    """Test basic FittedLikelihoodModel properties."""

    def test_params_property(self):
        """params returns MLE estimates as dict."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        assert isinstance(fit.params, dict)
        assert 'lambda' in fit.params
        # MLE for exponential is 1/mean = 1/3
        assert fit.params['lambda'] == pytest.approx(1/3, rel=0.01)

    def test_param_names_property(self):
        """param_names returns list of parameter names."""
        model = normal()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'mu': 0, 'sigma2': 1},
                       bounds={'sigma2': (0.1, None)})

        assert fit.param_names == ['mu', 'sigma2']

    def test_nobs_property(self):
        """nobs returns sample size."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        assert fit.nobs == 5

    def test_df_model_property(self):
        """df_model returns number of parameters."""
        model = normal()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'mu': 0, 'sigma2': 1},
                       bounds={'sigma2': (0.1, None)})

        assert fit.df_model == 2

    def test_converged_property(self):
        """converged indicates optimization success."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        assert fit.converged is True

    def test_n_iter_property(self):
        """n_iter returns iteration count."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        assert fit.n_iter >= 1


class TestLogLikelihood:
    """Test log-likelihood property."""

    def test_llf_finite(self):
        """llf returns finite log-likelihood."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        assert np.isfinite(fit.llf)

    def test_llf_cached(self):
        """llf is cached after first access."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        llf1 = fit.llf
        llf2 = fit.llf
        assert llf1 == llf2


class TestAIC:
    """Test AIC property."""

    def test_aic_formula(self):
        """AIC = -2*llf + 2*k."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        expected = -2 * fit.llf + 2 * fit.df_model
        assert fit.aic == pytest.approx(expected, rel=1e-10)

    def test_aic_normal(self):
        """AIC for normal model (2 parameters)."""
        model = normal()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'mu': 0, 'sigma2': 1},
                       bounds={'sigma2': (0.1, None)})

        expected = -2 * fit.llf + 2 * 2
        assert fit.aic == pytest.approx(expected, rel=1e-10)


class TestBIC:
    """Test BIC property."""

    def test_bic_formula(self):
        """BIC = -2*llf + k*log(n)."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        expected = -2 * fit.llf + fit.df_model * np.log(fit.nobs)
        assert fit.bic == pytest.approx(expected, rel=1e-10)

    def test_bic_vs_aic(self):
        """BIC penalty > AIC for n > e^2 ~ 7.4."""
        model = normal()
        data = {'x': list(range(1, 21))}  # n=20
        fit = model.fit(data, init={'mu': 0, 'sigma2': 10},
                       bounds={'sigma2': (0.1, None)})

        # For n=20, log(20) > 2, so BIC > AIC
        assert fit.bic > fit.aic


class TestStandardErrors:
    """Test standard error properties."""

    def test_se_property(self):
        """se returns dict of standard errors."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        assert isinstance(fit.se, dict)
        assert 'lambda' in fit.se
        assert fit.se['lambda'] > 0

    def test_bse_alias(self):
        """bse is alias for se (statsmodels compatibility)."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        assert fit.bse == fit.se


class TestCovParams:
    """Test covariance matrix method."""

    def test_cov_params_shape(self):
        """cov_params returns k x k matrix."""
        model = normal()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'mu': 0, 'sigma2': 1},
                       bounds={'sigma2': (0.1, None)})

        cov = fit.cov_params()
        assert cov.shape == (2, 2)

    def test_cov_params_symmetric(self):
        """Covariance matrix is symmetric."""
        model = normal()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'mu': 0, 'sigma2': 1},
                       bounds={'sigma2': (0.1, None)})

        cov = fit.cov_params()
        assert np.allclose(cov, cov.T)

    def test_cov_params_diagonal_matches_se(self):
        """sqrt(diag(cov)) equals se."""
        model = normal()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'mu': 0, 'sigma2': 1},
                       bounds={'sigma2': (0.1, None)})

        cov = fit.cov_params()
        se_from_cov = np.sqrt(np.diag(cov))
        se = fit.se

        for i, param in enumerate(fit.param_names):
            assert se_from_cov[i] == pytest.approx(se[param], rel=1e-6)


class TestConfInt:
    """Test confidence interval method."""

    def test_conf_int_contains_mle(self):
        """CI contains the MLE."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        ci = fit.conf_int()
        lo, hi = ci['lambda']
        assert lo < fit.params['lambda'] < hi

    def test_conf_int_width_increases_with_level(self):
        """Higher confidence = wider interval."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        ci_90 = fit.conf_int(alpha=0.10)
        ci_95 = fit.conf_int(alpha=0.05)
        ci_99 = fit.conf_int(alpha=0.01)

        width_90 = ci_90['lambda'][1] - ci_90['lambda'][0]
        width_95 = ci_95['lambda'][1] - ci_95['lambda'][0]
        width_99 = ci_99['lambda'][1] - ci_99['lambda'][0]

        assert width_90 < width_95 < width_99

    def test_conf_int_all_params(self):
        """conf_int returns CIs for all parameters."""
        model = normal()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'mu': 0, 'sigma2': 1},
                       bounds={'sigma2': (0.1, None)})

        ci = fit.conf_int()
        assert set(ci.keys()) == {'mu', 'sigma2'}


class TestWaldTest:
    """Test Wald hypothesis test."""

    def test_wald_at_mle(self):
        """Wald statistic = 0 when null equals MLE."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        result = fit.wald_test(fit.params)

        assert result['statistic'] == pytest.approx(0.0, abs=1e-6)
        assert result['pvalue'] == pytest.approx(1.0, abs=1e-3)

    def test_wald_structure(self):
        """Wald test returns correct structure."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        result = fit.wald_test({'lambda': 1.0})

        assert 'statistic' in result
        assert 'df' in result
        assert 'pvalue' in result
        assert result['df'] == 1

    def test_wald_far_from_mle(self):
        """Wald statistic large when null far from MLE."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        result = fit.wald_test({'lambda': 5.0})

        assert result['statistic'] > 10
        assert result['pvalue'] < 0.01


class TestScoreTest:
    """Test Score (Lagrange multiplier) test."""

    def test_score_at_mle(self):
        """Score statistic ~ 0 when null equals MLE."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        result = fit.score_test(fit.params)

        assert result['statistic'] == pytest.approx(0.0, abs=1e-4)
        assert result['pvalue'] > 0.9

    def test_score_structure(self):
        """Score test returns correct structure."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        result = fit.score_test({'lambda': 0.5})

        assert 'statistic' in result
        assert 'df' in result
        assert 'pvalue' in result
        assert result['df'] == 1

    def test_score_far_from_mle(self):
        """Score statistic large when null far from MLE."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        result = fit.score_test({'lambda': 2.0})

        assert result['statistic'] > 5
        assert result['pvalue'] < 0.05


class TestLRTest:
    """Test Likelihood Ratio test."""

    def test_lr_at_mle(self):
        """LR statistic = 0 when null equals MLE."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        result = fit.lr_test(fit.params)

        assert result['statistic'] == pytest.approx(0.0, abs=1e-10)

    def test_lr_structure(self):
        """LR test returns correct structure."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        result = fit.lr_test({'lambda': 1.0})

        assert 'statistic' in result
        assert 'df' in result
        assert 'pvalue' in result

    def test_lr_nonnegative(self):
        """LR statistic is always >= 0."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        for null_lambda in [0.1, 0.5, 1.0, 2.0, 5.0]:
            result = fit.lr_test({'lambda': null_lambda})
            assert result['statistic'] >= -1e-10

    def test_lr_far_from_mle(self):
        """LR statistic large when null far from MLE."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        result = fit.lr_test({'lambda': 5.0})

        assert result['statistic'] > 10
        assert result['pvalue'] < 0.01


class TestSummary:
    """Test summary output."""

    def test_summary_returns_string(self):
        """summary() returns formatted string."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        summary = fit.summary()

        assert isinstance(summary, str)
        assert 'lambda' in summary
        assert 'Log-Likelihood' in summary

    def test_summary_contains_key_info(self):
        """Summary includes all key information."""
        model = normal()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'mu': 0, 'sigma2': 1},
                       bounds={'sigma2': (0.1, None)})

        summary = fit.summary()

        assert 'mu' in summary
        assert 'sigma2' in summary
        assert 'AIC' in summary
        assert 'BIC' in summary
        assert 'Std.Err' in summary

    def test_str_equals_summary(self):
        """str(fit) returns summary."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        assert str(fit) == fit.summary()


class TestRepr:
    """Test repr output."""

    def test_repr_format(self):
        """repr shows params and llf."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        r = repr(fit)

        assert 'FittedLikelihoodModel' in r
        assert 'lambda' in r
        assert 'llf' in r


class TestAsymptoticTheory:
    """Test asymptotic properties of inference."""

    def test_coverage_probability(self):
        """95% CI covers true value ~95% of time (Monte Carlo)."""
        np.random.seed(42)
        true_lambda = 2.0
        n = 50
        n_sims = 100

        model = exponential()
        coverage_count = 0

        for _ in range(n_sims):
            data = {'x': np.random.exponential(1/true_lambda, n).tolist()}
            fit = model.fit(data, init={'lambda': 1.0},
                           bounds={'lambda': (0.01, None)})
            ci = fit.conf_int(alpha=0.05)

            if ci['lambda'][0] <= true_lambda <= ci['lambda'][1]:
                coverage_count += 1

        coverage = coverage_count / n_sims
        assert 0.85 < coverage < 1.0

    def test_trinity_equivalence(self):
        """Wald, Score, LR tests asymptotically equivalent."""
        np.random.seed(42)
        n = 200

        model = exponential()
        data = {'x': np.random.exponential(1.0, n).tolist()}
        fit = model.fit(data, init={'lambda': 1.0})

        null_val = {'lambda': 0.8}

        wald = fit.wald_test(null_val)
        score = fit.score_test(null_val)
        lr = fit.lr_test(null_val)

        # All should reject at similar levels
        pvals = [wald['pvalue'], score['pvalue'], lr['pvalue']]
        assert all(p < 0.01 for p in pvals), f"All should reject: {pvals}"


class TestEdgeCases:
    """Test edge cases."""

    def test_small_sample(self):
        """Works with small samples."""
        model = exponential()
        fit = model.fit({'x': [1.0, 2.0]}, init={'lambda': 1.0})

        assert np.isfinite(fit.aic)
        assert np.isfinite(fit.bic)
        assert np.isfinite(fit.wald_test({'lambda': 1.0})['statistic'])

    def test_poisson_inference(self):
        """Inference works for Poisson model."""
        model = poisson()
        fit = model.fit({'x': [1, 2, 3, 2, 1, 0, 2, 3, 1, 2]},
                       init={'lambda': 1.0},
                       bounds={'lambda': (0.01, None)})

        assert np.isfinite(fit.aic)
        assert np.isfinite(fit.bic)
        ci = fit.conf_int()
        assert ci['lambda'][0] < ci['lambda'][1]


class TestInformationMatrix:
    """Test information_matrix method."""

    def test_information_matrix_shape(self):
        """information_matrix returns k x k matrix."""
        model = normal()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'mu': 0, 'sigma2': 1},
                       bounds={'sigma2': (0.1, None)})

        info = fit.information_matrix()
        assert info.shape == (2, 2)

    def test_information_matrix_positive_definite(self):
        """Information matrix should be positive definite at MLE."""
        model = exponential()
        fit = model.fit({'x': [1, 2, 3, 4, 5]}, init={'lambda': 1.0})

        info = fit.information_matrix()
        eigenvalues = np.linalg.eigvalsh(info)
        assert all(ev > 0 for ev in eigenvalues)
