"""Tests for DataFrame support in symlik.

These tests verify that LikelihoodModel and ContributionModel accept
pandas DataFrames in addition to dict-of-lists format.
"""

import math
import pytest
import numpy as np

# Check if pandas is available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Check if polars is available
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

from symlik import LikelihoodModel, ContributionModel
from symlik.distributions import exponential, normal_mean
from symlik.contributions import complete_exponential, right_censored_exponential
from symlik.utils import to_data_dict


class TestToDataDict:
    """Test the to_data_dict utility function."""

    def test_dict_passthrough(self):
        """Dict input should be returned as-is."""
        data = {"x": [1, 2, 3], "y": [4, 5, 6]}
        result = to_data_dict(data)
        assert result == data

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_pandas_dataframe(self):
        """Pandas DataFrame should be converted to dict-of-lists."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = to_data_dict(df)
        assert result == {"x": [1, 2, 3], "y": [4, 5, 6]}

    @pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
    def test_polars_dataframe(self):
        """Polars DataFrame should be converted to dict-of-lists."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = to_data_dict(df)
        assert result == {"x": [1, 2, 3], "y": [4, 5, 6]}

    def test_invalid_type_raises(self):
        """Invalid input type should raise TypeError."""
        with pytest.raises(TypeError):
            to_data_dict([1, 2, 3])

        with pytest.raises(TypeError):
            to_data_dict("not a dataframe")


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestLikelihoodModelPandas:
    """Test LikelihoodModel with pandas DataFrames."""

    def test_mle_with_pandas(self):
        """MLE should work with pandas DataFrame input."""
        model = exponential()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

        fit = model.fit(data=df, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})

        # MLE for exponential: lambda = 1/mean = 1/3
        assert fit.params["lambda"] == pytest.approx(1/3, rel=1e-4)

    def test_se_with_pandas(self):
        """Standard errors should work with pandas DataFrame input."""
        model = exponential()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

        fit = model.fit(data=df, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})

        # SE should be positive and finite
        assert fit.se["lambda"] > 0
        assert np.isfinite(fit.se["lambda"])

    def test_mle_pandas_matches_dict(self):
        """MLE with pandas should match MLE with dict."""
        model = exponential()

        # Dict input
        data_dict = {"x": [1.0, 2.0, 3.0]}
        fit_dict = model.fit(data=data_dict, init={"lambda": 1.0})

        # Pandas input
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        fit_pandas = model.fit(data=df, init={"lambda": 1.0})

        assert fit_dict.params["lambda"] == pytest.approx(fit_pandas.params["lambda"], rel=1e-10)

    def test_normal_mle_with_pandas(self):
        """Normal distribution MLE should work with pandas."""
        model = normal_mean(known_var=1.0)
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

        fit = model.fit(data=df, init={"mu": 0.0})

        # MLE for normal mean is sample mean = 3
        assert fit.params["mu"] == pytest.approx(3.0, rel=1e-4)


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestContributionModelPandas:
    """Test ContributionModel with pandas DataFrames."""

    def test_mle_with_pandas(self):
        """ContributionModel MLE should work with pandas DataFrame."""
        model = ContributionModel(
            params=["lambda"],
            type_column="obs_type",
            contributions={
                "complete": complete_exponential(),
                "censored": right_censored_exponential(),
            },
        )

        df = pd.DataFrame({
            "obs_type": ["complete", "complete", "censored", "censored"],
            "t": [1.0, 2.0, 3.0, 4.0],
        })

        fit = model.fit(data=df, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})

        # With 2 complete and 2 censored: MLE = 2 / (1+2+3+4) = 0.2
        assert fit.params["lambda"] == pytest.approx(0.2, rel=0.05)

    def test_se_with_pandas(self):
        """ContributionModel SE should work with pandas DataFrame."""
        model = ContributionModel(
            params=["lambda"],
            type_column="obs_type",
            contributions={
                "complete": complete_exponential(),
                "censored": right_censored_exponential(),
            },
        )

        df = pd.DataFrame({
            "obs_type": ["complete", "complete", "complete", "censored"],
            "t": [1.0, 2.0, 1.5, 3.0],
        })

        fit = model.fit(data=df, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})

        assert "lambda" in fit.se
        assert fit.se["lambda"] > 0
        assert np.isfinite(fit.se["lambda"])

    def test_mle_pandas_matches_dict(self):
        """ContributionModel MLE with pandas should match dict."""
        model = ContributionModel(
            params=["lambda"],
            type_column="obs_type",
            contributions={"complete": complete_exponential()},
        )

        # Dict input
        data_dict = {
            "obs_type": ["complete", "complete", "complete"],
            "t": [1.0, 2.0, 3.0],
        }
        fit_dict = model.fit(data=data_dict, init={"lambda": 1.0})

        # Pandas input
        df = pd.DataFrame(data_dict)
        fit_pandas = model.fit(data=df, init={"lambda": 1.0})

        assert fit_dict.params["lambda"] == pytest.approx(fit_pandas.params["lambda"], rel=1e-10)


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
class TestLikelihoodModelPolars:
    """Test LikelihoodModel with polars DataFrames."""

    def test_mle_with_polars(self):
        """MLE should work with polars DataFrame input."""
        model = exponential()
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

        fit = model.fit(data=df, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})

        # MLE for exponential: lambda = 1/mean = 1/3
        assert fit.params["lambda"] == pytest.approx(1/3, rel=1e-4)

    def test_mle_polars_matches_dict(self):
        """MLE with polars should match MLE with dict."""
        model = exponential()

        # Dict input
        data_dict = {"x": [1.0, 2.0, 3.0]}
        mle_dict, _ = model.mle(data=data_dict, init={"lambda": 1.0})

        # Polars input
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        mle_polars, _ = model.mle(data=df, init={"lambda": 1.0})

        assert mle_dict["lambda"] == pytest.approx(mle_polars["lambda"], rel=1e-10)


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
class TestContributionModelPolars:
    """Test ContributionModel with polars DataFrames."""

    def test_mle_with_polars(self):
        """ContributionModel MLE should work with polars DataFrame."""
        model = ContributionModel(
            params=["lambda"],
            type_column="obs_type",
            contributions={
                "complete": complete_exponential(),
                "censored": right_censored_exponential(),
            },
        )

        df = pl.DataFrame({
            "obs_type": ["complete", "complete", "censored", "censored"],
            "t": [1.0, 2.0, 3.0, 4.0],
        })

        fit = model.fit(data=df, init={"lambda": 1.0}, bounds={"lambda": (0.01, 10)})

        # With 2 complete and 2 censored: MLE = 2 / (1+2+3+4) = 0.2
        assert fit.params["lambda"] == pytest.approx(0.2, rel=0.05)


class TestMixedWorkflows:
    """Test mixing DataFrames and dicts in workflows."""

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_fit_with_pandas_se_available(self):
        """SE should be available after fitting with pandas."""
        model = exponential()

        # Fit with pandas
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        fit = model.fit(data=df, init={"lambda": 1.0})

        # SE is available via fit.se
        assert fit.se["lambda"] > 0

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_fit_with_dict_se_available(self):
        """SE should be available after fitting with dict."""
        model = exponential()

        # Fit with dict
        data_dict = {"x": [1.0, 2.0, 3.0]}
        fit = model.fit(data=data_dict, init={"lambda": 1.0})

        # SE is available via fit.se
        assert fit.se["lambda"] > 0
