#!/usr/bin/env python3
"""
Survival Analysis Demo: Mixed Censoring Types
==============================================

This example demonstrates ContributionModel with three observation types:

1. **Complete**: Exact failure time observed
   - Contribution: log(λ) - λt  (exponential density)

2. **Right-censored**: Subject survived past observation time
   - Contribution: -λt  (survival function log S(t))

3. **Left-censored**: Subject failed before first observation
   - Contribution: log(1 - exp(-λt))  (CDF log F(t))

We simulate data from an exponential distribution with known λ, apply
different censoring mechanisms, then recover λ via MLE.
"""

import numpy as np
import pandas as pd
from symlik import ContributionModel
from symlik.contributions import (
    complete_exponential,
    right_censored_exponential,
    left_censored_exponential,
)


def simulate_mixed_censoring_data(
    n: int = 200,
    true_lambda: float = 0.5,
    right_censor_time: float = 3.0,
    left_censor_time: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate survival data with mixed censoring.

    - True failure times are exponential(λ)
    - Right-censoring: if T > right_censor_time, observe right_censor_time
    - Left-censoring: if T < left_censor_time, observe left_censor_time
    - Complete: otherwise observe exact T
    """
    np.random.seed(seed)

    # Generate true failure times
    true_times = np.random.exponential(1/true_lambda, n)

    # Apply censoring
    obs_types = []
    obs_times = []

    for t in true_times:
        if t < left_censor_time:
            obs_types.append("left_censored")
            obs_times.append(left_censor_time)
        elif t > right_censor_time:
            obs_types.append("right_censored")
            obs_times.append(right_censor_time)
        else:
            obs_types.append("complete")
            obs_times.append(t)

    return pd.DataFrame({
        "obs_type": obs_types,
        "t": obs_times,
    })


def main():
    print("=" * 60)
    print("Survival Analysis with Mixed Censoring Types")
    print("=" * 60)

    # True parameter
    TRUE_LAMBDA = 0.5
    print(f"\nTrue failure rate: λ = {TRUE_LAMBDA}")
    print(f"  (Mean survival time = 1/λ = {1/TRUE_LAMBDA} units)")

    # Simulate data
    print("\n" + "-" * 60)
    print("Simulating data...")
    df = simulate_mixed_censoring_data(
        n=200,
        true_lambda=TRUE_LAMBDA,
        right_censor_time=3.0,
        left_censor_time=0.5,
        seed=42,
    )

    # Summary statistics
    type_counts = df["obs_type"].value_counts()
    print(f"\nObservation type breakdown:")
    for obs_type, count in type_counts.items():
        pct = 100 * count / len(df)
        print(f"  {obs_type:15s}: {count:3d} ({pct:5.1f}%)")

    print(f"\nSample data (first 10 rows):")
    print(df.head(10).to_string(index=False))

    # Build contribution model
    print("\n" + "-" * 60)
    print("Building ContributionModel with 3 contribution types...")

    model = ContributionModel(
        params=["lambda"],
        type_column="obs_type",
        contributions={
            # Complete: log f(t) = log(λ) - λt
            "complete": complete_exponential(time_var="t", rate="lambda"),

            # Right-censored: log S(t) = -λt
            "right_censored": right_censored_exponential(time_var="t", rate="lambda"),

            # Left-censored: log F(t) = log(1 - exp(-λt))
            "left_censored": left_censored_exponential(time_var="t", rate="lambda"),
        },
    )

    print(f"  Model: {model}")

    # Fit the model
    print("\n" + "-" * 60)
    print("Fitting model via MLE...")

    mle, iterations = model.mle(
        data=df,  # Pass DataFrame directly!
        init={"lambda": 1.0},
        bounds={"lambda": (0.01, 10.0)},
    )

    print(f"  Converged in {iterations} iterations")
    print(f"  MLE: λ̂ = {mle['lambda']:.4f}")
    print(f"  True: λ = {TRUE_LAMBDA}")
    print(f"  Error: {100 * abs(mle['lambda'] - TRUE_LAMBDA) / TRUE_LAMBDA:.2f}%")

    # Standard errors
    print("\n" + "-" * 60)
    print("Computing standard errors...")

    se = model.se(mle, df)

    print(f"  SE(λ̂) = {se['lambda']:.4f}")

    # Confidence interval
    ci_lower = mle["lambda"] - 1.96 * se["lambda"]
    ci_upper = mle["lambda"] + 1.96 * se["lambda"]

    print(f"\n  95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

    covers = ci_lower < TRUE_LAMBDA < ci_upper
    print(f"  Contains true λ = {TRUE_LAMBDA}? {'Yes ✓' if covers else 'No ✗'}")

    # Compare with naive estimate (ignoring censoring)
    print("\n" + "-" * 60)
    print("Comparison: What if we ignored censoring?")

    # Naive: treat all observations as complete
    complete_only = df[df["obs_type"] == "complete"]
    naive_lambda = 1 / df["t"].mean()  # Treats all as complete
    correct_lambda = len(complete_only) / df["t"].sum()  # Proper MLE with censoring

    print(f"\n  Naive estimate (ignoring censoring): λ̂ = {naive_lambda:.4f}")
    print(f"  Proper MLE (accounting for censoring): λ̂ = {mle['lambda']:.4f}")
    print(f"  True value: λ = {TRUE_LAMBDA}")

    naive_error = 100 * abs(naive_lambda - TRUE_LAMBDA) / TRUE_LAMBDA
    proper_error = 100 * abs(mle["lambda"] - TRUE_LAMBDA) / TRUE_LAMBDA

    print(f"\n  Naive error: {naive_error:.1f}%")
    print(f"  Proper MLE error: {proper_error:.1f}%")
    print(f"\n  Improvement: {naive_error - proper_error:.1f} percentage points")

    # Log-likelihood at MLE
    print("\n" + "-" * 60)
    print("Log-likelihood evaluation...")

    eval_data = df.to_dict('list')
    eval_data["lambda"] = mle["lambda"]
    ll_at_mle = model.evaluate(eval_data)

    print(f"  log L(λ̂) = {ll_at_mle:.2f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
