#!/usr/bin/env python3
"""
Series System with Masked Component Cause: Demo
================================================

This example demonstrates ContributionModel for a 3-component series system
with mixed observation types including masked failure cause.

Observation Types:
1. **Known cause**: Exact failure time and component identified
   - Contribution: -t*(λ₁+λ₂+λ₃) + log(λⱼ)

2. **Masked cause**: Failure time known, but only candidate set C identified
   - Contribution: -t*(λ₁+λ₂+λ₃) + log(Σⱼ∈C λⱼ)

3. **Right-censored**: System survived past observation time
   - Contribution: -t*(λ₁+λ₂+λ₃)

The masked cause model assumes C1, C2, C3 conditions:
- C1: True cause is always in the candidate set
- C2: Candidate set probability independent of which component in set failed
- C3: Masking probabilities independent of parameter vector θ

Under these conditions, masking probabilities factor out and don't affect MLE.
"""

import numpy as np
import pandas as pd
from symlik import ContributionModel
from symlik.contributions import (
    series_exponential_known_cause,
    series_exponential_masked_cause,
    series_exponential_right_censored,
)


def simulate_series_system_data(
    n: int = 300,
    true_rates: tuple = (0.3, 0.5, 0.2),  # λ₁, λ₂, λ₃
    censor_time: float = 5.0,
    mask_prob: float = 0.4,  # Probability of masking when failure observed
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate series system data with masked component cause.

    For each system:
    1. Generate component lifetimes T₁, T₂, T₃ ~ Exp(λᵢ)
    2. System fails at T = min(T₁, T₂, T₃)
    3. If T > censor_time: right-censored
    4. Otherwise with prob mask_prob: mask the cause (create candidate set)
    5. Otherwise: known cause (exact component identified)

    Candidate sets are generated to satisfy C1 (true cause always in set):
    - With equal probability: {true, other1}, {true, other2}, or {true, other1, other2}
    """
    np.random.seed(seed)
    rates = np.array(true_rates)
    m = len(rates)

    records = []

    for _ in range(n):
        # Generate component lifetimes
        lifetimes = np.random.exponential(1/rates)
        system_time = np.min(lifetimes)
        true_cause = np.argmin(lifetimes)  # 0, 1, or 2

        if system_time > censor_time:
            # Right-censored
            records.append({
                "obs_type": "right_censored",
                "t": censor_time,
                "candidate_set": None,
            })
        elif np.random.random() < mask_prob:
            # Masked cause - generate candidate set containing true cause (C1)
            other_components = [i for i in range(m) if i != true_cause]
            # Randomly choose: {true, other1}, {true, other2}, or {true, all}
            choice = np.random.choice(3)
            if choice == 0:
                candidate = sorted([true_cause, other_components[0]])
            elif choice == 1:
                candidate = sorted([true_cause, other_components[1]])
            else:
                candidate = list(range(m))  # All components

            # Map candidate set to observation type
            candidate_key = "masked_" + "".join(str(c+1) for c in candidate)
            records.append({
                "obs_type": candidate_key,
                "t": system_time,
                "candidate_set": candidate,
            })
        else:
            # Known cause
            records.append({
                "obs_type": f"known_{true_cause+1}",
                "t": system_time,
                "candidate_set": [true_cause],
            })

    return pd.DataFrame(records)


def main():
    print("=" * 70)
    print("Series System with Masked Component Cause")
    print("=" * 70)

    # True parameters
    TRUE_RATES = (0.3, 0.5, 0.2)
    print(f"\nTrue component failure rates:")
    for i, rate in enumerate(TRUE_RATES):
        print(f"  λ{i+1} = {rate} (mean lifetime = {1/rate:.2f})")

    system_rate = sum(TRUE_RATES)
    print(f"\nSystem failure rate: λ_sys = {system_rate} (mean system lifetime = {1/system_rate:.2f})")

    # Simulate data
    print("\n" + "-" * 70)
    print("Simulating data...")
    df = simulate_series_system_data(
        n=300,
        true_rates=TRUE_RATES,
        censor_time=5.0,
        mask_prob=0.4,
        seed=42,
    )

    # Summary
    type_counts = df["obs_type"].value_counts().sort_index()
    print(f"\nObservation type breakdown:")
    for obs_type, count in type_counts.items():
        pct = 100 * count / len(df)
        print(f"  {obs_type:15s}: {count:3d} ({pct:5.1f}%)")

    print(f"\nSample data (first 10 rows):")
    print(df.head(10).to_string(index=False))

    # Build contribution model
    print("\n" + "-" * 70)
    print("Building ContributionModel...")

    # Rate parameter names
    rates = ["lambda1", "lambda2", "lambda3"]

    # Build contributions dict dynamically based on observation types in data
    contributions = {}

    for obs_type in df["obs_type"].unique():
        if obs_type == "right_censored":
            contributions[obs_type] = series_exponential_right_censored(rates=rates)
        elif obs_type.startswith("known_"):
            cause_idx = int(obs_type.split("_")[1]) - 1  # Convert 1-based to 0-based
            contributions[obs_type] = series_exponential_known_cause(
                rates=rates, cause_index=cause_idx
            )
        elif obs_type.startswith("masked_"):
            # Parse candidate set from type name (e.g., "masked_12" -> [0, 1])
            indices = [int(c) - 1 for c in obs_type.split("_")[1]]
            contributions[obs_type] = series_exponential_masked_cause(
                rates=rates, candidate_indices=indices
            )

    model = ContributionModel(
        params=["lambda1", "lambda2", "lambda3"],
        type_column="obs_type",
        contributions=contributions,
    )

    print(f"  Model: {model}")
    print(f"  Contribution types: {list(contributions.keys())}")

    # Fit the model
    print("\n" + "-" * 70)
    print("Fitting model via MLE...")

    mle, iterations = model.mle(
        data=df,
        init={"lambda1": 0.5, "lambda2": 0.5, "lambda3": 0.5},
        bounds={p: (0.01, 5.0) for p in ["lambda1", "lambda2", "lambda3"]},
    )

    print(f"  Converged in {iterations} iterations")
    print(f"\n  MLE estimates:")
    for i, p in enumerate(["lambda1", "lambda2", "lambda3"]):
        true_val = TRUE_RATES[i]
        error = 100 * abs(mle[p] - true_val) / true_val
        print(f"    {p}: {mle[p]:.4f}  (true: {true_val}, error: {error:.1f}%)")

    # Standard errors
    print("\n" + "-" * 70)
    print("Computing standard errors...")

    se = model.se(mle, df)

    print(f"  Standard errors:")
    for p in ["lambda1", "lambda2", "lambda3"]:
        print(f"    SE({p}) = {se[p]:.4f}")

    # Confidence intervals
    print(f"\n  95% Confidence Intervals:")
    for i, p in enumerate(["lambda1", "lambda2", "lambda3"]):
        ci_lower = mle[p] - 1.96 * se[p]
        ci_upper = mle[p] + 1.96 * se[p]
        true_val = TRUE_RATES[i]
        covers = ci_lower < true_val < ci_upper
        print(f"    {p}: [{ci_lower:.4f}, {ci_upper:.4f}]  "
              f"(contains true {true_val}? {'Yes' if covers else 'No'})")

    # Compare with naive estimate
    print("\n" + "-" * 70)
    print("Comparison: What if we ignored masking?")

    # Naive approach: only use known cause observations
    known_only = df[df["obs_type"].str.startswith("known_")]
    if len(known_only) > 0:
        print(f"\n  Using only known-cause observations ({len(known_only)} of {len(df)}):")

        # Count failures by component
        cause_counts = known_only["obs_type"].value_counts()
        total_time = known_only["t"].sum()

        for i in range(3):
            key = f"known_{i+1}"
            count = cause_counts.get(key, 0)
            naive_rate = count / total_time if total_time > 0 else 0
            true_rate = TRUE_RATES[i]
            error = 100 * abs(naive_rate - true_rate) / true_rate if true_rate > 0 else float('inf')
            print(f"    Naive λ{i+1} = {naive_rate:.4f} (true: {true_rate}, error: {error:.1f}%)")

    print("\n  Using all data with proper masked-cause likelihood:")
    for i, p in enumerate(["lambda1", "lambda2", "lambda3"]):
        error = 100 * abs(mle[p] - TRUE_RATES[i]) / TRUE_RATES[i]
        print(f"    MLE λ{i+1} = {mle[p]:.4f} (true: {TRUE_RATES[i]}, error: {error:.1f}%)")

    # Log-likelihood evaluation
    print("\n" + "-" * 70)
    print("Log-likelihood at MLE...")

    # Prepare data for evaluation
    eval_data = df.to_dict('list')
    eval_data.update(mle)
    ll_at_mle = model.evaluate(eval_data)

    print(f"  log L(MLE) = {ll_at_mle:.2f}")

    # Show the power of the method
    print("\n" + "=" * 70)
    print("Key Insight:")
    print("=" * 70)
    print("""
The masked-cause likelihood properly handles uncertainty about which
component caused failure. Under C1, C2, C3 conditions:

  - C1: True cause always in candidate set (no "false positives")
  - C2: P(candidate set) independent of which component in set failed
  - C3: Masking mechanism independent of failure rates

The likelihood contribution for masked observation with candidate set C:
  log L = -t*(λ₁+λ₂+λ₃) + log(Σⱼ∈C λⱼ)

This "attributes" the hazard across candidate components proportionally,
yielding consistent MLEs even when 40% of observations are masked.
""")

    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
