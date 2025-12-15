"""
Reproduction of results from:
    "Reliability Estimation in Series Systems: Maximum Likelihood Techniques
    for Right-Censored and Masked Failure Data"
    - Alex Towell's Masters Thesis

This script demonstrates using symlik to fit Weibull series system models
with masked component cause and right-censored observations under the
C1, C2, C3 conditions.

System Parameters (from Guo, Niu, and Szidarovszky 2013, extended):
    | Component | Shape (k) | Scale (θ) | Failure Probability |
    |-----------|-----------|-----------|---------------------|
    | 1         | 1.2576    | 994.37    | 0.17                |
    | 2         | 1.1635    | 908.95    | 0.21                |
    | 3         | 1.1308    | 840.11    | 0.23                |
    | 4         | 1.1802    | 940.13    | 0.20                |
    | 5         | 1.2034    | 923.16    | 0.20                |

C1, C2, C3 Conditions:
    C1: True cause is always in the candidate set
    C2: P(candidate set) independent of which component in set failed
    C3: Masking probabilities independent of parameter θ

Bernoulli Masking Model:
    - Failed component placed in candidate set deterministically
    - Each functioning component included with probability p
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import random

# Import symlik components
from symlik import ContributionModel
from symlik.series import build_weibull_series_contributions


# =============================================================================
# True Parameters from Thesis
# =============================================================================

TRUE_PARAMS = {
    "k1": 1.2576, "theta1": 994.37,
    "k2": 1.1635, "theta2": 908.95,
    "k3": 1.1308, "theta3": 840.11,
    "k4": 1.1802, "theta4": 940.13,
    "k5": 1.2034, "theta5": 923.16,
}

NUM_COMPONENTS = 5


# =============================================================================
# Data Generation Functions
# =============================================================================

def weibull_rand(shape: float, scale: float, size: int = 1) -> np.ndarray:
    """Generate Weibull random variates."""
    return scale * np.random.weibull(shape, size)


def weibull_quantile(p: float, shapes: List[float], scales: List[float]) -> float:
    """
    Compute the p-th quantile of a series system with Weibull components.

    For series system: F_sys(t) = 1 - prod(S_i(t)) = 1 - prod(exp(-(t/θ_i)^k_i))
    Solve: F_sys(τ) = p

    Uses numerical root finding.
    """
    from scipy.optimize import brentq

    def system_cdf(t):
        log_survival = sum(-((t / s)**k) for k, s in zip(shapes, scales))
        return 1 - np.exp(log_survival)

    # Find root
    return brentq(lambda t: system_cdf(t) - p, 1e-6, 10000)


def generate_bernoulli_candidate_set(
    true_cause: int,
    m: int,
    p: float,
) -> List[int]:
    """
    Generate candidate set using Bernoulli masking model.

    Satisfies C1, C2, C3 conditions:
    - True cause is always included (C1)
    - Each other component included with probability p (C2, C3)

    Args:
        true_cause: Index of true failing component (0-based)
        m: Number of components
        p: Probability of including each non-failing component

    Returns:
        List of candidate indices (0-based)
    """
    candidates = [true_cause]
    for j in range(m):
        if j != true_cause and random.random() < p:
            candidates.append(j)
    return sorted(candidates)


def generate_series_system_data(
    n: int,
    shapes: List[float],
    scales: List[float],
    tau: float,
    p: float,
    seed: Optional[int] = None,
) -> Dict:
    """
    Generate synthetic data from a Weibull series system.

    Args:
        n: Sample size
        shapes: Shape parameters [k1, ..., km]
        scales: Scale parameters [θ1, ..., θm]
        tau: Right-censoring time
        p: Bernoulli masking probability
        seed: Random seed

    Returns:
        Dictionary with:
            - obs_type: Observation type (e.g., "known_1", "masked_12", "right_censored")
            - t: Observed time (min of failure time and tau)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    m = len(shapes)
    obs_types = []
    times = []

    for i in range(n):
        # Generate component lifetimes
        component_times = [
            weibull_rand(shapes[j], scales[j])[0]
            for j in range(m)
        ]

        # System lifetime is minimum
        T = min(component_times)
        K = component_times.index(T)  # True cause (0-based)

        # Apply right-censoring
        if T > tau:
            # Right-censored
            obs_types.append("right_censored")
            times.append(tau)
        else:
            # Generate candidate set using Bernoulli masking
            candidates = generate_bernoulli_candidate_set(K, m, p)

            if len(candidates) == 1:
                # Known cause
                obs_types.append(f"known_{K+1}")
            else:
                # Masked cause
                key = "masked_" + "".join(str(c+1) for c in candidates)
                obs_types.append(key)

            times.append(T)

    return {
        "obs_type": obs_types,
        "t": times,
    }


def summarize_data(data: Dict) -> Dict:
    """Summarize observation types in dataset."""
    from collections import Counter
    counts = Counter(data["obs_type"])
    return dict(sorted(counts.items()))


# =============================================================================
# Model Building and Fitting
# =============================================================================

def build_model(m: int = 5) -> ContributionModel:
    """
    Build ContributionModel for m-component Weibull series system.
    """
    # Get all contribution types
    contributions = build_weibull_series_contributions(m=m)

    # Define parameters
    params = []
    for i in range(1, m + 1):
        params.extend([f"k{i}", f"theta{i}"])

    return ContributionModel(
        params=params,
        type_column="obs_type",
        contributions=contributions,
    )


def fit_model(
    model: ContributionModel,
    data: Dict,
    init: Optional[Dict[str, float]] = None,
    bounds: Optional[Dict] = None,
) -> Tuple[Dict[str, float], int]:
    """
    Fit the model to data using MLE.

    Returns (mle_estimates, num_iterations)
    """
    if init is None:
        # Use true values as initial guesses
        init = TRUE_PARAMS.copy()

    if bounds is None:
        # Default bounds: shapes in (0.1, 10), scales in (100, 5000)
        bounds = {}
        for i in range(1, NUM_COMPONENTS + 1):
            bounds[f"k{i}"] = (0.1, 10)
            bounds[f"theta{i}"] = (100, 5000)

    return model.mle(data=data, init=init, bounds=bounds, max_iter=200)


# =============================================================================
# Simulation Study
# =============================================================================

def run_single_replicate(
    model: ContributionModel,
    shapes: List[float],
    scales: List[float],
    n: int,
    tau: float,
    p: float,
    seed: Optional[int] = None,
) -> Optional[Dict]:
    """
    Run a single simulation replicate.

    Returns MLE estimates or None if optimization fails.
    """
    # Generate data
    data = generate_series_system_data(n, shapes, scales, tau, p, seed=seed)

    # Fit model
    try:
        mle, iters = fit_model(model, data)

        # Compute standard errors
        se = model.se(mle, data)

        return {
            "mle": mle,
            "se": se,
            "converged": True,
            "iterations": iters,
        }
    except Exception as e:
        return {"converged": False, "error": str(e)}


def run_simulation_study(
    n: int = 90,
    q: float = 0.825,
    p: float = 0.215,
    R: int = 100,
    seed: int = 42,
) -> Dict:
    """
    Run simulation study matching thesis parameters.

    Args:
        n: Sample size
        q: Quantile for right-censoring (P(failure before τ) = q)
        p: Bernoulli masking probability
        R: Number of replicates
        seed: Random seed for reproducibility

    Returns:
        Dictionary with simulation results
    """
    np.random.seed(seed)

    # Extract true parameters as lists
    shapes = [TRUE_PARAMS[f"k{i}"] for i in range(1, NUM_COMPONENTS + 1)]
    scales = [TRUE_PARAMS[f"theta{i}"] for i in range(1, NUM_COMPONENTS + 1)]

    # Compute right-censoring time
    tau = weibull_quantile(q, shapes, scales)
    print(f"Right-censoring time τ = {tau:.2f}")
    print(f"Expected {(1-q)*100:.1f}% right-censored observations")
    print()

    # Build model
    model = build_model()

    results = []
    converged_count = 0

    print(f"Running {R} simulation replicates...")
    for r in range(R):
        result = run_single_replicate(
            model, shapes, scales, n, tau, p, seed=seed + r
        )
        results.append(result)

        if result.get("converged"):
            converged_count += 1

        if (r + 1) % 10 == 0:
            print(f"  Completed {r+1}/{R} replicates ({converged_count} converged)")

    print(f"\nConvergence rate: {converged_count/R*100:.1f}%")

    # Compute summary statistics
    summary = compute_summary_stats(results, TRUE_PARAMS)

    return {
        "results": results,
        "summary": summary,
        "convergence_rate": converged_count / R,
        "settings": {"n": n, "q": q, "p": p, "tau": tau, "R": R},
    }


def compute_summary_stats(results: List[Dict], true_params: Dict) -> Dict:
    """
    Compute summary statistics from simulation results.
    """
    converged = [r for r in results if r.get("converged")]

    if not converged:
        return {"error": "No converged replicates"}

    summary = {}

    for param in true_params:
        mles = [r["mle"][param] for r in converged]
        true_val = true_params[param]

        mles_arr = np.array(mles)

        summary[param] = {
            "true": true_val,
            "mean_mle": np.mean(mles_arr),
            "median_mle": np.median(mles_arr),
            "std_mle": np.std(mles_arr),
            "bias": np.mean(mles_arr) - true_val,
            "rel_bias": (np.mean(mles_arr) - true_val) / true_val * 100,
            "q025": np.percentile(mles_arr, 2.5),
            "q975": np.percentile(mles_arr, 97.5),
        }

    return summary


def print_summary(sim_results: Dict):
    """Print formatted summary of simulation results."""
    summary = sim_results["summary"]
    settings = sim_results["settings"]

    print("\n" + "="*70)
    print("SIMULATION STUDY RESULTS")
    print("="*70)
    print(f"Sample size: {settings['n']}")
    print(f"Right-censoring quantile: {settings['q']}")
    print(f"Masking probability: {settings['p']}")
    print(f"Replicates: {settings['R']}")
    print(f"Convergence rate: {sim_results['convergence_rate']*100:.1f}%")
    print()

    print("-"*70)
    print(f"{'Parameter':<12} {'True':>10} {'Mean MLE':>12} {'Bias%':>10} {'95% CI Width':>14}")
    print("-"*70)

    for param in sorted(summary.keys()):
        stats = summary[param]
        ci_width = stats["q975"] - stats["q025"]
        print(f"{param:<12} {stats['true']:>10.4f} {stats['mean_mle']:>12.4f} "
              f"{stats['rel_bias']:>9.1f}% {ci_width:>14.2f}")


# =============================================================================
# Quick Demo
# =============================================================================

def demo():
    """
    Quick demonstration fitting a single dataset.
    """
    print("="*70)
    print("SYMLIK THESIS REPRODUCTION DEMO")
    print("5-Component Weibull Series System")
    print("="*70)
    print()

    # Parameters
    shapes = [TRUE_PARAMS[f"k{i}"] for i in range(1, NUM_COMPONENTS + 1)]
    scales = [TRUE_PARAMS[f"theta{i}"] for i in range(1, NUM_COMPONENTS + 1)]

    n = 90
    q = 0.825
    p = 0.215

    # Compute censoring time
    tau = weibull_quantile(q, shapes, scales)
    print(f"True parameters (from thesis):")
    for i in range(NUM_COMPONENTS):
        print(f"  Component {i+1}: k={shapes[i]:.4f}, θ={scales[i]:.2f}")
    print()
    print(f"Simulation settings:")
    print(f"  Sample size n = {n}")
    print(f"  Right-censoring quantile q = {q} (τ = {tau:.2f})")
    print(f"  Masking probability p = {p}")
    print()

    # Generate data
    print("Generating synthetic data...")
    data = generate_series_system_data(n, shapes, scales, tau, p, seed=42)

    print(f"\nObservation type distribution:")
    for obs_type, count in summarize_data(data).items():
        print(f"  {obs_type}: {count}")

    # Build model
    print("\nBuilding ContributionModel for Weibull series system...")
    model = build_model()
    print(f"  Parameters: {model.params}")
    print(f"  Contribution types: {len(model.contributions)}")

    # Fit model
    print("\nFitting model via MLE...")
    try:
        mle, iters = fit_model(model, data)
        print(f"  Converged in {iters} iterations")

        # Compute standard errors
        se = model.se(mle, data)

        print("\nResults:")
        print("-"*60)
        print(f"{'Component':<12} {'Param':>8} {'True':>10} {'MLE':>12} {'SE':>10}")
        print("-"*60)

        for i in range(1, NUM_COMPONENTS + 1):
            k_param = f"k{i}"
            t_param = f"theta{i}"
            print(f"Component {i:<3} {k_param:>8} {TRUE_PARAMS[k_param]:>10.4f} "
                  f"{mle[k_param]:>12.4f} {se[k_param]:>10.4f}")
            print(f"{'':12} {t_param:>8} {TRUE_PARAMS[t_param]:>10.2f} "
                  f"{mle[t_param]:>12.2f} {se[t_param]:>10.2f}")

    except Exception as e:
        print(f"  Optimization failed: {e}")

    print()
    print("="*70)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "sim":
        # Run full simulation study
        R = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        results = run_simulation_study(R=R)
        print_summary(results)
    else:
        # Quick demo
        demo()
