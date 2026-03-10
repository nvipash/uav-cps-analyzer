#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: Global Sensitivity Analysis Module
Implements Sobol indices and Morris method for variance-based sensitivity analysis.

Based on:
- Saltelli et al., "Sensitivity Analysis in Practice", Wiley, 2004
- Saltelli et al., "Global Sensitivity Analysis. The Primer", Wiley, 2008
- Morris, "Factorial Sampling Plans for Preliminary Computational Experiments", 1991

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

import numpy as np
from scipy.stats import qmc
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable

try:
    from monte_carlo_engine import SimulationParams, MonteCarloEngine
except ImportError:
    import sys
    sys.path.insert(0, '.')
    from monte_carlo_engine import SimulationParams, MonteCarloEngine


@dataclass
class SobolResult:
    """Results of Sobol sensitivity analysis."""
    parameter_names: List[str]
    S1: Dict[str, float]            # First-order indices
    ST: Dict[str, float]            # Total-order indices
    S1_conf: Dict[str, float]       # 95% CI half-width on S1
    ST_conf: Dict[str, float]       # 95% CI half-width on ST
    interaction: Dict[str, float]   # Interaction = ST - S1
    n_samples: int = 0
    total_model_evaluations: int = 0


@dataclass
class MorrisResult:
    """Results of Morris elementary effects screening."""
    parameter_names: List[str]
    mu_star: Dict[str, float]       # Mean of absolute elementary effects
    sigma: Dict[str, float]         # Std of elementary effects
    mu: Dict[str, float]            # Mean of elementary effects (signed)
    n_trajectories: int = 0


# Parameter definitions: (attribute_name, base_value, std/range, distribution_type)
PARAMETER_DEFS = {
    'jammer_power': {
        'attr': 'jammer_power_dbm',
        'std_attr': 'jammer_power_std',
        'base': 40.0,
        'range': (35.0, 45.0),  # dBm
    },
    'jammer_antenna': {
        'attr': 'jammer_antenna_gain_dbi',
        'std_attr': 'jammer_antenna_gain_std',
        'base': 6.0,
        'range': (3.0, 12.0),  # dBi
    },
    'signal_power': {
        'attr': 'signal_power_dbm',
        'std_attr': 'signal_power_std',
        'base': 20.0,
        'range': (17.0, 23.0),  # dBm
    },
    'rx_sensitivity': {
        'attr': 'rx_sensitivity_dbm',
        'std_attr': 'rx_sensitivity_std',
        'base': -90.0,
        'range': (-95.0, -85.0),  # dBm
    },
    'path_loss_std': {
        'attr': 'path_loss_std',
        'std_attr': None,
        'base': 3.0,
        'range': (1.0, 8.0),  # dB
    },
    'fading_margin_std': {
        'attr': 'fading_margin_std',
        'std_attr': None,
        'base': 5.0,
        'range': (2.0, 10.0),  # dB
    },
}


def _evaluate_model(param_values: np.ndarray, base_params: SimulationParams,
                    param_names: List[str], n_mc: int = 500) -> float:
    """
    Evaluate the MC model for a given set of parameter values.

    Args:
        param_values: Array of parameter values (normalized to actual scale)
        base_params: Base simulation parameters
        param_names: List of parameter names corresponding to columns
        n_mc: MC iterations per evaluation

    Returns:
        Mean J/S ratio in dB
    """
    params = SimulationParams(**vars(base_params))
    for i, name in enumerate(param_names):
        pdef = PARAMETER_DEFS[name]
        setattr(params, pdef['attr'], param_values[i])

    engine = MonteCarloEngine(n_processes=1)
    result = engine.run_simulation(params, n_iterations=n_mc, parallel=False)
    return result.mean_js_db


def sobol_analysis(base_params: SimulationParams = None,
                   n_samples: int = 512,
                   n_mc_per_eval: int = 500,
                   param_names: List[str] = None) -> SobolResult:
    """
    Perform Sobol variance-based global sensitivity analysis.

    Uses Saltelli's sampling scheme to compute first-order (S1) and
    total-order (ST) Sobol indices. S1 measures main effects;
    ST - S1 measures interaction contributions.

    The total number of model evaluations is N * (2k + 2) where k = number of parameters.

    Args:
        base_params: Base simulation parameters (uses defaults if None)
        n_samples: Base sample size N (power of 2 recommended for Sobol sequences)
        n_mc_per_eval: MC iterations per model evaluation
        param_names: Parameters to analyze (uses all if None)

    Returns:
        SobolResult with indices and confidence intervals
    """
    if base_params is None:
        base_params = SimulationParams(fhss_enabled=False)

    if param_names is None:
        param_names = list(PARAMETER_DEFS.keys())

    k = len(param_names)

    # Generate Sobol quasi-random samples in [0, 1]^k
    sampler = qmc.Sobol(d=2 * k, scramble=True)
    samples_unit = sampler.random(n_samples)

    # Split into matrices A and B
    A_unit = samples_unit[:, :k]
    B_unit = samples_unit[:, k:]

    # Scale to actual parameter ranges
    lower = np.array([PARAMETER_DEFS[n]['range'][0] for n in param_names])
    upper = np.array([PARAMETER_DEFS[n]['range'][1] for n in param_names])

    A = qmc.scale(A_unit, lower, upper)
    B = qmc.scale(B_unit, lower, upper)

    # Build all evaluation points into a single batch for parallel execution
    total_evals = n_samples * (2 * k + 2)
    print(f"  Sobol analysis: evaluating {total_evals} model runs "
          f"(parallel across {max(1, cpu_count() - 1)} cores)...")

    all_points = []  # (row_array, tag, index)

    for j in range(n_samples):
        all_points.append((A[j], 'A', j))
        all_points.append((B[j], 'B', j))

    for i in range(k):
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        for j in range(n_samples):
            all_points.append((AB_i[j], 'AB', (i, j)))

        BA_i = B.copy()
        BA_i[:, i] = A[:, i]
        for j in range(n_samples):
            all_points.append((BA_i[j], 'BA', (i, j)))

    # Parallel model evaluation
    eval_args = [(pt[0], base_params, param_names, n_mc_per_eval) for pt in all_points]
    n_workers = max(1, cpu_count() - 1)
    with Pool(n_workers) as pool:
        eval_results = pool.starmap(_evaluate_model, eval_args)

    # Unpack results
    f_A = np.zeros(n_samples)
    f_B = np.zeros(n_samples)
    f_AB = np.zeros((k, n_samples))
    f_BA = np.zeros((k, n_samples))

    for (_, tag, idx), val in zip(all_points, eval_results):
        if tag == 'A':
            f_A[idx] = val
        elif tag == 'B':
            f_B[idx] = val
        elif tag == 'AB':
            f_AB[idx[0], idx[1]] = val
        elif tag == 'BA':
            f_BA[idx[0], idx[1]] = val

    # Compute Sobol indices using Saltelli estimators
    f0_sq = np.mean(f_A) * np.mean(f_B)
    var_total = np.var(np.concatenate([f_A, f_B]))

    S1 = {}
    ST = {}
    S1_conf = {}
    ST_conf = {}
    interaction = {}

    for i, name in enumerate(param_names):
        # First-order: S1_i = V_i / V(Y)
        # Saltelli (2010) estimator
        v_i = np.mean(f_B * (f_AB[i] - f_A))
        s1_i = v_i / var_total if var_total > 0 else 0

        # Total-order: ST_i = E[(f_A - f_AB^i)^2] / (2 * V(Y))
        vt_i = 0.5 * np.mean((f_A - f_AB[i]) ** 2)
        st_i = vt_i / var_total if var_total > 0 else 0

        S1[name] = max(0.0, s1_i)
        ST[name] = max(0.0, st_i)
        interaction[name] = max(0.0, st_i - s1_i)

        # Bootstrap confidence intervals on indices
        n_boot = 100
        s1_boot = np.zeros(n_boot)
        st_boot = np.zeros(n_boot)
        for b in range(n_boot):
            idx = np.random.randint(0, n_samples, n_samples)
            var_b = np.var(np.concatenate([f_A[idx], f_B[idx]]))
            if var_b > 0:
                s1_boot[b] = np.mean(f_B[idx] * (f_AB[i, idx] - f_A[idx])) / var_b
                st_boot[b] = 0.5 * np.mean((f_A[idx] - f_AB[i, idx]) ** 2) / var_b

        S1_conf[name] = float(1.96 * np.std(s1_boot))
        ST_conf[name] = float(1.96 * np.std(st_boot))

    total_evals = n_samples * (2 * k + 2)

    return SobolResult(
        parameter_names=param_names,
        S1=S1,
        ST=ST,
        S1_conf=S1_conf,
        ST_conf=ST_conf,
        interaction=interaction,
        n_samples=n_samples,
        total_model_evaluations=total_evals
    )


def morris_screening(base_params: SimulationParams = None,
                     n_trajectories: int = 20,
                     n_levels: int = 4,
                     n_mc_per_eval: int = 500,
                     param_names: List[str] = None) -> MorrisResult:
    """
    Perform Morris method (elementary effects) screening.

    A cheaper alternative to Sobol for identifying important parameters.
    Total evaluations: r * (k + 1) where r = number of trajectories.

    Args:
        base_params: Base simulation parameters
        n_trajectories: Number of Morris trajectories (r)
        n_levels: Number of levels in the grid (p)
        n_mc_per_eval: MC iterations per evaluation
        param_names: Parameters to screen

    Returns:
        MorrisResult with mu_star (importance) and sigma (interaction/non-linearity)
    """
    if base_params is None:
        base_params = SimulationParams(fhss_enabled=False)

    if param_names is None:
        param_names = list(PARAMETER_DEFS.keys())

    k = len(param_names)
    lower = np.array([PARAMETER_DEFS[n]['range'][0] for n in param_names])
    upper = np.array([PARAMETER_DEFS[n]['range'][1] for n in param_names])
    delta = 1.0 / (n_levels - 1)

    print(f"  Morris screening: {n_trajectories * (k + 1)} model evaluations...")

    elementary_effects = {name: [] for name in param_names}

    for t in range(n_trajectories):
        # Generate random starting point on the grid
        x = np.random.choice(np.linspace(0, 1 - delta, n_levels), size=k)

        # Scale to actual values
        x_actual = lower + x * (upper - lower)
        y_base = _evaluate_model(x_actual, base_params, param_names, n_mc_per_eval)

        # Random permutation of parameter indices
        perm = np.random.permutation(k)

        for i in perm:
            x_new = x.copy()
            sign = np.random.choice([-1, 1])
            x_new[i] = np.clip(x_new[i] + sign * delta, 0, 1)

            x_new_actual = lower + x_new * (upper - lower)
            y_new = _evaluate_model(x_new_actual, base_params, param_names, n_mc_per_eval)

            # Elementary effect
            ee = (y_new - y_base) / delta
            elementary_effects[param_names[i]].append(ee)

            x = x_new
            y_base = y_new

    # Compute Morris measures
    mu_star = {}
    sigma = {}
    mu = {}

    for name in param_names:
        ees = np.array(elementary_effects[name])
        mu_star[name] = float(np.mean(np.abs(ees)))
        sigma[name] = float(np.std(ees))
        mu[name] = float(np.mean(ees))

    return MorrisResult(
        parameter_names=param_names,
        mu_star=mu_star,
        sigma=sigma,
        mu=mu,
        n_trajectories=n_trajectories
    )


def print_sobol_results(result: SobolResult):
    """Print formatted Sobol analysis results."""
    print(f"\nSobol Sensitivity Analysis (N={result.n_samples}, "
          f"{result.total_model_evaluations} evaluations)")
    print("=" * 70)
    print(f"{'Parameter':<20} {'S1':>8} {'S1 CI':>8} {'ST':>8} {'ST CI':>8} {'Interaction':>12}")
    print("-" * 70)
    sorted_params = sorted(result.parameter_names, key=lambda p: result.ST[p], reverse=True)
    for name in sorted_params:
        print(f"{name:<20} {result.S1[name]:>8.3f} {result.S1_conf[name]:>8.3f} "
              f"{result.ST[name]:>8.3f} {result.ST_conf[name]:>8.3f} "
              f"{result.interaction[name]:>12.3f}")


def print_morris_results(result: MorrisResult):
    """Print formatted Morris screening results."""
    print(f"\nMorris Elementary Effects Screening ({result.n_trajectories} trajectories)")
    print("=" * 55)
    print(f"{'Parameter':<20} {'mu*':>10} {'sigma':>10} {'mu':>10}")
    print("-" * 55)
    sorted_params = sorted(result.parameter_names,
                           key=lambda p: result.mu_star[p], reverse=True)
    for name in sorted_params:
        print(f"{name:<20} {result.mu_star[name]:>10.2f} "
              f"{result.sigma[name]:>10.2f} {result.mu[name]:>10.2f}")


if __name__ == "__main__":
    print("=" * 60)
    print("UAV-CPS-Analyzer: Sensitivity Analysis Module Test")
    print("=" * 60)

    # Quick Morris screening (fast)
    print("\n1. Morris Screening (fast):")
    morris_result = morris_screening(n_trajectories=5, n_mc_per_eval=200)
    print_morris_results(morris_result)

    # Small Sobol analysis (slow but informative)
    print("\n2. Sobol Analysis (reduced):")
    sobol_result = sobol_analysis(n_samples=32, n_mc_per_eval=200)
    print_sobol_results(sobol_result)

    print("\n" + "=" * 60)
    print("Sensitivity analysis complete!")
