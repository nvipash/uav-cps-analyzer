#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: Monte Carlo Simulation Engine
Implements parallel Monte Carlo simulation for statistical analysis of UAV communication reliability.

Based on:
- N. Metropolis and S. Ulam, "The Monte Carlo Method", J. Amer. Stat. Assoc., 1949
- A. Saltelli et al., "Sensitivity Analysis in Practice", Wiley, 2004

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

import numpy as np
from scipy import stats as scipy_stats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional
from multiprocessing import Pool, cpu_count
import time
import warnings
from enum import Enum

# Import local modules
try:
    from propagation_models import (
        AltitudeDependentModel, AlHouraniA2GModel, calculate_js_ratio,
        RiceFadingModel, BERModel, ModulationType,
        DopplerModel, CosinePattern, OmnidirectionalPattern,
        AtmosphericAbsorption, UrbanMultiPathCorrection
    )
    from fhss_emulator import (
        OcuSyncProtocol, JammingStrategy, JammingEffectivenessAnalyzer
    )
except ImportError:
    # For standalone testing
    import sys
    sys.path.insert(0, '.')
    from propagation_models import (
        AltitudeDependentModel, AlHouraniA2GModel, calculate_js_ratio,
        RiceFadingModel, BERModel, ModulationType,
        DopplerModel, CosinePattern, OmnidirectionalPattern,
        AtmosphericAbsorption, UrbanMultiPathCorrection
    )
    from fhss_emulator import (
        OcuSyncProtocol, JammingStrategy, JammingEffectivenessAnalyzer
    )


@dataclass
class SimulationParams:
    """Parameters for Monte Carlo simulation."""
    # Jammer parameters
    jammer_power_dbm: float = 40.0       # 10W
    jammer_power_std: float = 1.0        # +/-1 dB uncertainty
    jammer_distance_m: float = 500.0
    jammer_antenna_gain_dbi: float = 6.0
    jammer_antenna_gain_std: float = 1.5  # +/-1.5 dB uncertainty

    # Signal (UAV) parameters
    signal_power_dbm: float = 20.0       # 100mW
    signal_power_std: float = 1.0        # +/-1 dB uncertainty
    signal_distance_m: float = 5000.0
    signal_antenna_gain_dbi: float = 2.0

    # Target parameters
    altitude_m: float = 100.0
    frequency_mhz: float = 2437.0
    target_velocity_ms: float = 0.0      # UAV velocity (for Doppler)

    # Receiver parameters
    rx_sensitivity_dbm: float = -90.0
    rx_sensitivity_std: float = 2.0      # +/-2 dB uncertainty

    # Propagation uncertainty
    path_loss_std: float = 3.0           # Shadow fading std
    fading_margin_std: float = 5.0       # Rice fading margin uncertainty

    # FHSS parameters
    fhss_enabled: bool = True
    jamming_strategy: str = "broadband"

    # Jamming threshold (legacy, used as fallback only)
    js_threshold_db: float = 10.0

    # Model selection
    propagation_model: str = "altitude"  # "altitude" (legacy), "al_hourani"
    environment: str = "urban"           # For Al-Hourani: dense_urban, urban, suburban, rural
    jammer_beam_width_deg: float = 0.0   # 0 = omnidirectional (no pattern applied)

    # Long-range corrections (for distances > 1km)
    enable_atmospheric: bool = True       # ITU-R P.676 atmospheric absorption
    enable_multipath: bool = True          # Urban NLOS multi-path correction
    temperature_c: float = 15.0
    humidity_percent: float = 50.0

    # Heavy-tailed shadow fading
    shadow_distribution: str = "normal"   # "normal" | "student_t" | "mixture"
    shadow_df: float = 5.0                 # Degrees of freedom for Student's t

    # Spatial correlation of shadow fading (Gudmundson 1991 model)
    # rho = exp(-d_separation / decorrelation_distance)
    # Typical decorrelation distance: 50-200m urban, 100-500m suburban
    shadow_correlation: float = 0.0        # 0=independent, 1=fully correlated
    decorrelation_distance_m: float = 100.0  # Used if shadow_correlation < 0 (auto-compute)


@dataclass
class SimulationResult:
    """Result of a single simulation iteration."""
    js_ratio_db: float
    jamming_success: bool
    received_signal_dbm: float
    received_jammer_dbm: float
    path_loss_signal_db: float
    path_loss_jammer_db: float
    fhss_effectiveness: float = 1.0


@dataclass
class MCResult:
    """Aggregated Monte Carlo simulation results."""
    n_iterations: int
    mean_js_db: float
    std_js_db: float
    ci_95_lower: float
    ci_95_upper: float
    success_probability: float

    # Detailed statistics
    js_values: np.ndarray = field(default_factory=lambda: np.array([]))
    percentiles: Dict[int, float] = field(default_factory=dict)

    # Bootstrap CI uncertainty (CI of the CI bounds)
    ci_lower_uncertainty: Tuple[float, float] = (0.0, 0.0)  # (lower, upper) of CI lower bound
    ci_upper_uncertainty: Tuple[float, float] = (0.0, 0.0)  # (lower, upper) of CI upper bound

    # Convergence info
    converged: bool = False
    convergence_iteration: int = 0
    required_sample_size: int = 0

    # Sensitivity analysis
    sensitivity: Dict[str, float] = field(default_factory=dict)

    # Timing
    execution_time_s: float = 0.0


def _simulate_with_normals(params: SimulationParams, normals: np.ndarray) -> SimulationResult:
    """
    Single MC iteration using pre-generated standard normal samples.
    Used for QMC and antithetic variates where we control the noise inputs.

    normals shape: (8,) — standard normal samples for:
        [0] jammer_power, [1] jammer_antenna, [2] signal_power, [3] rx_sens,
        [4] shadow_signal, [5] shadow_jammer, [6] fading_margin, [7] azimuth_offset
    """
    # Use provided normals instead of np.random.randn
    jammer_power = params.jammer_power_dbm + normals[0] * params.jammer_power_std
    jammer_antenna_gain = params.jammer_antenna_gain_dbi + normals[1] * params.jammer_antenna_gain_std
    signal_power = params.signal_power_dbm + normals[2] * params.signal_power_std

    shadow_fading_signal = normals[4] * params.path_loss_std
    shadow_fading_jammer = normals[5] * params.path_loss_std
    fading_margin = abs(normals[6] * params.fading_margin_std)

    # Select propagation model
    if params.propagation_model == "al_hourani":
        model = AlHouraniA2GModel(environment=params.environment)
    else:
        model = AltitudeDependentModel()

    # Atmospheric and multi-path corrections
    atm_loss_signal = atm_loss_jammer = 0.0
    if params.enable_atmospheric:
        atm_loss_signal = AtmosphericAbsorption.total_loss_db(
            params.signal_distance_m, params.frequency_mhz,
            params.temperature_c, params.humidity_percent)
        atm_loss_jammer = AtmosphericAbsorption.total_loss_db(
            params.jammer_distance_m, params.frequency_mhz,
            params.temperature_c, params.humidity_percent)

    mp_loss_signal = mp_loss_jammer = 0.0
    if params.enable_multipath:
        mp_loss_signal = UrbanMultiPathCorrection.multi_path_loss_db(
            params.signal_distance_m, params.environment, params.altitude_m)
        mp_loss_jammer = UrbanMultiPathCorrection.multi_path_loss_db(
            params.jammer_distance_m, params.environment, params.altitude_m)

    pl_signal = (model.path_loss(params.signal_distance_m, params.altitude_m, params.frequency_mhz)
                  + shadow_fading_signal + fading_margin + atm_loss_signal + mp_loss_signal)
    pl_jammer = (model.path_loss(params.jammer_distance_m, params.altitude_m, params.frequency_mhz)
                  + shadow_fading_jammer + atm_loss_jammer + mp_loss_jammer)

    # Antenna pattern
    effective_jammer_gain = jammer_antenna_gain
    if params.jammer_beam_width_deg > 0:
        pattern = CosinePattern(beam_width_deg=params.jammer_beam_width_deg,
                                 max_gain_dbi=jammer_antenna_gain)
        if params.jammer_distance_m > 0:
            elev_deg = np.degrees(np.arctan(params.altitude_m / params.jammer_distance_m))
        else:
            elev_deg = 90.0
        azimuth_offset = normals[7] * 5.0
        off_boresight = np.sqrt(elev_deg**2 + azimuth_offset**2)
        effective_jammer_gain = pattern.gain_dbi(off_boresight)

    rx_signal = signal_power + params.signal_antenna_gain_dbi - pl_signal
    rx_jammer = jammer_power + effective_jammer_gain - pl_jammer
    js_ratio = rx_jammer - rx_signal

    fhss_effectiveness = 1.0
    if params.fhss_enabled:
        protocol = OcuSyncProtocol()
        analyzer = JammingEffectivenessAnalyzer(protocol)
        strategy = JammingStrategy(params.jamming_strategy)
        fhss_effectiveness = analyzer.calculate_effectiveness(strategy, 83.5)

    effective_js = js_ratio if not params.fhss_enabled else js_ratio * fhss_effectiveness
    jam_prob = BERModel.jamming_success_probability(
        js_ratio_db=effective_js, signal_power_db=rx_signal,
        modulation=ModulationType.QPSK
    )
    # Use deterministic threshold for QMC reproducibility (use first normal as uniform proxy)
    from scipy.stats import norm as _norm
    u = _norm.cdf(normals[0])  # convert to uniform
    jamming_success = u < jam_prob

    return SimulationResult(
        js_ratio_db=js_ratio,
        jamming_success=jamming_success,
        received_signal_dbm=rx_signal,
        received_jammer_dbm=rx_jammer,
        path_loss_signal_db=pl_signal,
        path_loss_jammer_db=pl_jammer,
        fhss_effectiveness=fhss_effectiveness
    )


def _simulate_single_iteration(args: Tuple[SimulationParams, int]) -> SimulationResult:
    """
    Execute single Monte Carlo iteration.

    Integrates:
    - Selectable propagation model (altitude-dependent or Al-Hourani A2G)
    - Directional antenna patterns (if beam_width specified)
    - Doppler effects on FHSS synchronization (if target velocity > 0)
    - BER/PER soft-degradation jamming model

    Args:
        args: Tuple of (SimulationParams, random_seed)

    Returns:
        SimulationResult for this iteration
    """
    params, seed = args
    np.random.seed(seed)

    # --- Sample uncertain parameters ---
    jammer_power = params.jammer_power_dbm + np.random.randn() * params.jammer_power_std
    jammer_antenna_gain = params.jammer_antenna_gain_dbi + np.random.randn() * params.jammer_antenna_gain_std
    signal_power = params.signal_power_dbm + np.random.randn() * params.signal_power_std

    # Compute spatial correlation (Gudmundson model)
    # rho = exp(-d_separation / decorrelation_distance)
    # d_separation = distance between signal-path midpoint and jammer-path midpoint
    if params.shadow_correlation > 0:
        rho = float(np.clip(params.shadow_correlation, 0.0, 1.0))
    elif params.decorrelation_distance_m > 0:
        # Auto-compute from path separation
        d_sep = abs(params.signal_distance_m - params.jammer_distance_m) / 2
        rho = float(np.exp(-d_sep / params.decorrelation_distance_m))
    else:
        rho = 0.0

    # Shadow fading with optional correlation
    if params.shadow_distribution == "student_t":
        from scipy.stats import t
        scale = params.path_loss_std / np.sqrt(params.shadow_df / (params.shadow_df - 2))
        z1 = float(t.rvs(params.shadow_df)) * scale
        z2 = float(t.rvs(params.shadow_df)) * scale
        shadow_fading_signal = z1
        shadow_fading_jammer = rho * z1 + np.sqrt(max(0.0, 1 - rho**2)) * z2
    elif params.shadow_distribution == "mixture":
        z1 = np.random.randn() * params.path_loss_std
        z2 = np.random.randn() * params.path_loss_std
        if np.random.random() > 0.9:
            z1 *= 3.0
        if np.random.random() > 0.9:
            z2 *= 3.0
        shadow_fading_signal = z1
        shadow_fading_jammer = rho * z1 + np.sqrt(max(0.0, 1 - rho**2)) * z2
    else:  # normal (default) — bivariate normal with correlation rho
        z1 = np.random.randn()
        z2 = np.random.randn()
        # Cholesky-style: signal = z1*sigma, jammer = (rho*z1 + sqrt(1-rho^2)*z2)*sigma
        shadow_fading_signal = z1 * params.path_loss_std
        shadow_fading_jammer = (rho * z1 + np.sqrt(max(0.0, 1 - rho**2)) * z2) * params.path_loss_std

    # Rice fading margin (always positive excess loss)
    fading_margin = np.abs(np.random.randn() * params.fading_margin_std)

    # --- Select propagation model ---
    if params.propagation_model == "al_hourani":
        model = AlHouraniA2GModel(environment=params.environment)
    else:
        model = AltitudeDependentModel()

    # --- Atmospheric absorption (ITU-R P.676) - significant for >5 km ---
    atm_loss_signal = 0.0
    atm_loss_jammer = 0.0
    if params.enable_atmospheric:
        atm_loss_signal = AtmosphericAbsorption.total_loss_db(
            params.signal_distance_m, params.frequency_mhz,
            params.temperature_c, params.humidity_percent
        )
        atm_loss_jammer = AtmosphericAbsorption.total_loss_db(
            params.jammer_distance_m, params.frequency_mhz,
            params.temperature_c, params.humidity_percent
        )

    # --- Urban multi-path correction (NLOS additional loss) ---
    mp_loss_signal = 0.0
    mp_loss_jammer = 0.0
    if params.enable_multipath:
        mp_loss_signal = UrbanMultiPathCorrection.multi_path_loss_db(
            params.signal_distance_m, params.environment, params.altitude_m
        )
        mp_loss_jammer = UrbanMultiPathCorrection.multi_path_loss_db(
            params.jammer_distance_m, params.environment, params.altitude_m
        )

    # --- Path loss calculation (base + atmospheric + multi-path + fading) ---
    pl_signal = (model.path_loss(
        params.signal_distance_m, params.altitude_m, params.frequency_mhz
    ) + shadow_fading_signal + fading_margin
        + atm_loss_signal + mp_loss_signal)

    pl_jammer = (model.path_loss(
        params.jammer_distance_m, params.altitude_m, params.frequency_mhz
    ) + shadow_fading_jammer
        + atm_loss_jammer + mp_loss_jammer)

    # --- Antenna pattern (directional gain reduction for off-boresight targets) ---
    effective_jammer_gain = jammer_antenna_gain
    if params.jammer_beam_width_deg > 0:
        pattern = CosinePattern(
            beam_width_deg=params.jammer_beam_width_deg,
            max_gain_dbi=jammer_antenna_gain
        )
        # Compute elevation angle from jammer to target
        if params.jammer_distance_m > 0:
            elev_deg = np.degrees(np.arctan(
                params.altitude_m / params.jammer_distance_m
            ))
        else:
            elev_deg = 90.0
        # Random azimuth offset (jammer may not be perfectly aimed)
        azimuth_offset = np.random.randn() * 5.0  # +/-5 deg pointing error
        off_boresight = np.sqrt(elev_deg**2 + azimuth_offset**2)
        effective_jammer_gain = pattern.gain_dbi(off_boresight)

    # --- Received powers ---
    rx_signal = signal_power + params.signal_antenna_gain_dbi - pl_signal
    rx_jammer = jammer_power + effective_jammer_gain - pl_jammer

    # J/S ratio
    js_ratio = rx_jammer - rx_signal

    # --- FHSS effectiveness reduction ---
    fhss_effectiveness = 1.0
    if params.fhss_enabled:
        protocol = OcuSyncProtocol()
        analyzer = JammingEffectivenessAnalyzer(protocol)
        strategy = JammingStrategy(params.jamming_strategy)
        fhss_effectiveness = analyzer.calculate_effectiveness(strategy, 83.5)

        # Doppler degradation of FHSS hop synchronization
        if params.target_velocity_ms > 0:
            doppler_degradation = DopplerModel.hop_sync_degradation(
                params.frequency_mhz, params.target_velocity_ms,
                channel_bandwidth_mhz=2.0
            )
            # Doppler degrades FHSS protection (jammer benefits)
            fhss_effectiveness = fhss_effectiveness + (1.0 - fhss_effectiveness) * doppler_degradation

    # --- Determine jamming success using BER/PER model ---
    effective_js = js_ratio if not params.fhss_enabled else js_ratio * fhss_effectiveness
    jam_prob = BERModel.jamming_success_probability(
        js_ratio_db=effective_js,
        signal_power_db=rx_signal,
        modulation=ModulationType.QPSK
    )
    jamming_success = np.random.random() < jam_prob

    return SimulationResult(
        js_ratio_db=js_ratio,
        jamming_success=jamming_success,
        received_signal_dbm=rx_signal,
        received_jammer_dbm=rx_jammer,
        path_loss_signal_db=pl_signal,
        path_loss_jammer_db=pl_jammer,
        fhss_effectiveness=fhss_effectiveness
    )


class MonteCarloEngine:
    """
    Monte Carlo simulation engine with parallel processing support.
    """
    
    def __init__(self, n_processes: int = None):
        """
        Initialize Monte Carlo engine.

        Args:
            n_processes: Number of parallel processes (default: auto-detect)
        """
        import platform
        if n_processes:
            self.n_processes = n_processes
        elif platform.system() == 'Windows':
            # Windows uses spawn (not fork), each worker reimports scipy (~200MB).
            # Cap at 2 to avoid paging file exhaustion.
            self.n_processes = min(2, max(1, cpu_count() - 1))
        else:
            self.n_processes = max(1, cpu_count() - 1)
        self.results_history: List[MCResult] = []

    @staticmethod
    def required_sample_size(desired_precision_db: float = 0.1,
                             confidence_level: float = 0.95,
                             estimated_std: float = 5.0) -> int:
        """
        Calculate theoretically required sample size for a given precision.

        Uses N = (z * sigma / epsilon)^2 where z is the critical value,
        sigma is estimated std, epsilon is desired half-width.

        Args:
            desired_precision_db: Desired half-width of CI in dB
            confidence_level: Confidence level (e.g. 0.95)
            estimated_std: Estimated standard deviation in dB

        Returns:
            Required number of iterations
        """
        z = scipy_stats.norm.ppf(1 - (1 - confidence_level) / 2)
        n = int(np.ceil((z * estimated_std / desired_precision_db) ** 2))
        return max(100, n)

    @staticmethod
    def _bootstrap_ci(js_values: np.ndarray, n_bootstrap: int = 1000,
                      ci_level: float = 0.95) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Compute bootstrap confidence intervals on the CI bounds themselves.

        Resamples the JS values B times, computes the percentile-based CI
        for each resample, then reports the uncertainty on the CI endpoints.

        Args:
            js_values: Array of J/S values
            n_bootstrap: Number of bootstrap resamples
            ci_level: Confidence level for the CI

        Returns:
            Tuple of ((ci_lower_low, ci_lower_high), (ci_upper_low, ci_upper_high))
        """
        alpha = (1 - ci_level) / 2
        n = len(js_values)

        boot_lowers = np.empty(n_bootstrap)
        boot_uppers = np.empty(n_bootstrap)

        for b in range(n_bootstrap):
            resample = js_values[np.random.randint(0, n, size=n)]
            boot_lowers[b] = np.percentile(resample, alpha * 100)
            boot_uppers[b] = np.percentile(resample, (1 - alpha) * 100)

        ci_lower_uncertainty = (float(np.percentile(boot_lowers, 2.5)),
                                float(np.percentile(boot_lowers, 97.5)))
        ci_upper_uncertainty = (float(np.percentile(boot_uppers, 2.5)),
                                float(np.percentile(boot_uppers, 97.5)))

        return ci_lower_uncertainty, ci_upper_uncertainty

    @staticmethod
    def _check_convergence(js_values: np.ndarray,
                           threshold: float = 0.05) -> bool:
        """
        Check if the simulation has converged.

        Convergence criterion: relative half-width of the 95% CI.
        R = (ci_upper - ci_lower) / (2 * |mean|) < threshold

        For J/S ratio distributions with std ~ 5-10 dB:
        - threshold=0.01 (1%) requires N ~ 300K+
        - threshold=0.05 (5%) requires N ~ 10K-30K (practical default)
        - threshold=0.10 (10%) requires N ~ 3K-5K

        Args:
            js_values: Array of J/S values so far
            threshold: Convergence threshold (default 0.05 = 5% relative precision)

        Returns:
            True if converged
        """
        mean = np.mean(js_values)
        if abs(mean) < 1e-10:
            return False
        ci_lower, ci_upper = np.percentile(js_values, [2.5, 97.5])
        relative_half_width = (ci_upper - ci_lower) / (2 * abs(mean))
        return relative_half_width < threshold
    
    def run_simulation_qmc(self, params: SimulationParams,
                            n_iterations: int = 10000,
                            random_seed: int = None) -> MCResult:
        """
        Run MC using Quasi-Monte Carlo (Sobol sequence).

        Sobol sequences provide low-discrepancy sampling that achieves
        O((log N)^d / N) convergence vs O(1/sqrt(N)) for standard MC,
        especially effective for moderate dimensions (d <= 10).

        Args:
            params: Simulation parameters
            n_iterations: Number of iterations
            random_seed: Reproducibility seed

        Returns:
            MCResult with reduced variance compared to standard MC
        """
        from scipy.stats import qmc, norm
        start_time = time.time()
        seed = random_seed if random_seed is not None else 42

        # Generate Sobol points in [0,1]^d for 8 uncertain inputs
        n_params = 8
        sampler = qmc.Sobol(d=n_params, scramble=True, seed=seed)
        u_samples = sampler.random(n_iterations)
        # Transform uniforms to standard normals via inverse CDF
        z_samples = norm.ppf(u_samples)
        # Avoid -inf at boundaries
        z_samples = np.clip(z_samples, -8, 8)

        results = [_simulate_with_normals(params, z_samples[i]) for i in range(n_iterations)]
        return self._aggregate_results(results, n_iterations, time.time() - start_time)

    def run_simulation_antithetic(self, params: SimulationParams,
                                   n_iterations: int = 10000,
                                   random_seed: int = None) -> MCResult:
        """
        Run MC with antithetic variates: pair each sample with its negation.

        For each sample with normals z = (z1, ..., zk), also generate -z.
        Reduces variance for monotonically-related outputs by canceling
        symmetric components.

        Args:
            params: Simulation parameters
            n_iterations: Number of iterations (will be made even)
            random_seed: Reproducibility seed

        Returns:
            MCResult with reduced variance
        """
        start_time = time.time()
        seed = random_seed if random_seed is not None else 42
        rng = np.random.default_rng(seed)

        n_pairs = n_iterations // 2
        n_iterations = 2 * n_pairs  # ensure even

        results = []
        for i in range(n_pairs):
            z = rng.standard_normal(8)
            results.append(_simulate_with_normals(params, z))
            results.append(_simulate_with_normals(params, -z))  # antithetic

        return self._aggregate_results(results, n_iterations, time.time() - start_time)

    def _aggregate_results(self, results, n_iterations, exec_time):
        """Common aggregation logic for all run_simulation_* variants."""
        js_values = np.array([r.js_ratio_db for r in results])
        success_count = sum(1 for r in results if r.jamming_success)
        mean_js = np.mean(js_values)
        std_js = np.std(js_values)
        ci_lower, ci_upper = np.percentile(js_values, [2.5, 97.5])
        ci_lower_unc, ci_upper_unc = self._bootstrap_ci(js_values)

        return MCResult(
            n_iterations=n_iterations,
            mean_js_db=mean_js,
            std_js_db=std_js,
            ci_95_lower=ci_lower,
            ci_95_upper=ci_upper,
            success_probability=success_count / n_iterations,
            js_values=js_values,
            percentiles={p: float(np.percentile(js_values, p)) for p in [5, 25, 50, 75, 95]},
            ci_lower_uncertainty=ci_lower_unc,
            ci_upper_uncertainty=ci_upper_unc,
            converged=self._check_convergence(js_values),
            execution_time_s=exec_time
        )

    def run_simulation(self, params: SimulationParams,
                       n_iterations: int = 1000000,
                       parallel: bool = True,
                       convergence_threshold: float = 0.0,
                       random_seed: int = None) -> MCResult:
        """
        Run Monte Carlo simulation.

        Args:
            params: Simulation parameters
            n_iterations: Number of iterations
            parallel: Whether to use parallel processing
            convergence_threshold: If > 0, enable adaptive stopping (0 = disabled)
            random_seed: Fixed seed for reproducibility (None = time-based)

        Returns:
            MCResult with aggregated results
        """
        start_time = time.time()

        # Check theoretical sample size requirement
        estimated_std = 5.0  # Conservative estimate for J/S std
        req_n = self.required_sample_size(
            desired_precision_db=0.1, confidence_level=0.95,
            estimated_std=estimated_std
        )
        if n_iterations < req_n:
            warnings.warn(
                f"Requested N={n_iterations} may be insufficient for 0.1 dB precision "
                f"(theoretical minimum: {req_n}). Consider increasing iterations.",
                stacklevel=2
            )

        # Prepare arguments for each iteration (deterministic if seed provided)
        if random_seed is not None:
            base_seed = random_seed
        else:
            base_seed = int(time.time() * 1000) % (2**31)
        args_list = [(params, base_seed + i) for i in range(n_iterations)]

        # Run simulation
        if parallel and n_iterations > 100:
            with Pool(self.n_processes) as pool:
                results = pool.map(_simulate_single_iteration, args_list)
        else:
            results = [_simulate_single_iteration(args) for args in args_list]

        # Aggregate results
        js_values = np.array([r.js_ratio_db for r in results])
        success_count = sum(1 for r in results if r.jamming_success)

        # Calculate statistics
        mean_js = np.mean(js_values)
        std_js = np.std(js_values)
        ci_lower, ci_upper = np.percentile(js_values, [2.5, 97.5])

        # Bootstrap CI uncertainty
        ci_lower_unc, ci_upper_unc = self._bootstrap_ci(js_values)

        # Convergence check
        converged = self._check_convergence(
            js_values, convergence_threshold if convergence_threshold > 0 else 0.05
        )

        # Percentiles
        percentiles = {
            5: np.percentile(js_values, 5),
            25: np.percentile(js_values, 25),
            50: np.percentile(js_values, 50),
            75: np.percentile(js_values, 75),
            95: np.percentile(js_values, 95)
        }

        execution_time = time.time() - start_time

        result = MCResult(
            n_iterations=n_iterations,
            mean_js_db=mean_js,
            std_js_db=std_js,
            ci_95_lower=ci_lower,
            ci_95_upper=ci_upper,
            success_probability=success_count / n_iterations,
            js_values=js_values,
            percentiles=percentiles,
            ci_lower_uncertainty=ci_lower_unc,
            ci_upper_uncertainty=ci_upper_unc,
            converged=converged,
            convergence_iteration=n_iterations if converged else 0,
            required_sample_size=req_n,
            execution_time_s=execution_time
        )

        self.results_history.append(result)
        return result

    def run_simulation_adaptive(self, params: SimulationParams,
                                convergence_threshold: float = 0.05,
                                min_iterations: int = 10000,
                                max_iterations: int = 1000000,
                                batch_size: int = 5000) -> MCResult:
        """
        Run Monte Carlo with adaptive stopping based on convergence criterion.

        Starts with min_iterations and adds batches until the relative half-width
        of the 95% CI drops below convergence_threshold, or max_iterations is reached.

        Args:
            params: Simulation parameters
            convergence_threshold: Stop when relative CI half-width < this value
            min_iterations: Minimum number of iterations before checking
            max_iterations: Maximum iterations (safety cap)
            batch_size: Number of iterations per batch after initial run

        Returns:
            MCResult with convergence info populated
        """
        start_time = time.time()
        all_results = []
        base_seed = int(time.time() * 1000) % (2**31)
        convergence_iteration = 0

        # Initial batch
        args_list = [(params, base_seed + i) for i in range(min_iterations)]
        with Pool(self.n_processes) as pool:
            all_results = pool.map(_simulate_single_iteration, args_list)

        total_n = min_iterations
        js_values = np.array([r.js_ratio_db for r in all_results])

        # Check convergence and add batches
        if self._check_convergence(js_values, convergence_threshold):
            convergence_iteration = total_n
        else:
            while total_n < max_iterations:
                remaining = min(batch_size, max_iterations - total_n)
                args_list = [(params, base_seed + total_n + i) for i in range(remaining)]
                batch_results = [_simulate_single_iteration(a) for a in args_list]
                all_results.extend(batch_results)
                total_n += remaining

                js_values = np.array([r.js_ratio_db for r in all_results])
                if self._check_convergence(js_values, convergence_threshold):
                    convergence_iteration = total_n
                    break

        # Aggregate final results
        success_count = sum(1 for r in all_results if r.jamming_success)
        mean_js = np.mean(js_values)
        std_js = np.std(js_values)
        ci_lower, ci_upper = np.percentile(js_values, [2.5, 97.5])
        ci_lower_unc, ci_upper_unc = self._bootstrap_ci(js_values)

        percentiles = {
            5: np.percentile(js_values, 5),
            25: np.percentile(js_values, 25),
            50: np.percentile(js_values, 50),
            75: np.percentile(js_values, 75),
            95: np.percentile(js_values, 95)
        }

        req_n = self.required_sample_size(estimated_std=std_js)
        execution_time = time.time() - start_time

        result = MCResult(
            n_iterations=total_n,
            mean_js_db=mean_js,
            std_js_db=std_js,
            ci_95_lower=ci_lower,
            ci_95_upper=ci_upper,
            success_probability=success_count / total_n,
            js_values=js_values,
            percentiles=percentiles,
            ci_lower_uncertainty=ci_lower_unc,
            ci_upper_uncertainty=ci_upper_unc,
            converged=convergence_iteration > 0,
            convergence_iteration=convergence_iteration,
            required_sample_size=req_n,
            execution_time_s=execution_time
        )

        self.results_history.append(result)
        return result
    
    def sensitivity_analysis(self, base_params: SimulationParams,
                            n_iterations: int = 1000000) -> Dict[str, float]:
        """
        Perform sensitivity analysis using one-at-a-time method.
        Based on Saltelli et al. methodology.
        
        Args:
            base_params: Base simulation parameters
            n_iterations: Iterations per parameter variation
            
        Returns:
            Dictionary of parameter sensitivities (dB change per std)
        """
        # Parameters to analyze with their std values
        param_variations = {
            'jammer_power': ('jammer_power_dbm', base_params.jammer_power_std),
            'jammer_antenna': ('jammer_antenna_gain_dbi', base_params.jammer_antenna_gain_std),
            'signal_power': ('signal_power_dbm', base_params.signal_power_std),
            'rx_sensitivity': ('rx_sensitivity_dbm', base_params.rx_sensitivity_std),
            'path_loss': ('path_loss_std', base_params.path_loss_std),
            'fading_margin': ('fading_margin_std', base_params.fading_margin_std)
        }
        
        # Baseline simulation
        baseline = self.run_simulation(base_params, n_iterations, parallel=False)
        baseline_mean = baseline.mean_js_db
        
        sensitivities = {}
        
        for param_name, (attr_name, std_value) in param_variations.items():
            # Simulation with increased parameter
            modified_params = SimulationParams(**vars(base_params))
            current_value = getattr(modified_params, attr_name)
            setattr(modified_params, attr_name, current_value + std_value)
            
            result_high = self.run_simulation(modified_params, n_iterations, parallel=False)
            
            # Simulation with decreased parameter
            setattr(modified_params, attr_name, current_value - std_value)
            result_low = self.run_simulation(modified_params, n_iterations, parallel=False)
            
            # Sensitivity = change in output per std of input
            sensitivity = (result_high.mean_js_db - result_low.mean_js_db) / 2
            sensitivities[param_name] = abs(sensitivity)
        
        return sensitivities
    
    def convergence_analysis(self, params: SimulationParams,
                            max_iterations: int = 1000000,
                            check_interval: int = 500,
                            convergence_threshold: float = 0.01
                            ) -> List[Tuple[int, float, float, float, bool]]:
        """
        Analyze convergence of Monte Carlo simulation.

        Args:
            params: Simulation parameters
            max_iterations: Maximum iterations
            check_interval: Interval to check convergence
            convergence_threshold: Relative half-width threshold for convergence

        Returns:
            List of (n_iterations, mean, std, relative_half_width, converged) tuples
        """
        convergence_data = []
        all_results = []

        for n in range(check_interval, max_iterations + 1, check_interval):
            # Run additional iterations
            base_seed = int(time.time() * 1000) % (2**31) + len(all_results)
            args_list = [(params, base_seed + i) for i in range(check_interval)]

            new_results = [_simulate_single_iteration(args) for args in args_list]
            all_results.extend(new_results)

            # Calculate running statistics
            js_values = np.array([r.js_ratio_db for r in all_results])
            mean = np.mean(js_values)
            std = np.std(js_values)

            # Compute relative half-width
            ci_lower, ci_upper = np.percentile(js_values, [2.5, 97.5])
            rel_hw = (ci_upper - ci_lower) / (2 * abs(mean)) if abs(mean) > 1e-10 else float('inf')
            converged = rel_hw < convergence_threshold

            convergence_data.append((n, mean, std, rel_hw, converged))

            # Early stop once converged (continue a few more for verification)
            if converged and n >= check_interval * 3:
                # Verify convergence is stable over 2 more intervals
                remaining_checks = 2
                for _ in range(remaining_checks):
                    n += check_interval
                    if n > max_iterations:
                        break
                    bs = int(time.time() * 1000) % (2**31) + len(all_results)
                    extra_args = [(params, bs + i) for i in range(check_interval)]
                    extra_results = [_simulate_single_iteration(a) for a in extra_args]
                    all_results.extend(extra_results)
                    js_values = np.array([r.js_ratio_db for r in all_results])
                    mean = np.mean(js_values)
                    std = np.std(js_values)
                    ci_l, ci_u = np.percentile(js_values, [2.5, 97.5])
                    rel_hw = (ci_u - ci_l) / (2 * abs(mean)) if abs(mean) > 1e-10 else float('inf')
                    still_converged = rel_hw < convergence_threshold
                    convergence_data.append((n, mean, std, rel_hw, still_converged))
                break

        return convergence_data


class ScenarioSimulator:
    """
    Simulates specific jamming scenarios from the paper.
    """
    
    def __init__(self):
        """Initialize scenario simulator."""
        self.engine = MonteCarloEngine()
    
    def simulate_portable_system(self, power_w: float, distance_m: float,
                                 target_distance_m: float = 5000,
                                 altitude_m: float = 100,
                                 n_iterations: int = 1000000) -> MCResult:
        """
        Simulate portable jamming system scenario.
        
        Args:
            power_w: Jammer power in Watts
            distance_m: Jammer to target distance
            target_distance_m: Operator to target distance
            altitude_m: Target altitude
            n_iterations: Monte Carlo iterations
            
        Returns:
            MCResult
        """
        power_dbm = 10 * np.log10(power_w * 1000)  # Convert W to dBm
        
        params = SimulationParams(
            jammer_power_dbm=power_dbm,
            jammer_distance_m=distance_m,
            jammer_antenna_gain_dbi=6.0,
            signal_distance_m=target_distance_m,
            altitude_m=altitude_m,
            fhss_enabled=False  # Base case without FHSS
        )
        
        return self.engine.run_simulation(params, n_iterations)
    
    def simulate_mobile_system(self, power_w: float, distance_m: float,
                               n_iterations: int = 1000000) -> MCResult:
        """Simulate mobile jamming system (higher power, vehicle-mounted)."""
        power_dbm = 10 * np.log10(power_w * 1000)
        
        params = SimulationParams(
            jammer_power_dbm=power_dbm,
            jammer_distance_m=distance_m,
            jammer_antenna_gain_dbi=10.0,  # Directional antenna
            signal_distance_m=8000,
            altitude_m=200,
            fhss_enabled=False
        )
        
        return self.engine.run_simulation(params, n_iterations)
    
    def simulate_with_fhss(self, power_w: float, distance_m: float,
                           strategy: str = "broadband",
                           n_iterations: int = 1000000) -> MCResult:
        """
        Simulate jamming against FHSS-enabled target.
        
        Args:
            power_w: Jammer power in Watts
            distance_m: Distance to target
            strategy: Jamming strategy ('broadband', 'narrowband', etc.)
            n_iterations: Monte Carlo iterations
            
        Returns:
            MCResult with FHSS effects
        """
        power_dbm = 10 * np.log10(power_w * 1000)
        
        params = SimulationParams(
            jammer_power_dbm=power_dbm,
            jammer_distance_m=distance_m,
            jammer_antenna_gain_dbi=6.0,
            signal_distance_m=5000,
            altitude_m=100,
            fhss_enabled=True,
            jamming_strategy=strategy
        )
        
        return self.engine.run_simulation(params, n_iterations)
    
    def run_paper_scenarios(self, n_iterations: int = 1000000) -> Dict[str, MCResult]:
        """
        Run all scenarios from the paper (Table 3).
        
        Returns:
            Dictionary of scenario results
        """
        scenarios = {
            'portable_10W_500m': self.simulate_portable_system(10, 500, n_iterations=n_iterations),
            'portable_10W_1000m': self.simulate_portable_system(10, 1000, n_iterations=n_iterations),
            'mobile_100W_2000m': self.simulate_mobile_system(100, 2000, n_iterations=n_iterations),
            'stationary_500W_3000m': self.simulate_mobile_system(500, 3000, n_iterations=n_iterations)
        }
        
        return scenarios


@dataclass
class MultiJammerParams:
    """Parameters for multi-jammer simulation."""
    jammers: List[SimulationParams] = field(default_factory=list)
    signal_power_dbm: float = 20.0
    signal_distance_m: float = 5000.0
    signal_antenna_gain_dbi: float = 2.0
    altitude_m: float = 100.0
    frequency_mhz: float = 2437.0
    fhss_enabled: bool = False


def _simulate_multi_jammer_iteration(args: Tuple) -> SimulationResult:
    """
    Execute single MC iteration for multi-jammer scenario.
    Combines jammer powers in linear domain.
    """
    multi_params, seed = args
    np.random.seed(seed)

    model = AltitudeDependentModel()

    # Signal path
    signal_power = multi_params.signal_power_dbm + np.random.randn() * 1.0
    shadow_signal = np.random.randn() * 3.0
    fading_margin = np.abs(np.random.randn() * 5.0)
    pl_signal = model.path_loss(
        multi_params.signal_distance_m, multi_params.altitude_m,
        multi_params.frequency_mhz
    ) + shadow_signal + fading_margin
    rx_signal = signal_power + multi_params.signal_antenna_gain_dbi - pl_signal

    # Combine jammer powers in linear domain
    total_jammer_linear = 0.0
    for jammer in multi_params.jammers:
        j_power = jammer.jammer_power_dbm + np.random.randn() * jammer.jammer_power_std
        j_antenna = jammer.jammer_antenna_gain_dbi + np.random.randn() * jammer.jammer_antenna_gain_std
        shadow_j = np.random.randn() * jammer.path_loss_std
        pl_j = model.path_loss(
            jammer.jammer_distance_m, multi_params.altitude_m,
            multi_params.frequency_mhz
        ) + shadow_j
        rx_j_dbm = j_power + j_antenna - pl_j
        total_jammer_linear += 10 ** (rx_j_dbm / 10)

    rx_jammer = 10 * np.log10(max(total_jammer_linear, 1e-20))
    js_ratio = rx_jammer - rx_signal

    # BER-based success
    jam_prob = BERModel.jamming_success_probability(js_ratio_db=js_ratio, signal_power_db=rx_signal)
    jamming_success = np.random.random() < jam_prob

    return SimulationResult(
        js_ratio_db=js_ratio,
        jamming_success=jamming_success,
        received_signal_dbm=rx_signal,
        received_jammer_dbm=rx_jammer,
        path_loss_signal_db=pl_signal,
        path_loss_jammer_db=0.0,
        fhss_effectiveness=1.0
    )


class MultiJammerSimulator:
    """Simulates scenarios with multiple coordinated jammers."""

    def __init__(self):
        self.engine = MonteCarloEngine()

    def simulate(self, multi_params: MultiJammerParams,
                 n_iterations: int = 10000) -> MCResult:
        """
        Run multi-jammer Monte Carlo simulation.

        Args:
            multi_params: Multi-jammer configuration
            n_iterations: Number of iterations

        Returns:
            MCResult with combined J/S statistics
        """
        start_time = time.time()
        base_seed = int(time.time() * 1000) % (2**31)
        args_list = [(multi_params, base_seed + i) for i in range(n_iterations)]

        results = [_simulate_multi_jammer_iteration(a) for a in args_list]

        js_values = np.array([r.js_ratio_db for r in results])
        success_count = sum(1 for r in results if r.jamming_success)

        mean_js = np.mean(js_values)
        std_js = np.std(js_values)
        ci_lower, ci_upper = np.percentile(js_values, [2.5, 97.5])

        return MCResult(
            n_iterations=n_iterations,
            mean_js_db=mean_js,
            std_js_db=std_js,
            ci_95_lower=ci_lower,
            ci_95_upper=ci_upper,
            success_probability=success_count / n_iterations,
            js_values=js_values,
            percentiles={p: float(np.percentile(js_values, p)) for p in [5, 25, 50, 75, 95]},
            execution_time_s=time.time() - start_time
        )


if __name__ == "__main__":
    print("=" * 60)
    print("UAV-CPS-Analyzer: Monte Carlo Engine Test")
    print("=" * 60)
    
    # Initialize engine
    engine = MonteCarloEngine()
    print(f"\nUsing {engine.n_processes} parallel processes")
    
    # Basic simulation test
    print("\n1. Basic Simulation (N=10,000):")
    params = SimulationParams(
        jammer_power_dbm=40,  # 10W
        jammer_distance_m=500,
        signal_distance_m=5000,
        altitude_m=100,
        fhss_enabled=False
    )
    
    result = engine.run_simulation(params, n_iterations=10000)
    print(f"   J/S Mean: {result.mean_js_db:.1f} dB")
    print(f"   J/S Std:  {result.std_js_db:.1f} dB")
    print(f"   95% CI:   [{result.ci_95_lower:.1f}, {result.ci_95_upper:.1f}] dB")
    print(f"   Success:  {result.success_probability*100:.1f}%")
    print(f"   Time:     {result.execution_time_s:.2f} s")
    
    # Scenario simulator
    print("\n2. Paper Scenarios (Table 3):")
    simulator = ScenarioSimulator()
    scenarios = simulator.run_paper_scenarios(n_iterations=5000)
    
    print(f"   {'Scenario':<25} {'J/S (dB)':<12} {'95% CI':<20} {'P(success)'}")
    print(f"   {'-'*70}")
    for name, res in scenarios.items():
        ci_str = f"[{res.ci_95_lower:.1f}, {res.ci_95_upper:.1f}]"
        print(f"   {name:<25} {res.mean_js_db:<12.1f} {ci_str:<20} {res.success_probability*100:.0f}%")
    
    # FHSS comparison
    print("\n3. FHSS Impact (BBN strategy):")
    result_fhss = simulator.simulate_with_fhss(10, 500, "broadband", n_iterations=5000)
    print(f"   With FHSS:    J/S={result_fhss.mean_js_db:.1f} dB, "
          f"Success={result_fhss.success_probability*100:.1f}%")
    
    # Sensitivity analysis
    print("\n4. Sensitivity Analysis:")
    sensitivities = engine.sensitivity_analysis(params, n_iterations=1000)
    sorted_sens = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
    for param, sens in sorted_sens:
        print(f"   {param:<20}: ±{sens:.1f} dB")
    
    print("\n" + "=" * 60)
    print("Tests completed successfully!")
