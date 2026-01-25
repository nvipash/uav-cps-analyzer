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
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional
from multiprocessing import Pool, cpu_count
import time
from enum import Enum

# Import local modules
try:
    from propagation_models import (
        AltitudeDependentModel, calculate_js_ratio, RiceFadingModel
    )
    from fhss_emulator import (
        OcuSyncProtocol, JammingStrategy, JammingEffectivenessAnalyzer
    )
except ImportError:
    # For standalone testing
    import sys
    sys.path.insert(0, '.')
    from propagation_models import (
        AltitudeDependentModel, calculate_js_ratio, RiceFadingModel
    )
    from fhss_emulator import (
        OcuSyncProtocol, JammingStrategy, JammingEffectivenessAnalyzer
    )


@dataclass
class SimulationParams:
    """Parameters for Monte Carlo simulation."""
    # Jammer parameters
    jammer_power_dbm: float = 40.0       # 10W
    jammer_power_std: float = 1.0        # ±1 dB uncertainty
    jammer_distance_m: float = 500.0
    jammer_antenna_gain_dbi: float = 6.0
    jammer_antenna_gain_std: float = 1.5  # ±1.5 dB uncertainty
    
    # Signal (UAV) parameters
    signal_power_dbm: float = 20.0       # 100mW
    signal_power_std: float = 1.0        # ±1 dB uncertainty
    signal_distance_m: float = 5000.0
    signal_antenna_gain_dbi: float = 2.0
    
    # Target parameters
    altitude_m: float = 100.0
    frequency_mhz: float = 2437.0
    
    # Receiver parameters
    rx_sensitivity_dbm: float = -90.0
    rx_sensitivity_std: float = 2.0      # ±2 dB uncertainty
    
    # Propagation uncertainty
    path_loss_std: float = 3.0           # Shadow fading std
    fading_margin_std: float = 5.0       # Rice fading margin uncertainty
    
    # FHSS parameters
    fhss_enabled: bool = True
    jamming_strategy: str = "broadband"
    
    # Jamming threshold
    js_threshold_db: float = 10.0        # Required J/S for successful jamming


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
    
    # Sensitivity analysis
    sensitivity: Dict[str, float] = field(default_factory=dict)
    
    # Timing
    execution_time_s: float = 0.0


def _simulate_single_iteration(args: Tuple[SimulationParams, int]) -> SimulationResult:
    """
    Execute single Monte Carlo iteration.
    
    Args:
        args: Tuple of (SimulationParams, random_seed)
        
    Returns:
        SimulationResult for this iteration
    """
    params, seed = args
    np.random.seed(seed)
    
    # Sample uncertain parameters
    jammer_power = params.jammer_power_dbm + np.random.randn() * params.jammer_power_std
    jammer_antenna = params.jammer_antenna_gain_dbi + np.random.randn() * params.jammer_antenna_gain_std
    signal_power = params.signal_power_dbm + np.random.randn() * params.signal_power_std
    rx_sensitivity = params.rx_sensitivity_dbm + np.random.randn() * params.rx_sensitivity_std
    
    # Shadow fading (log-normal)
    shadow_fading_signal = np.random.randn() * params.path_loss_std
    shadow_fading_jammer = np.random.randn() * params.path_loss_std
    
    # Rice fading margin
    fading_margin = np.abs(np.random.randn() * params.fading_margin_std)
    
    # Calculate path losses using altitude-dependent model
    model = AltitudeDependentModel()
    
    pl_signal = model.path_loss(
        params.signal_distance_m,
        params.altitude_m,
        params.frequency_mhz
    ) + shadow_fading_signal + fading_margin
    
    pl_jammer = model.path_loss(
        params.jammer_distance_m,
        params.altitude_m,
        params.frequency_mhz
    ) + shadow_fading_jammer
    
    # Calculate received powers
    rx_signal = signal_power + params.signal_antenna_gain_dbi - pl_signal
    rx_jammer = jammer_power + jammer_antenna - pl_jammer
    
    # J/S ratio
    js_ratio = rx_jammer - rx_signal
    
    # FHSS effectiveness reduction
    fhss_effectiveness = 1.0
    if params.fhss_enabled:
        protocol = OcuSyncProtocol()
        analyzer = JammingEffectivenessAnalyzer(protocol)
        strategy = JammingStrategy(params.jamming_strategy)
        fhss_effectiveness = analyzer.calculate_effectiveness(strategy, 83.5)
    
    # Determine jamming success
    effective_js = js_ratio if not params.fhss_enabled else js_ratio * fhss_effectiveness
    jamming_success = effective_js > params.js_threshold_db
    
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
            n_processes: Number of parallel processes (default: CPU count)
        """
        self.n_processes = n_processes or max(1, cpu_count() - 1)
        self.results_history: List[MCResult] = []
    
    def run_simulation(self, params: SimulationParams, 
                       n_iterations: int = 10000,
                       parallel: bool = True) -> MCResult:
        """
        Run Monte Carlo simulation.
        
        Args:
            params: Simulation parameters
            n_iterations: Number of iterations
            parallel: Whether to use parallel processing
            
        Returns:
            MCResult with aggregated results
        """
        start_time = time.time()
        
        # Prepare arguments for each iteration
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
            execution_time_s=execution_time
        )
        
        self.results_history.append(result)
        return result
    
    def sensitivity_analysis(self, base_params: SimulationParams,
                            n_iterations: int = 1000) -> Dict[str, float]:
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
                            max_iterations: int = 10000,
                            check_interval: int = 500) -> List[Tuple[int, float, float]]:
        """
        Analyze convergence of Monte Carlo simulation.
        
        Args:
            params: Simulation parameters
            max_iterations: Maximum iterations
            check_interval: Interval to check convergence
            
        Returns:
            List of (n_iterations, mean, std) tuples
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
            
            convergence_data.append((n, mean, std))
        
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
                                 n_iterations: int = 10000) -> MCResult:
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
                               n_iterations: int = 10000) -> MCResult:
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
                           n_iterations: int = 10000) -> MCResult:
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
    
    def run_paper_scenarios(self, n_iterations: int = 10000) -> Dict[str, MCResult]:
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
