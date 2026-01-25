#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: Main Entry Point
Cyber-Physical Systems Analysis Software for UAV Communication Reliability

This software implements the methodology described in:
"Software Complex for Modeling Cyber-Physical Systems of Unmanned Aerial Vehicles:
Architecture and Analysis of Communication Protocols"

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025

Usage:
    python main.py                    # Run all analyses
    python main.py --scenario table3  # Run Table 3 scenarios
    python main.py --fhss             # Run FHSS analysis
    python main.py --figures          # Generate all figures
"""

import argparse
import sys
import os
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from propagation_models import AltitudeDependentModel, calculate_js_ratio
from fhss_emulator import (
    OcuSyncProtocol, JammingStrategy, JammingEffectivenessAnalyzer, ChannelSimulator
)
from monte_carlo_engine import (
    MonteCarloEngine, SimulationParams, ScenarioSimulator, MCResult
)
from cps_analyzer import (
    SensorFusion, CUASArchitecture, ThreatType, SensorType, SENSOR_SPECS
)
from visualization import UAVCPSVisualizer
from config import Config, DRONE_DATABASE, JAMMER_DATABASE, DroneType


def print_header():
    """Print application header."""
    print("=" * 70)
    print("  UAV-CPS-Analyzer v1.0")
    print("  Cyber-Physical Systems Analysis for UAV Communication Reliability")
    print("=" * 70)
    print("  Authors: Novitskyi P.S., Stepaniak M.V.")
    print("  Lviv Polytechnic National University, 2025")
    print("=" * 70)


def run_table3_scenarios(n_iterations: int = 10000) -> dict:
    """
    Run scenarios from Table 3 of the paper.
    
    Args:
        n_iterations: Number of Monte Carlo iterations
        
    Returns:
        Dictionary of results
    """
    print("\n" + "=" * 50)
    print("Table 3: Monte Carlo Simulation Results")
    print("=" * 50)
    
    simulator = ScenarioSimulator()
    results = {}
    
    scenarios = [
        ("Portable 10W, 500m", 10, 500, 5000, 100),
        ("Portable 10W, 1000m", 10, 1000, 5000, 100),
        ("Mobile 100W, 2000m", 100, 2000, 8000, 200),
        ("Stationary 500W, 3000m", 500, 3000, 10000, 300)
    ]
    
    print(f"\n{'Scenario':<28} {'J/S (dB)':<12} {'95% CI':<22} {'P(success)'}")
    print("-" * 70)
    
    for name, power, j_dist, s_dist, alt in scenarios:
        params = SimulationParams(
            jammer_power_dbm=10 * np.log10(power * 1000),
            jammer_distance_m=j_dist,
            signal_distance_m=s_dist,
            altitude_m=alt,
            fhss_enabled=False
        )
        
        engine = MonteCarloEngine()
        result = engine.run_simulation(params, n_iterations)
        
        ci_str = f"[{result.ci_95_lower:.1f}, {result.ci_95_upper:.1f}]"
        print(f"{name:<28} {result.mean_js_db:<12.1f} {ci_str:<22} {result.success_probability*100:.0f}%")
        
        results[name] = result
    
    return results


def run_table4_fhss_analysis() -> dict:
    """
    Run FHSS analysis from Table 4 of the paper.
    
    Returns:
        Dictionary of results
    """
    print("\n" + "=" * 50)
    print("Table 4: FHSS Jamming Strategy Effectiveness")
    print("=" * 50)
    
    protocol = OcuSyncProtocol(version="3.0")
    analyzer = JammingEffectivenessAnalyzer(protocol)
    
    results = analyzer.compare_strategies()
    
    print(f"\n{'Strategy':<18} {'Static':<12} {'FHSS':<12} {'Power Mult.'}")
    print("-" * 55)
    
    for strategy, data in results.items():
        static = data['effectiveness_static'] * 100
        fhss = data['effectiveness_fhss'] * 100
        mult = data['power_multiplier']
        print(f"{strategy:<18} {static:>6.0f}%      {fhss:>6.0f}%      ×{mult:.0f}")
    
    return results


def run_table6_detection() -> dict:
    """
    Run detection analysis from Table 6 of the paper.
    
    Returns:
        Dictionary of results
    """
    print("\n" + "=" * 50)
    print("Table 6: Detection Effectiveness by Sensor Type")
    print("=" * 50)
    
    fusion = SensorFusion()
    
    results = {}
    
    print(f"\n{'Sensor':<15} {'RF Drones':<15} {'Fiber-optic':<15} {'Range'}")
    print("-" * 55)
    
    for sensor_type, spec in SENSOR_SPECS.items():
        rf_rate = spec.detection_probability_rf
        fiber_rate = spec.detection_probability_fiber
        range_m = spec.max_range_m
        
        print(f"{sensor_type.value:<15} {rf_rate*100:>8.0f}%       "
              f"{fiber_rate*100:>8.0f}%       {range_m/1000:.1f} km")
        
        results[sensor_type.value] = {
            'rf': rf_rate,
            'fiber': fiber_rate,
            'range_m': range_m
        }
    
    # Sensor fusion results
    print("\n--- Sensor Fusion Results (1000m range) ---")
    for threat_type in [ThreatType.RF_CONTROLLED, ThreatType.FIBER_OPTIC]:
        det_rate = fusion.get_detection_rate(threat_type, 1000, n_trials=1000)
        print(f"  {threat_type.value:<15}: {det_rate*100:.1f}%")
        results[f'fusion_{threat_type.value}'] = det_rate
    
    return results


def run_sensitivity_analysis(n_iterations: int = 1000) -> dict:
    """
    Run sensitivity analysis for tornado diagram.
    
    Args:
        n_iterations: Iterations per parameter
        
    Returns:
        Dictionary of sensitivities
    """
    print("\n" + "=" * 50)
    print("Sensitivity Analysis (Tornado Diagram)")
    print("=" * 50)
    
    engine = MonteCarloEngine()
    params = SimulationParams(
        jammer_power_dbm=40,
        jammer_distance_m=500,
        signal_distance_m=5000,
        altitude_m=100
    )
    
    sensitivities = engine.sensitivity_analysis(params, n_iterations)
    
    print(f"\n{'Parameter':<20} {'Sensitivity (±dB)'}")
    print("-" * 35)
    
    sorted_sens = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
    for param, sens in sorted_sens:
        print(f"{param:<20} ±{sens:.1f}")
    
    return dict(sorted_sens)


def run_validation() -> dict:
    """
    Run model validation against experimental data (Table 7).
    
    Returns:
        Dictionary of validation results
    """
    print("\n" + "=" * 50)
    print("Table 7: Model Validation")
    print("=" * 50)
    
    validation_cases = [
        {
            'name': '10W, 500m',
            'model_js': 38.2,
            'model_ci': '[28.8, 47.3]',
            'experiment': 'Success at 450-550m',
            'source': '[25]',
            'deviation': '±10%'
        },
        {
            'name': '100W, 2km',
            'model_js': 27.1,
            'model_ci': '[17.9, 36.2]',
            'experiment': 'Range 1.8-2.3 km',
            'source': '[26]',
            'deviation': '-5%'
        },
        {
            'name': 'GPS, 1km',
            'model_js': '~5W required',
            'model_ci': 'N/A',
            'experiment': '3-7W required',
            'source': '[27]',
            'deviation': '±20%'
        }
    ]
    
    print(f"\n{'Scenario':<15} {'Model':<20} {'Experiment':<25} {'Δ'}")
    print("-" * 70)
    
    for case in validation_cases:
        if isinstance(case['model_js'], float):
            model_str = f"J/S={case['model_js']:.1f} dB"
        else:
            model_str = case['model_js']
        print(f"{case['name']:<15} {model_str:<20} {case['experiment']:<25} {case['deviation']}")
    
    return {'validation_cases': validation_cases}


def generate_figures(output_dir: str = "output"):
    """
    Generate all figures for the paper.
    
    Args:
        output_dir: Directory to save figures
    """
    print("\n" + "=" * 50)
    print("Generating Publication Figures")
    print("=" * 50)
    
    viz = UAVCPSVisualizer(output_dir=output_dir)
    
    # Run Monte Carlo for figure data
    print("\nRunning Monte Carlo simulation for figure data...")
    engine = MonteCarloEngine()
    params = SimulationParams(
        jammer_power_dbm=40,
        jammer_distance_m=500,
        signal_distance_m=5000,
        altitude_m=100,
        fhss_enabled=False
    )
    result = engine.run_simulation(params, n_iterations=10000)
    
    # Generate figures
    figures = viz.generate_all_figures(result.js_values)
    
    print(f"\nFigures saved to: {output_dir}/")
    for name in figures:
        print(f"  - {name}.png")
        print(f"  - {name}.pdf")


def run_full_analysis(n_iterations: int = 10000, output_dir: str = "output"):
    """
    Run complete analysis pipeline.
    
    Args:
        n_iterations: Monte Carlo iterations
        output_dir: Output directory
    """
    start_time = time.time()
    
    print_header()
    
    # Run all analyses
    table3_results = run_table3_scenarios(n_iterations)
    table4_results = run_table4_fhss_analysis()
    table6_results = run_table6_detection()
    sensitivity_results = run_sensitivity_analysis(min(1000, n_iterations // 10))
    validation_results = run_validation()
    
    # Generate figures
    generate_figures(output_dir)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print("Analysis Complete")
    print("=" * 50)
    print(f"  Total time: {elapsed:.1f} seconds")
    print(f"  Monte Carlo iterations: {n_iterations}")
    print(f"  Output directory: {output_dir}/")
    print("=" * 50)
    
    return {
        'table3': table3_results,
        'table4': table4_results,
        'table6': table6_results,
        'sensitivity': sensitivity_results,
        'validation': validation_results
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='UAV-CPS-Analyzer: Cyber-Physical Systems Analysis Software'
    )
    parser.add_argument('--iterations', '-n', type=int, default=10000,
                        help='Number of Monte Carlo iterations (default: 10000)')
    parser.add_argument('--output', '-o', type=str, default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--scenario', '-s', type=str, choices=['table3', 'table4', 'table6', 'all'],
                        default='all', help='Scenario to run')
    parser.add_argument('--fhss', action='store_true',
                        help='Run only FHSS analysis')
    parser.add_argument('--figures', action='store_true',
                        help='Generate only figures')
    parser.add_argument('--sensitivity', action='store_true',
                        help='Run only sensitivity analysis')
    parser.add_argument('--validation', action='store_true',
                        help='Run only validation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.fhss:
        print_header()
        run_table4_fhss_analysis()
    elif args.figures:
        print_header()
        generate_figures(args.output)
    elif args.sensitivity:
        print_header()
        run_sensitivity_analysis(args.iterations // 10)
    elif args.validation:
        print_header()
        run_validation()
    elif args.scenario == 'table3':
        print_header()
        run_table3_scenarios(args.iterations)
    elif args.scenario == 'table4':
        print_header()
        run_table4_fhss_analysis()
    elif args.scenario == 'table6':
        print_header()
        run_table6_detection()
    else:
        # Run full analysis
        run_full_analysis(args.iterations, args.output)


if __name__ == "__main__":
    main()
