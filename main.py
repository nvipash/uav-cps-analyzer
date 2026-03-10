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
    python main.py --sobol            # Run Sobol sensitivity analysis
    python main.py --validate         # Run formal validation
"""

import argparse
import sys
import os
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from propagation_models import AltitudeDependentModel, AlHouraniA2GModel, calculate_js_ratio
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
from config import Config, DRONE_DATABASE, JAMMER_DATABASE, DroneType, ENVIRONMENT_PRESETS
from sensitivity import sobol_analysis, morris_screening, print_sobol_results, print_morris_results
from validation import ValidationEngine, InternalConsistencyChecker
from reporting import (
    table3_latex, table4_latex, table7_latex, sobol_latex,
    generate_summary_report
)


def print_header():
    """Print application header."""
    print("=" * 70)
    print("  UAV-CPS-Analyzer v1.0")
    print("  Cyber-Physical Systems Analysis for UAV Communication Reliability")
    print("=" * 70)
    print("  Authors: Novitskyi P.S., Stepaniak M.V.")
    print("  Lviv Polytechnic National University, 2025")
    print("=" * 70)


def run_table3_scenarios(n_iterations: int = 1000000) -> dict:
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

    results = {}

    scenarios = [
        ("Portable 10W, 500m", 10, 500, 5000, 100),
        ("Portable 10W, 1000m", 10, 1000, 5000, 100),
        ("Mobile 100W, 2000m", 100, 2000, 8000, 200),
        ("Stationary 500W, 3000m", 500, 3000, 10000, 300)
    ]

    # Print sample size justification
    req_n = MonteCarloEngine.required_sample_size(
        desired_precision_db=0.1, confidence_level=0.95, estimated_std=5.0
    )
    print(f"\nTheoretical minimum N for 0.1 dB precision: {req_n:,}")
    print(f"Using N = {n_iterations:,}")

    print(f"\n{'Scenario':<28} {'J/S (dB)':<12} {'95% CI':<22} {'P(jam)':<10} {'Conv.'}")
    print("-" * 80)

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
        conv_str = "Yes" if result.converged else "No"
        print(f"{name:<28} {result.mean_js_db:<12.1f} {ci_str:<22} "
              f"{result.success_probability*100:<10.0f}% {conv_str}")

        # Print bootstrap CI uncertainty
        if result.ci_lower_uncertainty != (0.0, 0.0):
            print(f"  Bootstrap CI uncertainty: lower [{result.ci_lower_uncertainty[0]:.2f}, "
                  f"{result.ci_lower_uncertainty[1]:.2f}], "
                  f"upper [{result.ci_upper_uncertainty[0]:.2f}, "
                  f"{result.ci_upper_uncertainty[1]:.2f}]")

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
        print(f"{strategy:<18} {static:>6.0f}%      {fhss:>6.0f}%      x{mult:.0f}")

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
    Run OAT sensitivity analysis for tornado diagram (backward compatible).

    Args:
        n_iterations: Iterations per parameter

    Returns:
        Dictionary of sensitivities
    """
    print("\n" + "=" * 50)
    print("Sensitivity Analysis (OAT / Tornado Diagram)")
    print("=" * 50)

    engine = MonteCarloEngine()
    params = SimulationParams(
        jammer_power_dbm=40,
        jammer_distance_m=500,
        signal_distance_m=5000,
        altitude_m=100,
        fhss_enabled=False
    )

    sensitivities = engine.sensitivity_analysis(params, n_iterations)

    print(f"\n{'Parameter':<20} {'Sensitivity (+/-dB)'}")
    print("-" * 35)

    sorted_sens = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
    for param, sens in sorted_sens:
        print(f"{param:<20} +/-{sens:.1f}")

    return dict(sorted_sens)


def run_sobol_analysis(n_samples: int = 64, n_mc: int = 500) -> dict:
    """
    Run global sensitivity analysis using Sobol indices and Morris screening.

    Args:
        n_samples: Sobol base sample size
        n_mc: MC iterations per model evaluation

    Returns:
        Dictionary with sobol and morris results
    """
    print("\n" + "=" * 50)
    print("Global Sensitivity Analysis")
    print("=" * 50)

    # Morris screening (fast)
    print("\n--- Morris Elementary Effects Screening ---")
    morris_result = morris_screening(n_trajectories=10, n_mc_per_eval=n_mc)
    print_morris_results(morris_result)

    # Sobol indices (comprehensive)
    print("\n--- Sobol Variance-Based Analysis ---")
    sobol_result = sobol_analysis(n_samples=n_samples, n_mc_per_eval=n_mc)
    print_sobol_results(sobol_result)

    return {'sobol': sobol_result, 'morris': morris_result}


def run_validation(n_iterations: int = 10000) -> dict:
    """
    Run formal model validation against experimental data (Table 7).
    Uses ASME V&V 20 framework with actual model runs.

    Args:
        n_iterations: MC iterations per validation case

    Returns:
        Dictionary of validation results
    """
    print("\n" + "=" * 50)
    print("Table 7: Model Validation (ASME V&V 20)")
    print("=" * 50)

    # Internal consistency checks first
    print("\n--- Internal Consistency Checks ---")
    checker = InternalConsistencyChecker()
    checker.check_all()
    checker.print_results()

    # Formal validation
    print("\n--- Formal Validation Against References ---")
    val_engine = ValidationEngine(n_iterations=n_iterations)
    report = val_engine.run_validation()
    val_engine.print_report(report)

    return {'report': report, 'consistency': checker.results}


def run_environment_comparison(n_iterations: int = 5000) -> dict:
    """
    Run multi-environment comparative study.

    Args:
        n_iterations: MC iterations per environment

    Returns:
        Dictionary of results by environment
    """
    print("\n" + "=" * 50)
    print("Multi-Environment Comparative Study")
    print("=" * 50)

    results = {}

    print(f"\n{'Environment':<15} {'J/S (dB)':<12} {'95% CI':<22} {'P(jam)':<10} {'Shadow std'}")
    print("-" * 75)

    for env_name, preset in ENVIRONMENT_PRESETS.items():
        params = SimulationParams(
            jammer_power_dbm=40.0,
            jammer_distance_m=500.0,
            signal_distance_m=5000.0,
            altitude_m=100.0,
            fhss_enabled=False,
            path_loss_std=preset['shadow_fading_std_db']
        )

        engine = MonteCarloEngine()
        result = engine.run_simulation(params, n_iterations)

        ci_str = f"[{result.ci_95_lower:.1f}, {result.ci_95_upper:.1f}]"
        print(f"{env_name:<15} {result.mean_js_db:<12.1f} {ci_str:<22} "
              f"{result.success_probability*100:<10.0f}% {preset['shadow_fading_std_db']:.1f} dB")

        results[env_name] = result

    return results


def generate_figures(output_dir: str = "output",
                     mc_result: MCResult = None,
                     sensitivities: dict = None):
    """
    Generate all figures for the paper.

    Args:
        output_dir: Directory to save figures
        mc_result: Pre-computed MC result (runs simulation if None)
        sensitivities: Pre-computed sensitivities (runs analysis if None)
    """
    print("\n" + "=" * 50)
    print("Generating Publication Figures")
    print("=" * 50)

    viz = UAVCPSVisualizer(output_dir=output_dir)

    if mc_result is None:
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
        mc_result = engine.run_simulation(params, n_iterations=10000)

    # Generate figures
    figures = viz.generate_all_figures(mc_result, sensitivities)

    print(f"\nFigures saved to: {output_dir}/")
    for name in figures:
        print(f"  - {name}.png")
        print(f"  - {name}.pdf")


def generate_latex_tables(scenario_results: dict = None,
                          fhss_results: dict = None,
                          validation_results: dict = None,
                          sobol_results: dict = None,
                          output_dir: str = "output"):
    """Generate LaTeX tables and save to files."""
    print("\n" + "=" * 50)
    print("Generating LaTeX Tables")
    print("=" * 50)

    os.makedirs(output_dir, exist_ok=True)

    if scenario_results:
        latex = table3_latex(scenario_results)
        path = os.path.join(output_dir, "table3.tex")
        with open(path, 'w') as f:
            f.write(latex)
        print(f"  Table 3 saved to: {path}")

    if fhss_results:
        latex = table4_latex(fhss_results)
        path = os.path.join(output_dir, "table4.tex")
        with open(path, 'w') as f:
            f.write(latex)
        print(f"  Table 4 saved to: {path}")

    if validation_results and 'report' in validation_results:
        latex = table7_latex(validation_results['report'])
        path = os.path.join(output_dir, "table7.tex")
        with open(path, 'w') as f:
            f.write(latex)
        print(f"  Table 7 saved to: {path}")

    if sobol_results and 'sobol' in sobol_results:
        latex = sobol_latex(sobol_results['sobol'])
        path = os.path.join(output_dir, "table_sobol.tex")
        with open(path, 'w') as f:
            f.write(latex)
        print(f"  Sobol table saved to: {path}")


def run_full_analysis(n_iterations: int = 1000000, output_dir: str = "output"):
    """
    Run complete analysis pipeline.

    Args:
        n_iterations: Monte Carlo iterations
        output_dir: Output directory
    """
    start_time = time.time()

    print_header()

    # Core analyses
    table3_results = run_table3_scenarios(n_iterations)
    table4_results = run_table4_fhss_analysis()
    table6_results = run_table6_detection()
    sensitivity_results = run_sensitivity_analysis(min(1000, n_iterations // 10))
    validation_results = run_validation(min(10000, n_iterations))

    # Multi-environment study
    env_results = run_environment_comparison(min(5000, n_iterations // 10))

    # Get a representative MC result for figures
    first_result = list(table3_results.values())[0] if table3_results else None

    # Generate figures
    generate_figures(output_dir, mc_result=first_result,
                     sensitivities=sensitivity_results)

    # Generate LaTeX tables
    generate_latex_tables(
        scenario_results=table3_results,
        fhss_results=table4_results,
        validation_results=validation_results,
        output_dir=output_dir
    )

    # Generate summary report
    report = generate_summary_report(
        scenario_results=table3_results,
        fhss_results=table4_results,
        validation_report=validation_results.get('report'),
        output_path=os.path.join(output_dir, "summary_report.md")
    )

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print("Analysis Complete")
    print("=" * 50)
    print(f"  Total time: {elapsed:.1f} seconds")
    print(f"  Monte Carlo iterations: {n_iterations:,}")
    print(f"  Output directory: {output_dir}/")
    print("=" * 50)

    return {
        'table3': table3_results,
        'table4': table4_results,
        'table6': table6_results,
        'sensitivity': sensitivity_results,
        'validation': validation_results,
        'environment': env_results,
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
    parser.add_argument('--scenario', '-s', type=str,
                        choices=['table3', 'table4', 'table6', 'all'],
                        default='all', help='Scenario to run')
    parser.add_argument('--fhss', action='store_true',
                        help='Run only FHSS analysis')
    parser.add_argument('--figures', action='store_true',
                        help='Generate only figures')
    parser.add_argument('--sensitivity', action='store_true',
                        help='Run only OAT sensitivity analysis')
    parser.add_argument('--sobol', action='store_true',
                        help='Run Sobol global sensitivity analysis')
    parser.add_argument('--validate', action='store_true',
                        help='Run formal validation (ASME V&V 20)')
    parser.add_argument('--environment', action='store_true',
                        help='Run multi-environment comparison')
    parser.add_argument('--report', action='store_true',
                        help='Generate summary report only')

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
    elif args.sobol:
        print_header()
        run_sobol_analysis(n_samples=64, n_mc=min(500, args.iterations))
    elif args.validate:
        print_header()
        run_validation(args.iterations)
    elif args.environment:
        print_header()
        run_environment_comparison(args.iterations)
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
