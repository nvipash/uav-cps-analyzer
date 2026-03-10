#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: AI Bayesian Optimization
Optimizes jammer placement and sensor suite configuration using surrogate-assisted optimization.

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

try:
    from monte_carlo_engine import MonteCarloEngine, SimulationParams
    from ai_surrogate import SurrogatePredictor, train_and_evaluate
    from cps_analyzer import SensorFusion, ThreatType, RFSensor, RadarSensor, AcousticSensor, EOIRSensor
except ImportError:
    import sys
    sys.path.insert(0, '.')
    from monte_carlo_engine import MonteCarloEngine, SimulationParams
    from ai_surrogate import SurrogatePredictor, train_and_evaluate
    from cps_analyzer import SensorFusion, ThreatType, RFSensor, RadarSensor, AcousticSensor, EOIRSensor


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    optimal_params: Dict[str, float]
    objective_value: float
    n_evaluations: int
    converged: bool
    method: str


class JammerOptimizer:
    """
    Optimizes jammer parameters to maximize jamming effectiveness.
    Uses surrogate model for fast objective evaluation + differential evolution.
    """

    def __init__(self, predictor: SurrogatePredictor = None):
        self.predictor = predictor
        self.engine = MonteCarloEngine(n_processes=1)

    def optimize_power_distance(self, budget_watts: float = 100.0,
                                 target_altitude_m: float = 100.0,
                                 signal_distance_m: float = 5000.0
                                 ) -> OptimizationResult:
        """
        Find optimal jammer power and distance allocation given power budget.

        Args:
            budget_watts: Total available power in watts
            target_altitude_m: Target UAV altitude
            signal_distance_m: Operator-to-UAV distance

        Returns:
            OptimizationResult with optimal power/distance/gain
        """
        def objective(x):
            power_dbm, distance, antenna_gain = x
            # Power constraint: actual watts must be <= budget
            power_w = 10 ** ((power_dbm - 30) / 10)
            if power_w > budget_watts:
                return 100.0  # penalty

            if self.predictor:
                result = self.predictor.predict(
                    power_dbm, distance, antenna_gain,
                    signal_distance_m, target_altitude_m, 3.0
                )
                return -result.mean_js_db  # maximize J/S
            else:
                params = SimulationParams(
                    jammer_power_dbm=power_dbm, jammer_distance_m=distance,
                    jammer_antenna_gain_dbi=antenna_gain,
                    signal_distance_m=signal_distance_m,
                    altitude_m=target_altitude_m, fhss_enabled=False
                )
                r = self.engine.run_simulation(params, 1000, parallel=False, random_seed=42)
                return -r.mean_js_db

        max_power_dbm = 10 * np.log10(budget_watts * 1000)
        bounds = [
            (20.0, max_power_dbm),  # power dBm
            (50.0, 3000.0),          # distance m
            (2.0, 18.0),             # antenna gain dBi
        ]

        result = differential_evolution(objective, bounds, seed=42,
                                         maxiter=100, tol=0.01, popsize=10)

        return OptimizationResult(
            optimal_params={
                'power_dbm': result.x[0],
                'power_watts': 10 ** ((result.x[0] - 30) / 10),
                'distance_m': result.x[1],
                'antenna_gain_dbi': result.x[2],
                'predicted_js_db': -result.fun,
            },
            objective_value=-result.fun,
            n_evaluations=result.nfev,
            converged=result.success,
            method='differential_evolution'
        )

    def generate_coverage_map(self, power_dbm: float = 40.0,
                               altitude_m: float = 100.0,
                               grid_size: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 2D coverage heatmap showing J/S ratio over area.

        Returns:
            (X_grid, Y_grid, JS_grid) for plotting
        """
        x_range = np.linspace(-3000, 3000, grid_size)
        y_range = np.linspace(-3000, 3000, grid_size)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        JS_grid = np.zeros_like(X_grid)

        for i in range(grid_size):
            for j in range(grid_size):
                dist = np.sqrt(X_grid[i, j]**2 + Y_grid[i, j]**2)
                dist = max(dist, 10.0)  # minimum distance

                if self.predictor:
                    result = self.predictor.predict(
                        power_dbm, dist, 6.0, 5000.0, altitude_m, 3.0
                    )
                    JS_grid[i, j] = result.mean_js_db
                else:
                    params = SimulationParams(
                        jammer_power_dbm=power_dbm, jammer_distance_m=dist,
                        signal_distance_m=5000, altitude_m=altitude_m, fhss_enabled=False
                    )
                    r = self.engine.run_simulation(params, 500, parallel=False, random_seed=42)
                    JS_grid[i, j] = r.mean_js_db

        return X_grid, Y_grid, JS_grid


class SensorSuiteOptimizer:
    """Optimizes sensor suite configuration for maximum detection coverage."""

    def optimize_for_budget(self, budget_usd: float = 50000.0,
                             target_range_m: float = 1000.0,
                             n_trials: int = 300) -> Dict:
        """
        Find best sensor combination within budget.

        Args:
            budget_usd: Maximum budget
            target_range_m: Expected threat range
            n_trials: MC trials for detection rate estimation

        Returns:
            Optimal configuration dict
        """
        from itertools import combinations
        from cps_analyzer import SENSOR_SPECS, SensorType

        sensor_classes = {
            'RF': (RFSensor, 15000),
            'Radar': (RadarSensor, 50000),
            'Acoustic': (AcousticSensor, 5000),
            'EO/IR': (EOIRSensor, 25000),
        }

        best_config = None
        best_score = -1

        # Enumerate all combinations within budget
        names = list(sensor_classes.keys())
        for r in range(1, len(names) + 1):
            for combo in combinations(names, r):
                cost = sum(sensor_classes[n][1] for n in combo)
                if cost > budget_usd:
                    continue

                sensors = [sensor_classes[n][0]() for n in combo]
                fusion = SensorFusion(sensors=sensors)

                det_rf = fusion.get_detection_rate(ThreatType.RF_CONTROLLED, target_range_m, n_trials)
                det_fiber = fusion.get_detection_rate(ThreatType.FIBER_OPTIC, target_range_m, n_trials)

                # Weighted score: RF threats more common (60/40 split)
                score = 0.6 * det_rf + 0.4 * det_fiber

                if score > best_score:
                    best_score = score
                    best_config = {
                        'sensors': list(combo),
                        'cost_usd': cost,
                        'detection_rf': det_rf,
                        'detection_fiber': det_fiber,
                        'score': score,
                        'budget_utilization': cost / budget_usd * 100,
                    }

        return best_config


def run_optimization(predictor: SurrogatePredictor = None) -> Dict:
    """Run all optimizations and return results."""
    results = {}

    # Jammer optimization
    opt = JammerOptimizer(predictor)
    for budget in [10, 100, 500]:
        r = opt.optimize_power_distance(budget_watts=budget)
        results[f'jammer_{budget}W'] = r

    # Sensor optimization
    sopt = SensorSuiteOptimizer()
    for budget in [20000, 50000, 100000]:
        r = sopt.optimize_for_budget(budget_usd=budget, n_trials=200)
        results[f'sensor_${budget//1000}K'] = r

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("AI Bayesian Optimization — Test")
    print("=" * 60)
    results = run_optimization()
    for name, r in results.items():
        if isinstance(r, OptimizationResult):
            print(f"\n{name}: J/S={r.objective_value:.1f} dB ({r.n_evaluations} evals)")
            for k, v in r.optimal_params.items():
                print(f"  {k}: {v:.1f}")
        else:
            print(f"\n{name}: score={r['score']:.3f}, cost=${r['cost_usd']:,}")
            print(f"  Sensors: {r['sensors']}")
