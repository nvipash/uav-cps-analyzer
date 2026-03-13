#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: AI Bayesian Optimization
Optimizes jammer placement and sensor suite configuration using Bayesian Optimization
with a Gaussian Process surrogate and Expected Improvement acquisition function.

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
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


class BayesianOptimizer:
    """
    Bayesian Optimization with GP surrogate and Expected Improvement acquisition.

    Iteratively builds a probabilistic model of the objective and queries the
    point that maximizes Expected Improvement (EI) over the current best
    observation.  EI balances exploration (high GP uncertainty) and exploitation
    (high GP mean), making it far more sample-efficient than random or grid
    search and provably more principled than differential evolution.
    """

    def __init__(self, bounds, n_initial: int = 8, n_iterations: int = 25,
                 random_state: int = 42):
        self.bounds = np.array(bounds, dtype=float)
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.rng = np.random.RandomState(random_state)
        self._gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
        )

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return (X - lo) / (hi - lo)

    def _denormalize(self, X_norm: np.ndarray) -> np.ndarray:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return X_norm * (hi - lo) + lo

    def _expected_improvement(self, X_norm: np.ndarray, y_best: float,
                               xi: float = 0.01) -> np.ndarray:
        """EI = E[max(f(x) - y_best, 0)] under the GP posterior."""
        mu, sigma = self._gp.predict(X_norm, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        Z = (mu - y_best - xi) / sigma
        return (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)

    def _maximize_ei(self, y_best: float) -> np.ndarray:
        """Find next query point by maximising EI with random L-BFGS-B restarts."""
        d = len(self.bounds)
        best_ei, best_x = -np.inf, None
        for _ in range(20):
            x0 = self.rng.uniform(0.0, 1.0, size=d)
            res = minimize(
                lambda x: -float(self._expected_improvement(x.reshape(1, -1), y_best)),
                x0, method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * d,
            )
            if -res.fun > best_ei:
                best_ei = -res.fun
                best_x = res.x
        return self._denormalize(best_x)

    def optimize(self, objective) -> Tuple[np.ndarray, float, int]:
        """
        Maximise objective(x) -> float using Bayesian Optimisation.
        Returns (best_x, best_y, n_evaluations).
        """
        d = len(self.bounds)
        sampler = LatinHypercube(d=d, seed=int(self.rng.randint(0, 9999)))
        X_obs = self._denormalize(sampler.random(n=self.n_initial))
        Y_obs = np.array([objective(x) for x in X_obs])

        for _ in range(self.n_iterations):
            self._gp.fit(self._normalize(X_obs), Y_obs)
            x_next = self._maximize_ei(float(np.max(Y_obs)))
            y_next = objective(x_next)
            X_obs = np.vstack([X_obs, x_next])
            Y_obs = np.append(Y_obs, y_next)

        best_idx = int(np.argmax(Y_obs))
        return X_obs[best_idx], float(Y_obs[best_idx]), len(Y_obs)


class JammerOptimizer:
    """
    Optimises jammer parameters to maximise jamming effectiveness.
    Uses Bayesian Optimisation (GP surrogate + EI acquisition) for
    sample-efficient search, with optional surrogate model for fast
    objective evaluation.
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
        Uses Bayesian Optimisation: GP surrogate + Expected Improvement acquisition.
        """
        def objective(x):
            power_dbm, distance, antenna_gain = x
            power_w = 10 ** ((power_dbm - 30) / 10)
            if power_w > budget_watts:
                return -50.0  # infeasible: return low value to guide BO away

            if self.predictor:
                result = self.predictor.predict(
                    power_dbm, distance, antenna_gain,
                    signal_distance_m, target_altitude_m, 3.0
                )
                return result.mean_js_db
            else:
                params = SimulationParams(
                    jammer_power_dbm=power_dbm, jammer_distance_m=distance,
                    jammer_antenna_gain_dbi=antenna_gain,
                    signal_distance_m=signal_distance_m,
                    altitude_m=target_altitude_m, fhss_enabled=False
                )
                r = self.engine.run_simulation(params, 500, parallel=False, random_seed=42)
                return r.mean_js_db

        max_power_dbm = 10 * np.log10(budget_watts * 1000)
        bounds = [
            (20.0, max_power_dbm),   # power dBm
            (50.0, 3000.0),           # distance m
            (2.0, 18.0),              # antenna gain dBi
        ]

        bo = BayesianOptimizer(bounds=bounds, n_initial=8, n_iterations=25)
        best_x, best_y, n_evals = bo.optimize(objective)

        return OptimizationResult(
            optimal_params={
                'power_dbm': float(best_x[0]),
                'power_watts': 10 ** ((float(best_x[0]) - 30) / 10),
                'distance_m': float(best_x[1]),
                'antenna_gain_dbi': float(best_x[2]),
                'predicted_js_db': best_y,
            },
            objective_value=best_y,
            n_evaluations=n_evals,
            converged=True,
            method='bayesian_optimization',
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
                dist = max(dist, 10.0)

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
    """Optimises sensor suite configuration for maximum detection coverage."""

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
    """Run all optimisations and return results."""
    results = {}

    opt = JammerOptimizer(predictor)
    for budget in [10, 100, 500]:
        r = opt.optimize_power_distance(budget_watts=budget)
        results[f'jammer_{budget}W'] = r

    sopt = SensorSuiteOptimizer()
    for budget in [20000, 50000, 100000]:
        r = sopt.optimize_for_budget(budget_usd=budget, n_trials=200)
        results[f'sensor_${budget//1000}K'] = r

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("AI Bayesian Optimisation — Test")
    print("=" * 60)
    results = run_optimization()
    for name, r in results.items():
        if isinstance(r, OptimizationResult):
            print(f"\n{name}: J/S={r.objective_value:.1f} dB ({r.n_evaluations} evals, {r.method})")
            for k, v in r.optimal_params.items():
                print(f"  {k}: {v:.2f}")
        else:
            print(f"\n{name}: score={r['score']:.3f}, cost=${r['cost_usd']:,}")
            print(f"  Sensors: {r['sensors']}")
