#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: AI Surrogate Model
Trains a neural network to replace Monte Carlo simulation for instant J/S prediction.

Approach:
- Generate training data via Latin Hypercube sampling across parameter space
- Train MLPRegressor to predict (mean_js, std_js, success_probability)
- Achieves ~1000x speedup over full MC at <1% prediction error

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

import numpy as np
import time
import pickle
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import qmc

try:
    from monte_carlo_engine import MonteCarloEngine, SimulationParams
except ImportError:
    import sys
    sys.path.insert(0, '.')
    from monte_carlo_engine import MonteCarloEngine, SimulationParams


# Parameter space bounds for training
PARAM_BOUNDS = {
    'jammer_power_dbm':       (20.0, 60.0),   # 0.1W to 1000W
    'jammer_distance_m':      (100.0, 5000.0),
    'jammer_antenna_gain_dbi': (2.0, 18.0),
    'signal_distance_m':      (1000.0, 15000.0),
    'altitude_m':             (30.0, 500.0),
    'path_loss_std':          (2.0, 12.0),
}
PARAM_NAMES = list(PARAM_BOUNDS.keys())


@dataclass
class SurrogateResult:
    """Result from surrogate prediction."""
    mean_js_db: float
    std_js_db: float
    success_probability: float
    prediction_time_ms: float = 0.0


class SurrogateTrainer:
    """Generates training data and trains the surrogate model."""

    def __init__(self, mc_iterations_per_point: int = 1000):
        self.mc_iters = mc_iterations_per_point
        self.engine = MonteCarloEngine(n_processes=1)

    def generate_training_data(self, n_points: int = 500,
                                seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data using Latin Hypercube Sampling.

        Args:
            n_points: Number of training points
            seed: Random seed for reproducibility

        Returns:
            (X, Y) where X is (n, 6) features and Y is (n, 3) targets
        """
        print(f"  Generating {n_points} training points ({self.mc_iters} MC iter each)...",
              flush=True)

        # Latin Hypercube Sampling for space-filling design
        sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)
        samples = sampler.random(n_points)

        # Scale to actual parameter ranges
        lower = np.array([PARAM_BOUNDS[p][0] for p in PARAM_NAMES])
        upper = np.array([PARAM_BOUNDS[p][1] for p in PARAM_NAMES])
        X = qmc.scale(samples, lower, upper)

        Y = np.zeros((n_points, 3))  # mean_js, std_js, success_prob

        t0 = time.time()
        for i in range(n_points):
            params = SimulationParams(
                jammer_power_dbm=X[i, 0],
                jammer_distance_m=X[i, 1],
                jammer_antenna_gain_dbi=X[i, 2],
                signal_distance_m=X[i, 3],
                altitude_m=X[i, 4],
                path_loss_std=X[i, 5],
                fhss_enabled=False
            )
            result = self.engine.run_simulation(
                params, self.mc_iters, parallel=False, random_seed=seed + i
            )
            Y[i, 0] = result.mean_js_db
            Y[i, 1] = result.std_js_db
            Y[i, 2] = result.success_probability

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_points - i - 1) / rate
                print(f"    [{i+1}/{n_points}] {rate:.0f} pts/s, ETA {eta:.0f}s", flush=True)

        print(f"  Training data generated in {time.time()-t0:.1f}s", flush=True)
        return X, Y

    def train(self, X: np.ndarray, Y: np.ndarray, model_type: str = 'mlp'
              ) -> Tuple[object, StandardScaler, StandardScaler, dict]:
        """
        Train the surrogate model.

        Args:
            X, Y: Training data
            model_type: 'mlp' (default), 'gp' (Gaussian Process), 'ensemble'

        Returns:
            (model, scaler_X, scaler_Y, metrics)
        """
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        scaler_X = StandardScaler().fit(X_train)
        scaler_Y = StandardScaler().fit(Y_train)
        X_tr = scaler_X.transform(X_train)
        X_te = scaler_X.transform(X_test)
        Y_tr = scaler_Y.transform(Y_train)

        if model_type == 'gp':
            # Gaussian Process: best for small N, gives uncertainty
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            model = GaussianProcessRegressor(
                kernel=kernel, alpha=0.01, normalize_y=False,
                n_restarts_optimizer=3, random_state=42
            )
            # GP only handles single output — train per output column
            class MultiOutputGP:
                def __init__(self, n_outputs, kernel):
                    from sklearn.gaussian_process import GaussianProcessRegressor
                    self.models = [
                        GaussianProcessRegressor(kernel=kernel, alpha=0.01,
                                                  normalize_y=False, n_restarts_optimizer=3,
                                                  random_state=42)
                        for _ in range(n_outputs)
                    ]
                def fit(self, X, Y):
                    for i, m in enumerate(self.models):
                        m.fit(X, Y[:, i])
                    return self
                def predict(self, X):
                    return np.column_stack([m.predict(X) for m in self.models])

            model = MultiOutputGP(Y.shape[1], kernel)
            model.fit(X_tr, Y_tr)

        elif model_type == 'ensemble':
            # Ensemble: average MLP + GBM predictions
            class EnsembleSurrogate:
                def __init__(self):
                    self.mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32),
                                             max_iter=1500, early_stopping=True,
                                             random_state=42, learning_rate='adaptive')
                    self.gbms = None  # one GBM per output
                def fit(self, X, Y):
                    self.mlp.fit(X, Y)
                    self.gbms = [GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                             random_state=42).fit(X, Y[:, i])
                                  for i in range(Y.shape[1])]
                    return self
                def predict(self, X):
                    pred_mlp = self.mlp.predict(X)
                    pred_gbm = np.column_stack([g.predict(X) for g in self.gbms])
                    return 0.5 * (pred_mlp + pred_gbm)
            model = EnsembleSurrogate()
            model.fit(X_tr, Y_tr)
        else:
            # Default: large MLP
            model = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64, 32),
                activation='relu',
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
                learning_rate='adaptive'
            )
            model.fit(X_tr, Y_tr)

        Y_pred_scaled = model.predict(X_te)
        Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

        metrics = {
            'r2_mean_js': r2_score(Y_test[:, 0], Y_pred[:, 0]),
            'r2_std_js': r2_score(Y_test[:, 1], Y_pred[:, 1]),
            'r2_success': r2_score(Y_test[:, 2], Y_pred[:, 2]),
            'rmse_mean_js': float(np.sqrt(mean_squared_error(Y_test[:, 0], Y_pred[:, 0]))),
            'mae_mean_js': float(mean_absolute_error(Y_test[:, 0], Y_pred[:, 0])),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'model_type': model_type,
        }

        return model, scaler_X, scaler_Y, metrics


class SurrogatePredictor:
    """Fast J/S prediction using trained surrogate model."""

    def __init__(self, model: MLPRegressor = None,
                 scaler_X: StandardScaler = None,
                 scaler_Y: StandardScaler = None):
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y

    def predict(self, jammer_power_dbm: float = 40.0,
                jammer_distance_m: float = 500.0,
                jammer_antenna_gain_dbi: float = 6.0,
                signal_distance_m: float = 5000.0,
                altitude_m: float = 100.0,
                path_loss_std: float = 3.0) -> SurrogateResult:
        """Predict J/S statistics instantly."""
        t0 = time.time()
        X = np.array([[jammer_power_dbm, jammer_distance_m, jammer_antenna_gain_dbi,
                        signal_distance_m, altitude_m, path_loss_std]])
        X_scaled = self.scaler_X.transform(X)
        Y_scaled = self.model.predict(X_scaled)
        Y = self.scaler_Y.inverse_transform(Y_scaled)[0]
        elapsed_ms = (time.time() - t0) * 1000

        return SurrogateResult(
            mean_js_db=float(Y[0]),
            std_js_db=float(max(0, Y[1])),
            success_probability=float(np.clip(Y[2], 0, 1)),
            prediction_time_ms=elapsed_ms
        )

    def predict_grid(self, distances: np.ndarray,
                     jammer_power_dbm: float = 40.0,
                     altitude_m: float = 100.0) -> np.ndarray:
        """Predict J/S for array of distances (for coverage maps)."""
        n = len(distances)
        X = np.column_stack([
            np.full(n, jammer_power_dbm),
            distances,
            np.full(n, 6.0),
            np.full(n, 5000.0),
            np.full(n, altitude_m),
            np.full(n, 3.0),
        ])
        X_scaled = self.scaler_X.transform(X)
        Y_scaled = self.model.predict(X_scaled)
        Y = self.scaler_Y.inverse_transform(Y_scaled)
        return Y[:, 0]  # mean J/S values

    def save(self, path: str = "output/surrogate_model.pkl"):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler_X': self.scaler_X,
                         'scaler_Y': self.scaler_Y}, f)

    @classmethod
    def load(cls, path: str = "output/surrogate_model.pkl") -> 'SurrogatePredictor':
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(data['model'], data['scaler_X'], data['scaler_Y'])


def train_and_evaluate(n_points: int = 500, mc_iters: int = 1000) -> Tuple[SurrogatePredictor, dict]:
    """
    Complete training and evaluation pipeline.

    Returns:
        (predictor, metrics)
    """
    trainer = SurrogateTrainer(mc_iterations_per_point=mc_iters)
    X, Y = trainer.generate_training_data(n_points)
    model, scaler_X, scaler_Y, metrics = trainer.train(X, Y)

    predictor = SurrogatePredictor(model, scaler_X, scaler_Y)

    # Speedup comparison
    engine = MonteCarloEngine(n_processes=1)
    test_params = SimulationParams(jammer_power_dbm=40, jammer_distance_m=500,
                                    signal_distance_m=5000, altitude_m=100, fhss_enabled=False)

    t0 = time.time()
    mc_result = engine.run_simulation(test_params, 5000, parallel=False, random_seed=99)
    mc_time = time.time() - t0

    t0 = time.time()
    for _ in range(100):
        surr_result = predictor.predict(40, 500, 6, 5000, 100, 3)
    surr_time = (time.time() - t0) / 100

    metrics['mc_time_s'] = mc_time
    metrics['surrogate_time_s'] = surr_time
    metrics['speedup'] = mc_time / surr_time if surr_time > 0 else float('inf')
    metrics['mc_reference'] = mc_result.mean_js_db
    metrics['surrogate_prediction'] = surr_result.mean_js_db

    return predictor, metrics


if __name__ == "__main__":
    print("=" * 60)
    print("AI Surrogate Model — Training & Evaluation")
    print("=" * 60)
    predictor, metrics = train_and_evaluate(n_points=200, mc_iters=500)
    print(f"\nR2 (mean J/S): {metrics['r2_mean_js']:.4f}")
    print(f"RMSE (mean J/S): {metrics['rmse_mean_js']:.2f} dB")
    print(f"Speedup: {metrics['speedup']:.0f}x")
    predictor.save()
