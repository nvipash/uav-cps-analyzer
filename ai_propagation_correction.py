#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: ML Propagation Model Correction
Learns systematic bias in propagation models from reference data and applies corrections.

Approach:
- Collect (model_prediction, reference_value) pairs across diverse scenarios
- Train GradientBoostingRegressor on features -> residual (model - reference)
- Apply learned correction: corrected = model_prediction - predicted_residual

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    from monte_carlo_engine import MonteCarloEngine, SimulationParams
    from propagation_models import AltitudeDependentModel, AlHouraniA2GModel
except ImportError:
    import sys
    sys.path.insert(0, '.')
    from monte_carlo_engine import MonteCarloEngine, SimulationParams
    from propagation_models import AltitudeDependentModel, AlHouraniA2GModel


@dataclass
class CorrectionResult:
    """Result of applying ML correction."""
    original_prediction: float
    correction_db: float
    corrected_prediction: float
    reference_value: float = 0.0
    original_error: float = 0.0
    corrected_error: float = 0.0


class PropagationCorrector:
    """
    ML-based correction for propagation model systematic errors.
    Wraps any propagation model and adds a learned correction term.
    """

    def __init__(self):
        self.model = None
        self.engine = MonteCarloEngine(n_processes=1)
        self.training_data = None

    def _generate_reference_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate training data: model predictions vs reference values.
        Uses synthetic reference points derived from literature empirical formulas.

        Returns:
            (features, model_predictions, reference_values)
        """
        # Scenarios with known approximate J/S from literature
        # Format: (power_W, j_dist, s_dist, alt, gain, environment, ref_js)
        reference_scenarios = [
            # Close range portable (Adamy 2015)
            (10,   200, 3000,  50,  6, 'urban', 55.0),
            (10,   500, 5000, 100,  6, 'urban', 45.0),
            (10,  1000, 5000, 100,  6, 'urban', 35.0),
            (10,  1500, 5000, 150,  6, 'urban', 28.0),
            (10,  2000, 5000, 200,  6, 'urban', 22.0),
            # Mobile systems (Poisel 2011)
            (100,  500, 5000, 100, 10, 'urban', 60.0),
            (100, 1000, 8000, 150, 10, 'urban', 48.0),
            (100, 2000, 8000, 200, 10, 'urban', 38.0),
            (100, 3000, 8000, 200, 10, 'urban', 30.0),
            (100, 4000, 10000, 250, 10, 'urban', 25.0),
            # High-power stationary (Skolnik 2008)
            (500, 1000, 10000, 200, 15, 'urban', 58.0),
            (500, 2000, 10000, 300, 15, 'urban', 48.0),
            (500, 3000, 10000, 300, 15, 'urban', 42.0),
            (500, 5000, 10000, 400, 15, 'urban', 32.0),
            # Suburban scenarios (lower path loss)
            (10,   500, 5000, 100,  6, 'suburban', 50.0),
            (100, 2000, 8000, 200, 10, 'suburban', 42.0),
            (500, 3000, 10000, 300, 15, 'suburban', 48.0),
            # Rural / open field
            (10,   500, 5000, 100,  6, 'rural', 52.0),
            (100, 2000, 8000, 200, 10, 'rural', 45.0),
            (500, 3000, 10000, 300, 15, 'rural', 50.0),
        ]

        n = len(reference_scenarios)
        features = np.zeros((n, 7))  # power, j_dist, s_dist, alt, gain, env_code, los_prob
        model_preds = np.zeros(n)
        ref_values = np.zeros(n)

        env_map = {'urban': 0, 'suburban': 1, 'rural': 2}

        for i, (pw, jd, sd, alt, gain, env, ref_js) in enumerate(reference_scenarios):
            power_dbm = 10 * np.log10(pw * 1000)

            # Run model prediction
            params = SimulationParams(
                jammer_power_dbm=power_dbm, jammer_distance_m=jd,
                jammer_antenna_gain_dbi=gain, signal_distance_m=sd,
                altitude_m=alt, fhss_enabled=False,
                propagation_model='al_hourani', environment=env
            )
            result = self.engine.run_simulation(params, 2000, parallel=False, random_seed=42 + i)

            # LOS probability for this geometry
            a2g = AlHouraniA2GModel(env)
            los_prob = a2g.los_probability(alt, jd)

            features[i] = [power_dbm, jd, sd, alt, gain, env_map.get(env, 0), los_prob]
            model_preds[i] = result.mean_js_db
            ref_values[i] = ref_js

        return features, model_preds, ref_values

    def train(self) -> Dict:
        """
        Train the correction model.

        Returns:
            Training metrics dict
        """
        print("  Generating reference data...", flush=True)
        features, model_preds, ref_values = self._generate_reference_data()

        # Target: residual (model - reference)
        residuals = model_preds - ref_values

        self.training_data = {
            'features': features,
            'model_preds': model_preds,
            'ref_values': ref_values,
            'residuals': residuals,
        }

        # Train Gradient Boosting on residuals
        self.model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42, subsample=0.8
        )
        self.model.fit(features, residuals)

        # Cross-validation
        cv_scores = cross_val_score(self.model, features, residuals,
                                     cv=min(5, len(features)), scoring='r2')

        # Compute corrected predictions
        predicted_residuals = self.model.predict(features)
        corrected_preds = model_preds - predicted_residuals

        # Metrics
        original_mape = np.mean(np.abs(model_preds - ref_values) / np.abs(ref_values)) * 100
        corrected_mape = np.mean(np.abs(corrected_preds - ref_values) / np.abs(ref_values)) * 100
        original_rmse = np.sqrt(np.mean((model_preds - ref_values)**2))
        corrected_rmse = np.sqrt(np.mean((corrected_preds - ref_values)**2))

        metrics = {
            'original_mape': original_mape,
            'corrected_mape': corrected_mape,
            'mape_improvement': original_mape - corrected_mape,
            'original_rmse': original_rmse,
            'corrected_rmse': corrected_rmse,
            'cv_r2_mean': float(np.mean(cv_scores)),
            'cv_r2_std': float(np.std(cv_scores)),
            'n_training_points': len(features),
            'mean_correction_db': float(np.mean(np.abs(predicted_residuals))),
        }

        # Feature importance
        importances = self.model.feature_importances_
        feature_names = ['power', 'j_dist', 's_dist', 'altitude', 'gain', 'env', 'los_prob']
        metrics['feature_importance'] = dict(zip(feature_names, importances.tolist()))

        return metrics

    def correct(self, power_dbm: float, jammer_dist: float, signal_dist: float,
                altitude: float, gain: float, environment: str = 'urban',
                model_prediction: float = None) -> CorrectionResult:
        """
        Apply ML correction to a model prediction.

        Args:
            power_dbm: Jammer power in dBm
            jammer_dist: Jammer distance in m
            signal_dist: Signal distance in m
            altitude: Altitude in m
            gain: Antenna gain in dBi
            environment: Environment type
            model_prediction: Pre-computed model prediction (runs MC if None)

        Returns:
            CorrectionResult with original and corrected values
        """
        env_map = {'urban': 0, 'suburban': 1, 'rural': 2}
        a2g = AlHouraniA2GModel(environment)
        los_prob = a2g.los_probability(altitude, jammer_dist)

        features = np.array([[power_dbm, jammer_dist, signal_dist,
                               altitude, gain, env_map.get(environment, 0), los_prob]])

        if model_prediction is None:
            params = SimulationParams(
                jammer_power_dbm=power_dbm, jammer_distance_m=jammer_dist,
                jammer_antenna_gain_dbi=gain, signal_distance_m=signal_dist,
                altitude_m=altitude, fhss_enabled=False,
                propagation_model='al_hourani', environment=environment
            )
            result = self.engine.run_simulation(params, 2000, parallel=False, random_seed=42)
            model_prediction = result.mean_js_db

        correction = float(self.model.predict(features)[0])
        corrected = model_prediction - correction

        return CorrectionResult(
            original_prediction=model_prediction,
            correction_db=correction,
            corrected_prediction=corrected,
        )


if __name__ == "__main__":
    print("=" * 60)
    print("ML Propagation Correction — Training & Evaluation")
    print("=" * 60)
    corrector = PropagationCorrector()
    metrics = corrector.train()
    print(f"\nOriginal MAPE:  {metrics['original_mape']:.1f}%")
    print(f"Corrected MAPE: {metrics['corrected_mape']:.1f}%")
    print(f"Improvement:    {metrics['mape_improvement']:.1f}%")
    print(f"CV R2:          {metrics['cv_r2_mean']:.3f} +/- {metrics['cv_r2_std']:.3f}")
