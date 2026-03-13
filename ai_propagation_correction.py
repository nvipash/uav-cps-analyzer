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
    from propagation_models import AltitudeDependentModel, AlHouraniA2GModel, FriisModel
    from literature_dataset import LiteratureDataset
except ImportError:
    import sys
    sys.path.insert(0, '.')
    from monte_carlo_engine import MonteCarloEngine, SimulationParams
    from propagation_models import AltitudeDependentModel, AlHouraniA2GModel, FriisModel
    from literature_dataset import LiteratureDataset


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
        Generate training data: model predictions vs literature-measured reference.

        Еталонні значення обчислені з РЕАЛЬНО ВИМІРЯНИХ показників path loss exponent
        (n_LOS, n_NLOS) з польових кампаній Khawaja et al. (2019) та Matolak & Sun (2017).
        Замість синтетичної формули Фріса (n=2.0 скрізь) тепер використовуються
        середовищно-залежні та висотно-залежні виміряні значення n.

        Джерела: Khawaja (2019) IEEE Commun. Surveys Tuts. Tables 3–5;
                 Matolak & Sun (2017) IEEE Trans. Veh. Technol.

        Returns:
            (features, model_predictions, reference_values)
        """
        FREQ_MHZ      = 2437.0
        SIG_POWER_DBM = 20.0   # DJI controller 100 mW
        SIG_GAIN_DBI  = 2.0

        # Load scenarios with measured reference J/S values
        lit_scenarios = LiteratureDataset.get_a2g_training_scenarios()

        n = len(lit_scenarios)
        features   = np.zeros((n, 7))  # power, j_dist, s_dist, alt, gain, env_code, los_prob
        model_preds = np.zeros(n)
        ref_values  = np.zeros(n)

        env_map = {'dense_urban': -1, 'urban': 0, 'suburban': 1, 'rural': 2, 'open_field': 3}

        for i, s in enumerate(lit_scenarios):
            power_dbm = 10.0 * np.log10(s['power_w'] * 1000.0)
            jd  = s['j_dist']
            sd  = s['s_dist']
            alt = s['alt']
            gain = s['gain']
            env  = s['env']

            # Al-Hourani model prediction via Monte Carlo
            sim_params = SimulationParams(
                jammer_power_dbm=power_dbm, jammer_distance_m=jd,
                jammer_antenna_gain_dbi=gain,
                signal_power_dbm=SIG_POWER_DBM,
                signal_distance_m=sd,
                signal_antenna_gain_dbi=SIG_GAIN_DBI,
                frequency_mhz=FREQ_MHZ,
                altitude_m=alt, fhss_enabled=False,
                propagation_model='al_hourani', environment=env,
            )
            result = self.engine.run_simulation(
                sim_params, 2000, parallel=False, random_seed=42 + i
            )

            # Еталон: J/S з ВИМІРЯНИХ показників path loss (Khawaja / Matolak)
            ref_js = s['ref_js']

            # LOS probability for feature engineering
            a2g = AlHouraniA2GModel(env)
            los_prob = a2g.los_probability(alt, jd)

            features[i]   = [power_dbm, jd, sd, alt, gain, env_map.get(env, 0), los_prob]
            model_preds[i] = result.mean_js_db
            ref_values[i]  = ref_js

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
        metrics['n_lit_scenarios'] = len(LiteratureDataset.get_a2g_training_scenarios())

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
