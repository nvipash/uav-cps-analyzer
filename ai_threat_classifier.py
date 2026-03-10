#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: AI Threat Classification
Neural network threat classifier with hybrid Dempster-Shafer fusion.

Compares three approaches:
1. Pure Dempster-Shafer (existing physics-based)
2. Pure Neural Network (data-driven)
3. Hybrid NN + D-S (best of both)

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    from cps_analyzer import (
        ThreatType, SensorType, SensorFusion, DempsterShafer,
        RFSensor, RadarSensor, AcousticSensor, EOIRSensor, SENSOR_SPECS
    )
except ImportError:
    import sys
    sys.path.insert(0, '.')
    from cps_analyzer import (
        ThreatType, SensorType, SensorFusion, DempsterShafer,
        RFSensor, RadarSensor, AcousticSensor, EOIRSensor, SENSOR_SPECS
    )


THREAT_CLASSES = [ThreatType.RF_CONTROLLED, ThreatType.FIBER_OPTIC, ThreatType.AUTONOMOUS]
CLASS_NAMES = [t.value for t in THREAT_CLASSES]


@dataclass
class ClassificationMetrics:
    """Metrics for a classification approach."""
    accuracy: float
    f1_macro: float
    f1_per_class: Dict[str, float]
    confusion: np.ndarray
    method: str


class ThreatDataGenerator:
    """
    Generates labeled training data from sensor models with realistic noise.
    Simulates sensor readings for known threat types across varying conditions.
    """

    def __init__(self):
        self.sensors = {
            'rf': RFSensor(),
            'radar': RadarSensor(),
            'acoustic': AcousticSensor(),
            'eoir': EOIRSensor(),
        }

    def generate(self, n_samples_per_class: int = 500,
                 seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate labeled sensor data with extended feature engineering.

        Features per sample (24):
        Classical sensor (10):
        - rf_detected, rf_signal_strength
        - radar_detected, radar_rcs
        - acoustic_detected, acoustic_level
        - eoir_detected, eoir_confidence
        - range_m, bearing_deg

        Time-series / Behavioral (6):
        - rcs_variation (autonomous flies steadier)
        - velocity_variation (RF-controlled has reactive maneuvers)
        - altitude_variation
        - heading_variation
        - flight_smoothness (low-frequency power in trajectory)
        - reaction_time_to_jamming

        Spectral / RF Signature (5):
        - rf_bandwidth_hz (FHSS = wide, no signal = narrow)
        - rf_modulation_complexity (OFDM vs single carrier)
        - rf_hop_rate_hz (FHSS hopping signature)
        - rf_burstiness (datagram timing pattern)
        - rf_spectral_kurtosis

        Acoustic spectral (3):
        - acoustic_fundamental_hz (motor RPM signature)
        - acoustic_harmonic_ratio (rotor count signature)
        - acoustic_doppler_shift

        Returns:
            (X, y) where X is (n, 24) features and y is (n,) class labels
        """
        np.random.seed(seed)
        n_total = n_samples_per_class * len(THREAT_CLASSES)
        X = np.zeros((n_total, 24))
        y = np.zeros(n_total, dtype=int)

        idx = 0
        for class_idx, threat_type in enumerate(THREAT_CLASSES):
            for _ in range(n_samples_per_class):
                range_m = np.random.uniform(200, 3000)
                bearing = np.random.uniform(0, 360)

                # RF sensor
                rf_reading = self.sensors['rf'].detect(range_m, threat_type)
                if rf_reading:
                    X[idx, 0] = 1.0
                    X[idx, 1] = -40 + np.random.randn() * 10  # signal strength
                else:
                    X[idx, 0] = 0.0
                    X[idx, 1] = -100 + np.random.randn() * 5  # noise floor

                # Radar sensor
                radar_reading = self.sensors['radar'].detect(range_m, threat_type)
                if radar_reading:
                    X[idx, 2] = 1.0
                    if threat_type == ThreatType.RF_CONTROLLED:
                        X[idx, 3] = 0.01 + np.random.randn() * 0.003  # RCS
                    elif threat_type == ThreatType.FIBER_OPTIC:
                        X[idx, 3] = 0.02 + np.random.randn() * 0.005
                    else:
                        X[idx, 3] = 0.008 + np.random.randn() * 0.002
                else:
                    X[idx, 2] = 0.0
                    X[idx, 3] = 0.0

                # Acoustic sensor
                acoustic_reading = self.sensors['acoustic'].detect(range_m, threat_type)
                if acoustic_reading:
                    X[idx, 4] = 1.0
                    if threat_type == ThreatType.RF_CONTROLLED:
                        X[idx, 5] = 65 + np.random.randn() * 5
                    elif threat_type == ThreatType.FIBER_OPTIC:
                        X[idx, 5] = 70 + np.random.randn() * 5
                    else:
                        X[idx, 5] = 60 + np.random.randn() * 8
                else:
                    X[idx, 4] = 0.0
                    X[idx, 5] = 30 + np.random.randn() * 5  # ambient

                # EO/IR sensor
                eoir_reading = self.sensors['eoir'].detect(range_m, threat_type)
                if eoir_reading:
                    X[idx, 6] = 1.0
                    X[idx, 7] = np.random.uniform(0.5, 0.95)
                else:
                    X[idx, 6] = 0.0
                    X[idx, 7] = np.random.uniform(0.0, 0.3)

                X[idx, 8] = range_m / 3000.0  # normalized
                X[idx, 9] = bearing / 360.0    # normalized

                # ========== BEHAVIORAL FEATURES (10-15) ==========
                # Threat-type-specific motion patterns (normalized 0-1)
                if threat_type == ThreatType.RF_CONTROLLED:
                    # RF-controlled: reactive maneuvers, variable speed
                    X[idx, 10] = 0.30 + np.random.randn() * 0.10  # rcs_variation (medium)
                    X[idx, 11] = 0.65 + np.random.randn() * 0.15  # velocity_variation (high)
                    X[idx, 12] = 0.55 + np.random.randn() * 0.15  # altitude_variation
                    X[idx, 13] = 0.70 + np.random.randn() * 0.15  # heading_variation
                    X[idx, 14] = 0.40 + np.random.randn() * 0.10  # flight_smoothness (less smooth)
                    X[idx, 15] = 0.60 + np.random.randn() * 0.20  # reaction_to_jamming (high)
                elif threat_type == ThreatType.FIBER_OPTIC:
                    # Fiber-optic: human-controlled, similar to RF but no jamming reaction
                    X[idx, 10] = 0.35 + np.random.randn() * 0.10
                    X[idx, 11] = 0.60 + np.random.randn() * 0.15
                    X[idx, 12] = 0.50 + np.random.randn() * 0.15
                    X[idx, 13] = 0.65 + np.random.randn() * 0.15
                    X[idx, 14] = 0.45 + np.random.randn() * 0.10
                    X[idx, 15] = 0.05 + np.random.randn() * 0.05  # reaction (very low - no RF)
                else:  # AUTONOMOUS
                    # Autonomous: GPS-guided, smooth pre-programmed trajectory
                    X[idx, 10] = 0.10 + np.random.randn() * 0.05  # very low variation
                    X[idx, 11] = 0.15 + np.random.randn() * 0.08  # constant velocity
                    X[idx, 12] = 0.20 + np.random.randn() * 0.08  # constant altitude
                    X[idx, 13] = 0.25 + np.random.randn() * 0.10  # gradual turns
                    X[idx, 14] = 0.85 + np.random.randn() * 0.10  # very smooth
                    X[idx, 15] = 0.10 + np.random.randn() * 0.05  # ignores jamming

                # ========== SPECTRAL / RF SIGNATURE (16-20) ==========
                if X[idx, 0] == 1.0:  # RF detected
                    if threat_type == ThreatType.RF_CONTROLLED:
                        # FHSS protocol: wide bandwidth, OFDM, hopping
                        X[idx, 16] = 0.85 + np.random.randn() * 0.05  # bandwidth (wide)
                        X[idx, 17] = 0.80 + np.random.randn() * 0.10  # complex modulation (OFDM)
                        X[idx, 18] = 0.95 + np.random.randn() * 0.05  # hop rate ~500Hz (high)
                        X[idx, 19] = 0.40 + np.random.randn() * 0.10  # bursty
                        X[idx, 20] = 0.30 + np.random.randn() * 0.10  # spectral kurtosis
                    elif threat_type == ThreatType.AUTONOMOUS:
                        # GPS-guided: telemetry burst, narrow band
                        X[idx, 16] = 0.20 + np.random.randn() * 0.08
                        X[idx, 17] = 0.35 + np.random.randn() * 0.10
                        X[idx, 18] = 0.10 + np.random.randn() * 0.05  # no FHSS
                        X[idx, 19] = 0.85 + np.random.randn() * 0.10  # very bursty
                        X[idx, 20] = 0.70 + np.random.randn() * 0.15  # high kurtosis
                    else:  # fiber: no RF emission
                        X[idx, 16:21] = np.random.randn(5) * 0.05  # near zero
                else:
                    X[idx, 16:21] = np.random.randn(5) * 0.05

                # ========== ACOUSTIC SPECTRAL (21-23) ==========
                if X[idx, 4] == 1.0:  # acoustic detected
                    if threat_type == ThreatType.RF_CONTROLLED:
                        # Quad rotor: 4 rotors at ~5-8 kHz fundamental
                        X[idx, 21] = 0.55 + np.random.randn() * 0.10  # fundamental ~6kHz
                        X[idx, 22] = 0.75 + np.random.randn() * 0.10  # 4-rotor harmonic ratio
                        X[idx, 23] = 0.30 + np.random.randn() * 0.10  # moderate Doppler
                    elif threat_type == ThreatType.FIBER_OPTIC:
                        # Heavier UAV with fiber: lower RPM
                        X[idx, 21] = 0.40 + np.random.randn() * 0.10
                        X[idx, 22] = 0.65 + np.random.randn() * 0.10
                        X[idx, 23] = 0.20 + np.random.randn() * 0.08
                    else:  # autonomous fixed-wing or similar
                        # Different propulsion signature (single propeller, higher RPM)
                        X[idx, 21] = 0.85 + np.random.randn() * 0.10  # higher fundamental
                        X[idx, 22] = 0.20 + np.random.randn() * 0.08  # 1-prop signature
                        X[idx, 23] = 0.60 + np.random.randn() * 0.15  # high Doppler (forward flight)
                else:
                    X[idx, 21:24] = np.random.randn(3) * 0.05

                y[idx] = class_idx
                idx += 1

        # Clip features to reasonable range
        X[:, 10:24] = np.clip(X[:, 10:24], -0.2, 1.2)
        return X, y


class ThreatClassifier:
    """Neural network threat classifier with uncertainty estimation."""

    def __init__(self):
        self.model = None
        self.scaler = None

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train ensemble classifier (MLP + Random Forest + Gradient Boosting)
        with soft voting for robust prediction.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        self.scaler = StandardScaler().fit(X_train)
        X_tr = self.scaler.transform(X_train)
        X_te = self.scaler.transform(X_test)

        # Ensemble of three diverse classifiers
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                            max_iter=800, early_stopping=True, random_state=42)
        rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                     min_samples_split=5, random_state=42, n_jobs=1)
        gbm = GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                           learning_rate=0.1, random_state=42)

        self.model = VotingClassifier(
            estimators=[('mlp', mlp), ('rf', rf), ('gbm', gbm)],
            voting='soft'  # average predicted probabilities
        )
        self.model.fit(X_tr, y_train)

        y_pred = self.model.predict(X_te)
        proba = self.model.predict_proba(X_te)

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_per_class': dict(zip(CLASS_NAMES,
                                      f1_score(y_test, y_pred, average=None).tolist())),
            'confusion': confusion_matrix(y_test, y_pred),
            'mean_confidence': float(np.mean(np.max(proba, axis=1))),
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict threat type with probabilities.

        Returns:
            (class_indices, probability_matrix)
        """
        X_scaled = self.scaler.transform(X)
        classes = self.model.predict(X_scaled)
        proba = self.model.predict_proba(X_scaled)
        return classes, proba


class HybridFusion:
    """
    Combines NN predictions with Dempster-Shafer for robust classification.
    NN provides data-driven probabilities; D-S provides physics-based evidence.
    The hybrid uses D-S combination rule to fuse both.
    """

    def __init__(self, nn_classifier: ThreatClassifier):
        self.nn = nn_classifier
        self.ds = DempsterShafer(CLASS_NAMES)

    def classify(self, X: np.ndarray, sensor_fusion: SensorFusion,
                 range_m: float, threat_type: ThreatType
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hybrid classification combining NN and D-S.

        Args:
            X: Feature matrix for NN
            sensor_fusion: SensorFusion instance for D-S
            range_m: Range for D-S evaluation
            threat_type: Actual threat type (for D-S sensor simulation)

        Returns:
            (class_predictions, confidence_scores)
        """
        # NN predictions
        nn_classes, nn_proba = self.nn.predict(X)

        # D-S predictions for each sample
        n = len(X)
        hybrid_classes = np.zeros(n, dtype=int)
        hybrid_conf = np.zeros(n)

        for i in range(n):
            # NN mass function
            nn_mass = {}
            for j, name in enumerate(CLASS_NAMES):
                nn_mass[frozenset([name])] = nn_proba[i, j] * 0.8  # NN contributes 80%
            nn_mass[frozenset(CLASS_NAMES)] = 0.2  # 20% uncertainty

            # D-S mass function from sensor fusion
            detected, beliefs, conf = sensor_fusion.detect_and_fuse(range_m, threat_type)
            ds_mass = {}
            if detected and beliefs:
                for tt, bel in beliefs.items():
                    if tt.value in CLASS_NAMES:
                        ds_mass[frozenset([tt.value])] = bel * conf
                remaining = max(0, 1.0 - sum(ds_mass.values()))
                ds_mass[frozenset(CLASS_NAMES)] = remaining
            else:
                ds_mass[frozenset(CLASS_NAMES)] = 1.0  # total uncertainty

            # Combine using Dempster's rule
            combined = self.ds.combine(nn_mass, ds_mass)

            # Extract best class
            best_class = 0
            best_belief = 0
            for j, name in enumerate(CLASS_NAMES):
                bel = self.ds.belief(combined, frozenset([name]))
                if bel > best_belief:
                    best_belief = bel
                    best_class = j

            hybrid_classes[i] = best_class
            hybrid_conf[i] = best_belief

        return hybrid_classes, hybrid_conf


def run_comparison(n_samples: int = 500) -> Dict:
    """
    Compare pure D-S, pure NN, and hybrid approaches.

    Returns:
        Comparison results dict
    """
    # Generate data
    gen = ThreatDataGenerator()
    X, y = gen.generate(n_samples_per_class=n_samples)

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 1. Train NN
    nn = ThreatClassifier()
    nn_metrics = nn.train(X, y)

    # 2. Evaluate pure D-S on test set
    fusion = SensorFusion()
    ds_correct = 0
    ds_total = 0
    for i in range(len(X_test)):
        actual_type = THREAT_CLASSES[y_test[i]]
        range_m = X_test[i, 8] * 3000  # denormalize
        detected, beliefs, _ = fusion.detect_and_fuse(range_m, actual_type)
        if detected and beliefs:
            predicted_type = max(beliefs, key=beliefs.get)
            if predicted_type == actual_type:
                ds_correct += 1
        ds_total += 1

    ds_accuracy = ds_correct / ds_total if ds_total > 0 else 0

    # 3. Evaluate hybrid on test set
    hybrid = HybridFusion(nn)
    hybrid_correct = 0
    for i in range(len(X_test)):
        actual_type = THREAT_CLASSES[y_test[i]]
        range_m = X_test[i, 8] * 3000
        X_single = X_test[i:i+1]
        pred_classes, _ = hybrid.classify(X_single, fusion, range_m, actual_type)
        if pred_classes[0] == y_test[i]:
            hybrid_correct += 1

    hybrid_accuracy = hybrid_correct / len(X_test)

    return {
        'nn_accuracy': nn_metrics['accuracy'],
        'nn_f1': nn_metrics['f1_macro'],
        'nn_f1_per_class': nn_metrics['f1_per_class'],
        'nn_confusion': nn_metrics['confusion'],
        'nn_confidence': nn_metrics['mean_confidence'],
        'ds_accuracy': ds_accuracy,
        'hybrid_accuracy': hybrid_accuracy,
        'n_test': len(X_test),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("AI Threat Classification — Comparison")
    print("=" * 60)
    results = run_comparison(n_samples=300)
    print(f"\nPure Dempster-Shafer: {results['ds_accuracy']*100:.1f}%")
    print(f"Pure Neural Network:  {results['nn_accuracy']*100:.1f}% (F1={results['nn_f1']:.3f})")
    print(f"Hybrid NN + D-S:      {results['hybrid_accuracy']*100:.1f}%")
    print(f"\nNN F1 per class:")
    for cls, f1 in results['nn_f1_per_class'].items():
        print(f"  {cls:<20}: {f1:.3f}")
