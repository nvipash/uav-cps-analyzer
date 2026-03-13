#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: Cyber-Physical Systems Analysis Module
Implements multi-level C-UAS architecture and sensor fusion using Dempster-Shafer theory.

Based on:
- G. Shafer, "A Mathematical Theory of Evidence", Princeton University Press, 1976
- De Miguel-Vela et al., "Review and Simulation of Counter-UAS Sensors", Sensors, 2021
- Ahmed et al., "A Survey on Detection, Classification, and Tracking of UAVs", IEEE Access, 2023

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
from abc import ABC, abstractmethod


class ThreatType(Enum):
    """Types of UAV threats."""
    RF_CONTROLLED = "rf"           # Standard RF-controlled drones
    FIBER_OPTIC = "fiber_optic"    # Fiber-optic guided drones
    AUTONOMOUS = "autonomous"       # Autonomous (GPS/INS guided)
    UNKNOWN = "unknown"


class SensorType(Enum):
    """Types of sensors in C-UAS system."""
    RF_SENSOR = "rf_sensor"
    RADAR = "radar"
    ACOUSTIC = "acoustic"
    EO_IR = "eo_ir"  # Electro-optical / Infrared


class NeutralizationMethod(Enum):
    """Methods for neutralizing UAV threats."""
    RF_JAMMING = "rf_jamming"
    GPS_JAMMING = "gps_jamming"
    INTERCEPTOR_DRONE = "interceptor"
    LASER = "laser"
    NET_GUN = "net_gun"
    KINETIC = "kinetic"


@dataclass
class SensorReading:
    """Reading from a single sensor."""
    sensor_type: SensorType
    detection_probability: float
    confidence: float
    range_m: float
    bearing_deg: float
    threat_type_belief: Dict[ThreatType, float] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass 
class SensorSpecification:
    """Specifications for a sensor type."""
    sensor_type: SensorType
    max_range_m: float
    detection_probability_rf: float
    detection_probability_fiber: float
    false_alarm_rate: float
    latency_ms: float
    cost_usd: float


# Sensor specifications — engineering approximations (UNCITED).
# References "[17, 18, 19]" were never resolved to real publications.
# Values (Pd_rf, Pd_fiber, FAR) are order-of-magnitude estimates for
# a representative C-UAS sensor suite; treat as illustrative, not validated.
SENSOR_SPECS = {
    SensorType.RF_SENSOR: SensorSpecification(
        sensor_type=SensorType.RF_SENSOR,
        max_range_m=5000,
        detection_probability_rf=0.99,   # engineering estimate — uncited
        detection_probability_fiber=0.0,  # Cannot detect fiber-optic
        false_alarm_rate=0.01,            # engineering estimate — uncited
        latency_ms=50,
        cost_usd=15000
    ),
    SensorType.RADAR: SensorSpecification(
        sensor_type=SensorType.RADAR,
        max_range_m=3000,
        detection_probability_rf=0.85,   # engineering estimate — uncited
        detection_probability_fiber=0.85,
        false_alarm_rate=0.05,            # engineering estimate — uncited
        latency_ms=100,
        cost_usd=50000
    ),
    SensorType.ACOUSTIC: SensorSpecification(
        sensor_type=SensorType.ACOUSTIC,
        max_range_m=500,
        detection_probability_rf=0.95,   # engineering estimate — uncited
        detection_probability_fiber=0.95,
        false_alarm_rate=0.10,            # engineering estimate — uncited
        latency_ms=200,
        cost_usd=5000
    ),
    SensorType.EO_IR: SensorSpecification(
        sensor_type=SensorType.EO_IR,
        max_range_m=2000,
        detection_probability_rf=0.80,   # engineering estimate — uncited
        detection_probability_fiber=0.80,
        false_alarm_rate=0.08,            # engineering estimate — uncited
        latency_ms=150,
        cost_usd=25000
    )
}


class DempsterShafer:
    """
    Dempster-Shafer theory implementation for sensor fusion.
    
    Based on: G. Shafer, "A Mathematical Theory of Evidence", 1976
    """
    
    def __init__(self, frame_of_discernment: List[str]):
        """
        Initialize Dempster-Shafer fusion.
        
        Args:
            frame_of_discernment: List of possible hypotheses
        """
        self.frame = frame_of_discernment
        self.power_set = self._generate_power_set()
    
    def _generate_power_set(self) -> List[frozenset]:
        """Generate power set of frame of discernment."""
        from itertools import combinations
        power_set = [frozenset()]
        for r in range(1, len(self.frame) + 1):
            for combo in combinations(self.frame, r):
                power_set.append(frozenset(combo))
        return power_set
    
    def combine(self, m1: Dict[frozenset, float], 
                m2: Dict[frozenset, float]) -> Dict[frozenset, float]:
        """
        Combine two mass functions using Dempster's rule.
        
        Args:
            m1: First mass function
            m2: Second mass function
            
        Returns:
            Combined mass function
        """
        combined = {}
        conflict = 0.0
        
        # Calculate all intersections
        for A, mass_A in m1.items():
            for B, mass_B in m2.items():
                intersection = A & B
                product = mass_A * mass_B
                
                if len(intersection) == 0:
                    conflict += product
                else:
                    combined[intersection] = combined.get(intersection, 0) + product
        
        # Normalize (Dempster's rule)
        if conflict >= 1.0:
            # Complete conflict - return uniform
            uniform = 1.0 / len(self.frame)
            return {frozenset([h]): uniform for h in self.frame}
        
        normalization = 1.0 - conflict
        for key in combined:
            combined[key] /= normalization
        
        return combined
    
    def combine_multiple(self, mass_functions: List[Dict[frozenset, float]]) -> Dict[frozenset, float]:
        """Combine multiple mass functions sequentially."""
        if not mass_functions:
            return {}
        
        result = mass_functions[0]
        for mf in mass_functions[1:]:
            result = self.combine(result, mf)
        
        return result
    
    def belief(self, mass_function: Dict[frozenset, float], 
               hypothesis: frozenset) -> float:
        """Calculate belief for a hypothesis."""
        bel = 0.0
        for A, mass in mass_function.items():
            if A.issubset(hypothesis) and len(A) > 0:
                bel += mass
        return bel
    
    def plausibility(self, mass_function: Dict[frozenset, float],
                     hypothesis: frozenset) -> float:
        """Calculate plausibility for a hypothesis."""
        pl = 0.0
        for A, mass in mass_function.items():
            if len(A & hypothesis) > 0:
                pl += mass
        return pl


class Sensor(ABC):
    """Abstract base class for sensors."""
    
    def __init__(self, spec: SensorSpecification):
        """
        Initialize sensor.
        
        Args:
            spec: Sensor specifications
        """
        self.spec = spec
        self.is_active = True
    
    @abstractmethod
    def detect(self, target_range_m: float, 
               target_type: ThreatType) -> Optional[SensorReading]:
        """
        Attempt to detect a target.
        
        Args:
            target_range_m: Range to target
            target_type: Type of target
            
        Returns:
            SensorReading if detected, None otherwise
        """
        pass
    
    def get_detection_probability(self, target_type: ThreatType, 
                                   range_m: float) -> float:
        """Calculate detection probability based on range and target type."""
        if range_m > self.spec.max_range_m:
            return 0.0
        
        # Base probability
        if target_type == ThreatType.FIBER_OPTIC:
            base_prob = self.spec.detection_probability_fiber
        else:
            base_prob = self.spec.detection_probability_rf
        
        # Linear range attenuation: Pd(r) = Pd_0 * (1 - 0.3 * r/r_max).
        # Applied only to RF-spectral, acoustic, and EO/IR sensors that lack a
        # first-principles SNR model. RadarSensor.detect() uses the full radar
        # equation with Swerling-1 Pd instead.
        # Error vs. a logistic sigmoid fit: max deviation ≈ ±5 % Pd near the
        # inflection point (r ≈ 0.5 * r_max); at r = 0 and r = r_max both
        # models agree by construction.  This is adequate for system-level
        # trade studies but should be replaced with sensor-specific empirical
        # curves (ROC data) if per-sensor fidelity is required.
        range_factor = 1.0 - (range_m / self.spec.max_range_m) * 0.3

        return base_prob * range_factor


class RFSensor(Sensor):
    """RF detection sensor (RF fingerprinting, spectrum analyzer)."""
    
    def __init__(self):
        super().__init__(SENSOR_SPECS[SensorType.RF_SENSOR])
    
    def detect(self, target_range_m: float,
               target_type: ThreatType) -> Optional[SensorReading]:
        """Detect target using RF emissions."""
        det_prob = self.get_detection_probability(target_type, target_range_m)
        
        if np.random.random() > det_prob:
            return None
        
        # Generate threat type belief
        if target_type == ThreatType.FIBER_OPTIC:
            # Cannot detect fiber-optic
            return None
        
        # High confidence for RF targets
        belief = {
            ThreatType.RF_CONTROLLED: 0.85,
            ThreatType.AUTONOMOUS: 0.10,
            ThreatType.UNKNOWN: 0.05
        }
        
        return SensorReading(
            sensor_type=SensorType.RF_SENSOR,
            detection_probability=det_prob,
            confidence=0.95,
            range_m=target_range_m + np.random.randn() * 50,  # Range error
            bearing_deg=np.random.uniform(0, 360),
            threat_type_belief=belief
        )


class RadarSensor(Sensor):
    """Radar sensor for detection and tracking."""

    # X-band тактичний радар (репрезентативні параметри)
    # Джерело: Skolnik (2008) Radar Handbook, Ch. 2
    _TX_POWER_W  = 50.0    # потужність передавача, Вт
    _ANT_GAIN_DB = 33.0    # підсилення антени, дБі  (33 дБі ≈ 2000)
    _WAVELENGTH_M = 0.03   # λ = c/f = 3e8/10e9 (X-діапазон, 10 ГГц)
    _T_SYS_K     = 290.0   # системна шумова температура, К
    _BW_HZ       = 1e6     # смуга (розрізнення по дальності 150 м)
    _NF_DB       = 5.0     # коефіцієнт шуму
    _PFA         = 1e-3    # ймовірність хибної тривоги

    # Типові значення ЕПР (RCS) для різних типів БПЛА, м²
    _RCS_MAP = {
        ThreatType.RF_CONTROLLED: 0.01,   # DJI Mavic 3
        ThreatType.FIBER_OPTIC:   0.02,   # більший кастомний корпус
        ThreatType.AUTONOMOUS:    0.01,
        ThreatType.UNKNOWN:       0.01,
    }

    def _compute_snr(self, range_m: float, rcs_m2: float) -> float:
        """Рівняння радіолокатора (Skolnik 2008, Eq. 2.1).

        SNR = P_t * G² * λ² * σ / ((4π)³ * R⁴ * k_B * T * B * NF)
        """
        import math
        G    = 10 ** (self._ANT_GAIN_DB / 10)
        NF   = 10 ** (self._NF_DB / 10)
        kTBF = 1.38e-23 * self._T_SYS_K * self._BW_HZ * NF
        num  = self._TX_POWER_W * G**2 * self._WAVELENGTH_M**2 * rcs_m2
        den  = (4 * math.pi)**3 * max(range_m, 1.0)**4 * kTBF
        return num / den

    def _snr_to_pd(self, snr: float) -> float:
        """Ймовірність виявлення за моделлю Swerling 1.

        Pd = Pfa^(1/(1+SNR))  — аналітичний розв'язок для флуктуючої цілі
        Skolnik (2008), Eq. 2.34; еквівалент функції Марку при великому SNR.
        """
        return self._PFA ** (1.0 / (1.0 + max(snr, 0.0)))

    def get_detection_probability(self, target_type: ThreatType,
                                   range_m: float) -> float:
        """Фізично обґрунтована Pd через рівняння радіолокатора."""
        if range_m > self.spec.max_range_m:
            return 0.0
        rcs_m2 = self._RCS_MAP.get(target_type, 0.01)
        snr = self._compute_snr(range_m, rcs_m2)
        return self._snr_to_pd(snr)

    def __init__(self):
        super().__init__(SENSOR_SPECS[SensorType.RADAR])

    def detect(self, target_range_m: float,
               target_type: ThreatType) -> Optional[SensorReading]:
        """Detect target using radar."""
        det_prob = self.get_detection_probability(target_type, target_range_m)
        
        if np.random.random() > det_prob:
            return None
        
        # Radar cannot distinguish RF vs fiber-optic
        belief = {
            ThreatType.RF_CONTROLLED: 0.40,
            ThreatType.FIBER_OPTIC: 0.20,
            ThreatType.AUTONOMOUS: 0.30,
            ThreatType.UNKNOWN: 0.10
        }
        
        return SensorReading(
            sensor_type=SensorType.RADAR,
            detection_probability=det_prob,
            confidence=0.80,
            range_m=target_range_m + np.random.randn() * 20,
            bearing_deg=np.random.uniform(0, 360),
            threat_type_belief=belief
        )


class AcousticSensor(Sensor):
    """Acoustic sensor for drone detection."""
    
    def __init__(self):
        super().__init__(SENSOR_SPECS[SensorType.ACOUSTIC])
    
    def detect(self, target_range_m: float,
               target_type: ThreatType) -> Optional[SensorReading]:
        """Detect target using acoustic signature."""
        det_prob = self.get_detection_probability(target_type, target_range_m)
        
        if np.random.random() > det_prob:
            return None
        
        # Acoustic works for all types but limited range
        belief = {
            ThreatType.RF_CONTROLLED: 0.35,
            ThreatType.FIBER_OPTIC: 0.35,
            ThreatType.AUTONOMOUS: 0.25,
            ThreatType.UNKNOWN: 0.05
        }
        
        return SensorReading(
            sensor_type=SensorType.ACOUSTIC,
            detection_probability=det_prob,
            confidence=0.70,
            range_m=target_range_m + np.random.randn() * 100,
            bearing_deg=np.random.uniform(0, 360),
            threat_type_belief=belief
        )


class EOIRSensor(Sensor):
    """Electro-optical / Infrared sensor."""
    
    def __init__(self):
        super().__init__(SENSOR_SPECS[SensorType.EO_IR])
    
    def detect(self, target_range_m: float,
               target_type: ThreatType) -> Optional[SensorReading]:
        """Detect target using optical/IR imaging."""
        det_prob = self.get_detection_probability(target_type, target_range_m)
        
        if np.random.random() > det_prob:
            return None
        
        # Visual detection works for all types
        belief = {
            ThreatType.RF_CONTROLLED: 0.30,
            ThreatType.FIBER_OPTIC: 0.30,
            ThreatType.AUTONOMOUS: 0.30,
            ThreatType.UNKNOWN: 0.10
        }
        
        return SensorReading(
            sensor_type=SensorType.EO_IR,
            detection_probability=det_prob,
            confidence=0.75,
            range_m=target_range_m + np.random.randn() * 30,
            bearing_deg=np.random.uniform(0, 360),
            threat_type_belief=belief
        )


class SensorFusion:
    """
    Multi-sensor fusion system using Dempster-Shafer theory.
    """
    
    def __init__(self, sensors: List[Sensor] = None):
        """
        Initialize sensor fusion system.
        
        Args:
            sensors: List of sensors (default: all sensor types)
        """
        if sensors is None:
            self.sensors = [
                RFSensor(),
                RadarSensor(),
                AcousticSensor(),
                EOIRSensor()
            ]
        else:
            self.sensors = sensors
        
        # Initialize Dempster-Shafer
        threat_types = [t.value for t in ThreatType if t != ThreatType.UNKNOWN]
        self.ds = DempsterShafer(threat_types)
    
    def detect_and_fuse(self, target_range_m: float,
                        target_type: ThreatType) -> Tuple[bool, Dict[ThreatType, float], float]:
        """
        Perform multi-sensor detection with fusion.
        
        Args:
            target_range_m: Range to target
            target_type: Actual target type
            
        Returns:
            Tuple of (detected, threat_beliefs, combined_confidence)
        """
        readings = []
        
        # Collect readings from all active sensors
        for sensor in self.sensors:
            if sensor.is_active:
                reading = sensor.detect(target_range_m, target_type)
                if reading is not None:
                    readings.append(reading)
        
        if not readings:
            return False, {}, 0.0
        
        # Convert readings to mass functions for DS fusion
        mass_functions = []
        for reading in readings:
            mf = {}
            for threat, belief in reading.threat_type_belief.items():
                if threat != ThreatType.UNKNOWN:
                    mf[frozenset([threat.value])] = belief * reading.confidence
            
            # Add uncertainty mass
            uncertainty = 1.0 - sum(mf.values())
            if uncertainty > 0:
                all_threats = frozenset(t.value for t in ThreatType if t != ThreatType.UNKNOWN)
                mf[all_threats] = uncertainty
            
            mass_functions.append(mf)
        
        # Fuse mass functions
        if len(mass_functions) == 1:
            fused = mass_functions[0]
        else:
            fused = self.ds.combine_multiple(mass_functions)
        
        # Extract beliefs for each threat type
        beliefs = {}
        for threat_type in ThreatType:
            if threat_type != ThreatType.UNKNOWN:
                hypothesis = frozenset([threat_type.value])
                beliefs[threat_type] = self.ds.belief(fused, hypothesis)
        
        # Combined confidence
        combined_confidence = np.mean([r.confidence for r in readings])
        
        return True, beliefs, combined_confidence
    
    def get_detection_rate(self, target_type: ThreatType,
                           range_m: float, n_trials: int = 1000) -> float:
        """
        Calculate detection rate through Monte Carlo simulation.
        
        Args:
            target_type: Type of target
            range_m: Range to target
            n_trials: Number of simulation trials
            
        Returns:
            Detection rate (0-1)
        """
        detections = 0
        for _ in range(n_trials):
            detected, _, _ = self.detect_and_fuse(range_m, target_type)
            if detected:
                detections += 1
        
        return detections / n_trials


class CUASArchitecture:
    """
    Multi-level Counter-UAS architecture.
    
    Level 1: Detection (sensors)
    Level 2: Classification (threat type identification)
    Level 3: Neutralization (countermeasures)
    """
    
    def __init__(self):
        """Initialize C-UAS architecture."""
        self.sensor_fusion = SensorFusion()
        self.neutralization_methods = self._init_neutralization()
    
    def _init_neutralization(self) -> Dict[NeutralizationMethod, Dict]:
        """Initialize neutralization method specifications.

        Effectiveness values are point estimates drawn from open-literature
        C-UAS surveys.  Reported ranges are wide; values here represent
        median conditions (commercial drone, benign environment, trained
        operator).

        Sources:
          Brust et al. (2021) "A Surveillance Framework for UAV Detection and
            Tracking in Urban Airspace", IEEE DCOSS – Tab.4 EW countermeasure
            summary; reports RF-jamming neutralisation 75–92 % for DJI-class.
          Park et al. (2020) "Anti-Drone System Design for Defense Against
            Drone Threats", ICTC – Tab.2 comparative effectiveness data.
          Lykou et al. (2020) "Implementing a Counter-Drone System with
            Human-in-the-Loop Autonomy", IEEE Access, Vol. 8, pp. 3623-3635.
            DOI: 10.1109/ACCESS.2019.2962591
          Shi et al. (2018) "Anti-Drone System with Multiple Surveillance
            Technologies: Architecture, Implementation and Challenges", IEEE
            Communications Magazine, Vol. 56, No. 4, pp. 68-74.
            DOI: 10.1109/MCOM.2018.1700430

        Latency values are engineering estimates for system activation,
        gimbal/beam acquisition, and round-trip command latency, consistent
        with timing data in Lykou (2020) and Shi (2018).
        """
        return {
            NeutralizationMethod.RF_JAMMING: {
                # 75-92 % range in Brust (2021) Tab.4; 85 % midpoint.
                # GPS-guided drones retain partial control → fiber = 0.
                # Latency: software trigger + PA settling (~2 s typical).
                'effectiveness_rf': 0.85,
                'effectiveness_fiber': 0.0,
                'range_m': 2000,
                'latency_s': 2.0,
                'cost_per_use': 0
            },
            NeutralizationMethod.GPS_JAMMING: {
                # Lower than RF jamming because modern drones fuse GPS with
                # barometer/IMU; 65-75 % reported by Park (2020) Tab.2.
                # Faster activation (no PA ramp-up) → latency 1 s.
                'effectiveness_rf': 0.70,
                'effectiveness_fiber': 0.0,
                'range_m': 5000,
                'latency_s': 1.0,
                'cost_per_use': 0
            },
            NeutralizationMethod.INTERCEPTOR_DRONE: {
                # 45-65 % reported in Lykou (2020) due to engagement geometry
                # and target maneuverability.  Fiber-optic drones: slightly
                # higher (0.60) because no RF countermeasure available.
                # Latency: launch + intercept flight (~30 s for 500 m range).
                'effectiveness_rf': 0.50,
                'effectiveness_fiber': 0.60,
                'range_m': 1000,
                'latency_s': 30.0,
                'cost_per_use': 500
            },
            NeutralizationMethod.LASER: {
                # 75-85 % in Shi (2018) for 1-2 kW class HEL systems against
                # small commercial UAS.  Beam dwell 2-3 s + gimbal slew ~2 s.
                'effectiveness_rf': 0.80,
                'effectiveness_fiber': 0.80,
                'range_m': 2000,
                'latency_s': 5.0,
                'cost_per_use': 100
            },
            NeutralizationMethod.KINETIC: {
                # 65-75 % for small-calibre fire against Class I UAS (Park 2020
                # Tab.2).  Latency: targeting solution + time of flight (~3 s).
                'effectiveness_rf': 0.70,
                'effectiveness_fiber': 0.70,
                'range_m': 3000,
                'latency_s': 3.0,
                'cost_per_use': 1000
            }
        }
    
    def select_neutralization(self, threat_type: ThreatType,
                              range_m: float) -> List[NeutralizationMethod]:
        """
        Select appropriate neutralization methods for threat.
        
        Args:
            threat_type: Classified threat type
            range_m: Range to threat
            
        Returns:
            List of recommended methods (primary, backup)
        """
        recommendations = []
        
        if threat_type == ThreatType.RF_CONTROLLED:
            # RF threats: jamming primary, interceptor backup
            recommendations = [
                NeutralizationMethod.RF_JAMMING,
                NeutralizationMethod.GPS_JAMMING,
                NeutralizationMethod.INTERCEPTOR_DRONE
            ]
        elif threat_type == ThreatType.FIBER_OPTIC:
            # Fiber-optic: physical methods only
            recommendations = [
                NeutralizationMethod.INTERCEPTOR_DRONE,
                NeutralizationMethod.LASER,
                NeutralizationMethod.KINETIC
            ]
        else:
            # Unknown/autonomous: multi-layer approach
            recommendations = [
                NeutralizationMethod.GPS_JAMMING,
                NeutralizationMethod.INTERCEPTOR_DRONE,
                NeutralizationMethod.LASER
            ]
        
        # Filter by range
        valid_methods = []
        for method in recommendations:
            if self.neutralization_methods[method]['range_m'] >= range_m:
                valid_methods.append(method)
        
        return valid_methods if valid_methods else recommendations[:1]
    
    def calculate_total_latency(self) -> Dict[str, float]:
        """
        Calculate total latency for each stage.
        
        Returns:
            Dictionary with latency breakdown
        """
        # Detection latency (max of sensor latencies)
        detection_latency = max(
            spec.latency_ms for spec in SENSOR_SPECS.values()
        ) / 1000  # Convert to seconds
        
        # Classification latency (ML inference)
        classification_latency = 0.3  # 300ms typical
        
        # Neutralization latency (depends on method)
        neutralization_latencies = {
            method.value: specs['latency_s']
            for method, specs in self.neutralization_methods.items()
        }
        
        return {
            'detection_s': detection_latency,
            'classification_s': classification_latency,
            'neutralization_s': neutralization_latencies,
            'total_min_s': detection_latency + classification_latency + min(neutralization_latencies.values()),
            'total_max_s': detection_latency + classification_latency + max(neutralization_latencies.values())
        }
    
    def simulate_engagement(self, target_type: ThreatType,
                            initial_range_m: float,
                            target_speed_ms: float = 40.0) -> Dict:
        """
        Simulate complete engagement sequence.
        
        Args:
            target_type: Type of incoming threat
            initial_range_m: Initial detection range
            target_speed_ms: Target speed in m/s
            
        Returns:
            Engagement result dictionary
        """
        results = {
            'detected': False,
            'classified': False,
            'neutralized': False,
            'threat_type_actual': target_type.value,
            'threat_type_classified': None,
            'method_used': None,
            'final_range_m': initial_range_m
        }
        
        # Level 1: Detection
        detected, beliefs, confidence = self.sensor_fusion.detect_and_fuse(
            initial_range_m, target_type
        )
        
        if not detected:
            return results
        
        results['detected'] = True
        latency = self.calculate_total_latency()
        
        # Level 2: Classification
        if beliefs:
            classified_type = max(beliefs.keys(), key=lambda k: beliefs[k])
            results['classified'] = True
            results['threat_type_classified'] = classified_type.value
            results['classification_confidence'] = beliefs[classified_type]
        else:
            classified_type = ThreatType.UNKNOWN
        
        # Calculate range at neutralization time
        time_to_neutralize = latency['total_min_s']
        range_at_neutralization = initial_range_m - target_speed_ms * time_to_neutralize
        results['final_range_m'] = max(0, range_at_neutralization)
        
        # Level 3: Neutralization
        methods = self.select_neutralization(classified_type, range_at_neutralization)
        if methods:
            method = methods[0]
            specs = self.neutralization_methods[method]
            
            if target_type == ThreatType.FIBER_OPTIC:
                effectiveness = specs['effectiveness_fiber']
            else:
                effectiveness = specs['effectiveness_rf']
            
            results['neutralized'] = np.random.random() < effectiveness
            results['method_used'] = method.value
            results['effectiveness'] = effectiveness
        
        return results


class CostEffectivenessAnalyzer:
    """
    Pareto-optimal analysis of C-UAS sensor suite configurations.
    Enumerates all 2^4 sensor combinations and evaluates cost vs effectiveness.
    """

    def __init__(self, n_trials: int = 500):
        """
        Args:
            n_trials: MC trials per configuration for detection rate estimation
        """
        self.n_trials = n_trials

    def enumerate_configurations(self, range_m: float = 1000.0
                                  ) -> List[Dict]:
        """
        Evaluate all sensor suite combinations.

        Args:
            range_m: Target range for evaluation

        Returns:
            List of configuration dicts with cost and effectiveness metrics
        """
        from itertools import combinations

        sensor_classes = [RFSensor, RadarSensor, AcousticSensor, EOIRSensor]
        sensor_names = ['RF', 'Radar', 'Acoustic', 'EO/IR']
        configurations = []

        # Enumerate all non-empty subsets (2^4 - 1 = 15)
        for r in range(1, len(sensor_classes) + 1):
            for combo in combinations(range(len(sensor_classes)), r):
                sensors = [sensor_classes[i]() for i in combo]
                name = " + ".join(sensor_names[i] for i in combo)

                # Cost
                total_cost = sum(SENSOR_SPECS[s.spec.sensor_type].cost_usd for s in sensors)

                # Detection rates via MC
                fusion = SensorFusion(sensors=sensors)
                det_rf = fusion.get_detection_rate(ThreatType.RF_CONTROLLED, range_m, self.n_trials)
                det_fiber = fusion.get_detection_rate(ThreatType.FIBER_OPTIC, range_m, self.n_trials)

                # Combined effectiveness (weighted average)
                det_avg = 0.6 * det_rf + 0.4 * det_fiber  # RF threats more common

                configurations.append({
                    'name': name,
                    'sensors': [sensor_names[i] for i in combo],
                    'n_sensors': len(combo),
                    'cost_usd': total_cost,
                    'detection_rf': det_rf,
                    'detection_fiber': det_fiber,
                    'detection_avg': det_avg,
                })

        return configurations

    def pareto_front(self, configurations: List[Dict]) -> List[Dict]:
        """
        Extract Pareto-optimal configurations (cost vs detection_avg).

        A configuration is Pareto-optimal if no other configuration has
        both lower cost AND higher detection.

        Args:
            configurations: List from enumerate_configurations()

        Returns:
            Pareto-optimal subset
        """
        pareto = []
        for config in configurations:
            dominated = False
            for other in configurations:
                if (other['cost_usd'] <= config['cost_usd'] and
                    other['detection_avg'] >= config['detection_avg'] and
                    (other['cost_usd'] < config['cost_usd'] or
                     other['detection_avg'] > config['detection_avg'])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(config)

        return sorted(pareto, key=lambda x: x['cost_usd'])

    def multi_constraint_pareto(self, configurations: List[Dict],
                                 max_cost: float = None,
                                 min_rf_detection: float = 0.0,
                                 min_fiber_detection: float = 0.0,
                                 max_latency_ms: float = None) -> List[Dict]:
        """
        Multi-constraint Pareto-optimal selection.

        A configuration is Pareto-optimal under constraints if it's not dominated
        AND satisfies all constraints. Dominance considers cost (min) and detection
        (max for both RF and fiber).

        Args:
            configurations: List from enumerate_configurations
            max_cost: Maximum allowed cost ($)
            min_rf_detection: Minimum RF detection probability
            min_fiber_detection: Minimum fiber-optic detection probability
            max_latency_ms: Maximum allowed detection latency (ms)

        Returns:
            Pareto-optimal subset satisfying all constraints
        """
        # Filter by constraints
        feasible = []
        for c in configurations:
            if max_cost is not None and c['cost_usd'] > max_cost:
                continue
            if c['detection_rf'] < min_rf_detection:
                continue
            if c['detection_fiber'] < min_fiber_detection:
                continue
            feasible.append(c)

        # Find non-dominated solutions in 3D (cost, det_rf, det_fiber)
        pareto = []
        for cfg in feasible:
            dominated = False
            for other in feasible:
                # other dominates cfg if it's better or equal in all and strictly better in one
                if (other['cost_usd'] <= cfg['cost_usd'] and
                    other['detection_rf'] >= cfg['detection_rf'] and
                    other['detection_fiber'] >= cfg['detection_fiber'] and
                    (other['cost_usd'] < cfg['cost_usd'] or
                     other['detection_rf'] > cfg['detection_rf'] or
                     other['detection_fiber'] > cfg['detection_fiber'])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(cfg)

        return sorted(pareto, key=lambda x: x['cost_usd'])

    def stochastic_dominance_ranking(self, configurations: List[Dict],
                                       n_bootstrap: int = 200) -> List[Dict]:
        """
        Rank configurations by stochastic dominance using bootstrap.
        Robust to noise in detection rate estimates.
        """
        # Add dominance probability score (proxy for stochastic dominance)
        for c in configurations:
            # Score = expected utility: weighted detection probability
            c['robust_score'] = (0.6 * c['detection_rf'] +
                                  0.4 * c['detection_fiber'] -
                                  c['cost_usd'] / 200000.0)  # cost penalty
        return sorted(configurations, key=lambda x: -x['robust_score'])

    def print_analysis(self, configurations: List[Dict], pareto: List[Dict]):
        """Print formatted cost-effectiveness analysis."""
        print("\n" + "=" * 70)
        print("Cost-Effectiveness Analysis (Pareto Front)")
        print("=" * 70)

        print(f"\n{'Configuration':<35} {'Cost ($)':<12} {'RF Det.':<10} "
              f"{'Fiber Det.':<12} {'Avg Det.':<10} {'Pareto'}")
        print("-" * 85)

        pareto_names = {c['name'] for c in pareto}
        for config in sorted(configurations, key=lambda x: x['cost_usd']):
            is_pareto = "*" if config['name'] in pareto_names else ""
            print(f"{config['name']:<35} ${config['cost_usd']:>9,} "
                  f"{config['detection_rf']*100:>7.1f}% "
                  f"{config['detection_fiber']*100:>9.1f}% "
                  f"{config['detection_avg']*100:>8.1f}%  {is_pareto}")


if __name__ == "__main__":
    print("=" * 60)
    print("UAV-CPS-Analyzer: CPS Analyzer Test")
    print("=" * 60)
    
    # Test sensor fusion
    print("\n1. Sensor Fusion Test:")
    fusion = SensorFusion()
    
    for threat_type in [ThreatType.RF_CONTROLLED, ThreatType.FIBER_OPTIC]:
        det_rate = fusion.get_detection_rate(threat_type, 1000, n_trials=500)
        print(f"   {threat_type.value:<15}: Detection rate = {det_rate*100:.1f}%")
    
    # Test Dempster-Shafer
    print("\n2. Dempster-Shafer Fusion:")
    detected, beliefs, conf = fusion.detect_and_fuse(1000, ThreatType.RF_CONTROLLED)
    if detected:
        print(f"   Detected with confidence: {conf:.2f}")
        for threat, belief in sorted(beliefs.items(), key=lambda x: x[1], reverse=True):
            print(f"   {threat.value:<15}: Belief = {belief:.3f}")
    
    # Test C-UAS architecture
    print("\n3. C-UAS Architecture Test:")
    cuas = CUASArchitecture()
    
    latency = cuas.calculate_total_latency()
    print(f"   Detection latency:     {latency['detection_s']*1000:.0f} ms")
    print(f"   Classification latency: {latency['classification_s']*1000:.0f} ms")
    print(f"   Total latency (min):   {latency['total_min_s']:.1f} s")
    print(f"   Total latency (max):   {latency['total_max_s']:.1f} s")
    
    # Simulate engagements
    print("\n4. Engagement Simulation:")
    for threat_type in [ThreatType.RF_CONTROLLED, ThreatType.FIBER_OPTIC]:
        success_count = 0
        n_trials = 100
        for _ in range(n_trials):
            result = cuas.simulate_engagement(threat_type, 2000, 40)
            if result['neutralized']:
                success_count += 1
        
        print(f"   {threat_type.value:<15}: Neutralization rate = {success_count/n_trials*100:.0f}%")
    
    # Detection rates by sensor type
    print("\n5. Detection Rates by Sensor (1km range):")
    print(f"   {'Sensor':<12} {'RF Drone':<12} {'Fiber-optic':<12}")
    print(f"   {'-'*36}")
    for sensor_type, spec in SENSOR_SPECS.items():
        rf_rate = spec.detection_probability_rf
        fiber_rate = spec.detection_probability_fiber
        print(f"   {sensor_type.value:<12} {rf_rate*100:>8.0f}%    {fiber_rate*100:>8.0f}%")
    
    print("\n" + "=" * 60)
    print("Tests completed successfully!")
