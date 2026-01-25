#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: Propagation Models Module
Models: COST 231-Hata, Friis free-space, Rice fading, Altitude-dependent combined model

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PropagationParams:
    """Parameters for propagation model calculations."""
    frequency_mhz: float = 2437.0  # Center frequency (WiFi channel 6)
    tx_power_dbm: float = 20.0  # Transmitter power in dBm
    tx_antenna_gain_dbi: float = 2.0  # Transmitter antenna gain
    rx_antenna_gain_dbi: float = 2.0  # Receiver antenna gain
    rx_sensitivity_dbm: float = -90.0  # Receiver sensitivity
    city_size: str = "medium"  # 'small', 'medium', 'large' for COST 231
    environment: str = "urban"  # 'urban', 'suburban', 'rural'


class PropagationModel(ABC):
    """Abstract base class for propagation models."""
    
    @abstractmethod
    def path_loss(self, distance_m: float, **kwargs) -> float:
        """Calculate path loss in dB."""
        pass
    
    def received_power(self, distance_m: float, params: PropagationParams, **kwargs) -> float:
        """Calculate received power in dBm."""
        pl = self.path_loss(distance_m, frequency_mhz=params.frequency_mhz, **kwargs)
        return (params.tx_power_dbm + params.tx_antenna_gain_dbi + 
                params.rx_antenna_gain_dbi - pl)


class FriisModel(PropagationModel):
    """
    Friis free-space path loss model.
    Based on ITU-R P.525-4 recommendation.
    
    L_fs = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c)
    Simplified: L_fs = 32.45 + 20*log10(f_MHz) + 20*log10(d_km)
    """
    
    def path_loss(self, distance_m: float, frequency_mhz: float = 2437.0, **kwargs) -> float:
        """
        Calculate free-space path loss.
        
        Args:
            distance_m: Distance in meters
            frequency_mhz: Frequency in MHz
            
        Returns:
            Path loss in dB
        """
        if distance_m <= 0:
            return 0.0
        
        distance_km = distance_m / 1000.0
        # ITU-R P.525-4 formula
        loss = 32.45 + 20 * np.log10(frequency_mhz) + 20 * np.log10(distance_km)
        return max(0.0, loss)


class COST231HataModel(PropagationModel):
    """
    COST 231-Hata propagation model for urban/suburban environments.
    Valid for: 1500-2000 MHz, extended to 2400 MHz for UAV applications.
    
    Based on: COST Action 231, "Digital Mobile Radio Towards Future Generation Systems"
    """
    
    def __init__(self, city_size: str = "medium"):
        """
        Args:
            city_size: 'small', 'medium', or 'large' city
        """
        self.city_size = city_size
        # City correction factor C_m
        self.cm = 3.0 if city_size == "large" else 0.0
    
    def _mobile_antenna_correction(self, frequency_mhz: float, h_mobile: float) -> float:
        """Calculate mobile antenna height correction factor a(h_m)."""
        if self.city_size == "large" and frequency_mhz >= 400:
            return 3.2 * (np.log10(11.75 * h_mobile))**2 - 4.97
        else:
            return (1.1 * np.log10(frequency_mhz) - 0.7) * h_mobile - \
                   (1.56 * np.log10(frequency_mhz) - 0.8)
    
    def path_loss(self, distance_m: float, frequency_mhz: float = 2437.0,
                  h_base: float = 30.0, h_mobile: float = 1.5, **kwargs) -> float:
        """
        Calculate COST 231-Hata path loss.
        
        Args:
            distance_m: Distance in meters
            frequency_mhz: Frequency in MHz (1500-2000, extended to 2400)
            h_base: Base station height in meters (30-200m)
            h_mobile: Mobile station height in meters (1-10m)
            
        Returns:
            Path loss in dB
        """
        if distance_m <= 0:
            return 0.0
        
        distance_km = distance_m / 1000.0
        
        # Clamp parameters to valid ranges
        h_base = np.clip(h_base, 30, 200)
        h_mobile = np.clip(h_mobile, 1, 10)
        
        # Mobile antenna correction
        a_hm = self._mobile_antenna_correction(frequency_mhz, h_mobile)
        
        # COST 231-Hata formula
        loss = (46.3 + 33.9 * np.log10(frequency_mhz) - 13.82 * np.log10(h_base) - a_hm +
                (44.9 - 6.55 * np.log10(h_base)) * np.log10(distance_km) + self.cm)
        
        return max(0.0, loss)


class RiceFadingModel:
    """
    Rice fading model for multipath propagation.
    Based on S.O. Rice, "Mathematical Analysis of Random Noise", 1944.
    
    Models the envelope of received signal as Rice distribution when
    there is a dominant line-of-sight component.
    """
    
    def __init__(self, k_factor: float = 6.0):
        """
        Args:
            k_factor: Rice K-factor in dB (ratio of LOS to scattered power)
                     Typical values: 6-12 dB for LOS, 0 dB = Rayleigh
        """
        self.k_factor_db = k_factor
        self.k_linear = 10 ** (k_factor / 10)
    
    def generate_fading(self, n_samples: int = 1) -> np.ndarray:
        """
        Generate Rice fading samples.
        
        Args:
            n_samples: Number of fading samples to generate
            
        Returns:
            Array of fading values in dB (relative to mean)
        """
        # Rice distribution parameters
        sigma = 1.0 / np.sqrt(2 * (1 + self.k_linear))
        nu = np.sqrt(self.k_linear / (1 + self.k_linear))
        
        # Generate complex Gaussian samples
        x = nu + sigma * np.random.randn(n_samples)
        y = sigma * np.random.randn(n_samples)
        
        # Envelope (Rice distributed)
        envelope = np.sqrt(x**2 + y**2)
        
        # Convert to dB relative to mean
        fading_db = 20 * np.log10(envelope / np.mean(envelope))
        
        return fading_db
    
    def get_fading_margin(self, outage_probability: float = 0.05) -> float:
        """
        Calculate fading margin for given outage probability.
        
        Args:
            outage_probability: Desired outage probability (e.g., 0.05 for 95% reliability)
            
        Returns:
            Required fading margin in dB
        """
        # Generate large sample and find percentile
        samples = self.generate_fading(100000)
        margin = -np.percentile(samples, outage_probability * 100)
        return margin


class AltitudeDependentModel(PropagationModel):
    """
    Altitude-dependent combined propagation model.
    
    Combines COST 231-Hata (urban) and Friis (free-space) based on UAV altitude.
    Based on: Khawaja et al., "A Survey of Air-to-Ground Propagation Channel Modeling"
              Al-Hourani et al., "Optimal LAP Altitude for Maximum Coverage"
    
    Model: L(h) = L_COST * (1 - alpha) + L_Friis * alpha
    where alpha transitions from 0 (urban) to 1 (free-space) with altitude.
    """
    
    def __init__(self, h_urban: float = 100.0, h_freespace: float = 500.0,
                 city_size: str = "medium"):
        """
        Args:
            h_urban: Height below which urban model dominates (meters)
            h_freespace: Height above which free-space model dominates (meters)
            city_size: City size for COST 231 model
        """
        self.h_urban = h_urban
        self.h_freespace = h_freespace
        self.friis = FriisModel()
        self.cost231 = COST231HataModel(city_size)
        self.rice = RiceFadingModel(k_factor=6.0)
    
    def _calculate_alpha(self, altitude_m: float) -> float:
        """
        Calculate blending factor alpha based on altitude.
        
        Args:
            altitude_m: UAV altitude in meters
            
        Returns:
            Alpha value between 0 (urban) and 1 (free-space)
        """
        if altitude_m <= self.h_urban:
            return 0.0
        elif altitude_m >= self.h_freespace:
            return 1.0
        else:
            # Linear interpolation
            return (altitude_m - self.h_urban) / (self.h_freespace - self.h_urban)
    
    def _altitude_k_factor(self, altitude_m: float) -> float:
        """
        Calculate Rice K-factor based on altitude.
        Higher altitude = more LOS = higher K-factor.
        
        Args:
            altitude_m: UAV altitude in meters
            
        Returns:
            K-factor in dB
        """
        # K-factor increases with altitude (more LOS)
        k_min = 3.0  # dB at ground level
        k_max = 15.0  # dB at high altitude
        alpha = self._calculate_alpha(altitude_m)
        return k_min + alpha * (k_max - k_min)
    
    def path_loss(self, distance_m: float, altitude_m: float = 100.0,
                  frequency_mhz: float = 2437.0, include_fading: bool = False,
                  **kwargs) -> float:
        """
        Calculate altitude-dependent path loss.
        
        Args:
            distance_m: Horizontal distance in meters
            altitude_m: UAV altitude in meters
            frequency_mhz: Frequency in MHz
            include_fading: Whether to include Rice fading
            
        Returns:
            Path loss in dB
        """
        # Calculate 3D distance
        distance_3d = np.sqrt(distance_m**2 + altitude_m**2)
        
        # Get blending factor
        alpha = self._calculate_alpha(altitude_m)
        
        # Calculate both models
        l_friis = self.friis.path_loss(distance_3d, frequency_mhz)
        l_cost231 = self.cost231.path_loss(distance_m, frequency_mhz, h_base=altitude_m)
        
        # Combined model
        path_loss = l_cost231 * (1 - alpha) + l_friis * alpha
        
        # Add fading if requested
        if include_fading:
            k_factor = self._altitude_k_factor(altitude_m)
            self.rice.k_factor_db = k_factor
            self.rice.k_linear = 10 ** (k_factor / 10)
            fading = self.rice.generate_fading(1)[0]
            path_loss += abs(fading)  # Fading adds to loss
        
        return path_loss
    
    def effective_range(self, tx_power_dbm: float, rx_sensitivity_dbm: float,
                        altitude_m: float, frequency_mhz: float = 2437.0,
                        margin_db: float = 10.0) -> float:
        """
        Calculate effective communication range at given altitude.
        
        Args:
            tx_power_dbm: Transmitter power in dBm
            rx_sensitivity_dbm: Receiver sensitivity in dBm
            altitude_m: UAV altitude in meters
            frequency_mhz: Frequency in MHz
            margin_db: Link margin in dB
            
        Returns:
            Maximum horizontal range in meters
        """
        # Available link budget
        link_budget = tx_power_dbm - rx_sensitivity_dbm - margin_db
        
        # Binary search for range
        d_min, d_max = 10, 50000
        while d_max - d_min > 10:
            d_mid = (d_min + d_max) / 2
            pl = self.path_loss(d_mid, altitude_m, frequency_mhz)
            if pl < link_budget:
                d_min = d_mid
            else:
                d_max = d_mid
        
        return d_min


class ShadowFading:
    """
    Log-normal shadow fading model.
    Models large-scale fading due to obstacles and terrain.
    """
    
    def __init__(self, sigma_db: float = 8.0):
        """
        Args:
            sigma_db: Standard deviation of shadow fading in dB
                     Typical: 4-12 dB depending on environment
        """
        self.sigma_db = sigma_db
    
    def generate_fading(self, n_samples: int = 1) -> np.ndarray:
        """Generate shadow fading samples in dB."""
        return np.random.randn(n_samples) * self.sigma_db


def calculate_js_ratio(jammer_power_dbm: float, jammer_distance_m: float,
                       jammer_antenna_gain_dbi: float,
                       signal_power_dbm: float, signal_distance_m: float,
                       signal_antenna_gain_dbi: float,
                       altitude_m: float = 100.0,
                       frequency_mhz: float = 2437.0) -> float:
    """
    Calculate Jamming-to-Signal (J/S) ratio.
    
    Args:
        jammer_power_dbm: Jammer transmit power in dBm
        jammer_distance_m: Distance from jammer to target in meters
        jammer_antenna_gain_dbi: Jammer antenna gain in dBi
        signal_power_dbm: Legitimate signal power in dBm
        signal_distance_m: Distance from transmitter to target in meters
        signal_antenna_gain_dbi: Signal antenna gain in dBi
        altitude_m: Target altitude in meters
        frequency_mhz: Operating frequency in MHz
        
    Returns:
        J/S ratio in dB
    """
    model = AltitudeDependentModel()
    
    # Jammer received power at target
    jammer_pl = model.path_loss(jammer_distance_m, altitude_m, frequency_mhz)
    jammer_rx = jammer_power_dbm + jammer_antenna_gain_dbi - jammer_pl
    
    # Signal received power at target
    signal_pl = model.path_loss(signal_distance_m, altitude_m, frequency_mhz)
    signal_rx = signal_power_dbm + signal_antenna_gain_dbi - signal_pl
    
    # J/S ratio
    js_ratio = jammer_rx - signal_rx
    
    return js_ratio


# Convenience functions
def get_model(model_type: str, **kwargs) -> PropagationModel:
    """
    Factory function to get propagation model by name.
    
    Args:
        model_type: 'friis', 'cost231', or 'altitude'
        **kwargs: Model-specific parameters
        
    Returns:
        PropagationModel instance
    """
    models = {
        'friis': FriisModel,
        'cost231': COST231HataModel,
        'altitude': AltitudeDependentModel
    }
    
    if model_type.lower() not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type.lower()](**kwargs)


if __name__ == "__main__":
    # Test the models
    print("=" * 60)
    print("UAV-CPS-Analyzer: Propagation Models Test")
    print("=" * 60)
    
    # Test parameters
    distances = [100, 500, 1000, 2000, 5000]
    altitudes = [50, 100, 200, 500]
    freq = 2437  # MHz
    
    print("\n1. Friis Free-Space Model:")
    friis = FriisModel()
    for d in distances:
        pl = friis.path_loss(d, freq)
        print(f"   Distance {d:5d}m: Path Loss = {pl:.1f} dB")
    
    print("\n2. COST 231-Hata Model:")
    cost231 = COST231HataModel("medium")
    for d in distances:
        pl = cost231.path_loss(d, freq)
        print(f"   Distance {d:5d}m: Path Loss = {pl:.1f} dB")
    
    print("\n3. Altitude-Dependent Model:")
    alt_model = AltitudeDependentModel()
    for h in altitudes:
        for d in [500, 2000]:
            pl = alt_model.path_loss(d, h, freq)
            alpha = alt_model._calculate_alpha(h)
            print(f"   h={h:3d}m, d={d:4d}m: PL={pl:.1f} dB (α={alpha:.2f})")
    
    print("\n4. J/S Ratio Calculation:")
    js = calculate_js_ratio(
        jammer_power_dbm=40,  # 10W
        jammer_distance_m=500,
        jammer_antenna_gain_dbi=6,
        signal_power_dbm=20,  # 100mW
        signal_distance_m=5000,
        signal_antenna_gain_dbi=2,
        altitude_m=100
    )
    print(f"   J/S ratio: {js:.1f} dB")
    
    print("\n" + "=" * 60)
    print("Tests completed successfully!")
