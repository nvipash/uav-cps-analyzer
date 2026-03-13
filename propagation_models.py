#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: Propagation Models Module
Models: COST 231-Hata, Friis free-space, Rice fading, Altitude-dependent combined model

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

import numpy as np
from scipy.special import erfc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


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
    Altitude-dependent combined propagation model for UAV A2G scenarios.

    Combines Al-Hourani A2G (low altitude) and Friis (free-space) based on UAV
    altitude. Al-Hourani (2014) is used for the urban/ground component because it
    is specifically designed for Air-to-Ground propagation and correctly handles
    UAV altitude via LOS probability weighting.

    COST 231-Hata is NOT used here for the A2G component because its valid domain
    (h_base=30-200 m as terrestrial base station, h_mobile=1-10 m as handset,
    f=1500-2000 MHz) does not apply to UAV elevation scenarios. It remains
    available via COST231HataModel for purely terrestrial link calculations.

    Model: L(h) = L_A2G * (1 - alpha) + L_Friis * alpha
    where alpha transitions 0→1 linearly between h_urban and h_freespace.

    References:
    - Al-Hourani et al., "Optimal LAP Altitude for Maximum Coverage", IEEE WCL 2014
    - Khawaja et al., "A Survey of A2G Propagation Channel Modeling", IEEE COMST 2019
    """

    def __init__(self, h_urban: float = 100.0, h_freespace: float = 500.0,
                 city_size: str = "medium", environment: str = "urban"):
        """
        Args:
            h_urban:      Altitude below which A2G model dominates (m)
            h_freespace:  Altitude above which free-space model dominates (m)
            city_size:    Retained for API compatibility (unused in A2G path)
            environment:  Propagation environment for Al-Hourani model
        """
        self.h_urban = h_urban
        self.h_freespace = h_freespace
        self.friis = FriisModel()
        self.cost231 = COST231HataModel(city_size)  # terrestrial use only
        # Map 'open_field' → 'rural' since AlHouraniA2GModel defines 4 environments
        a2g_env = environment if environment in AlHouraniA2GModel.ENVIRONMENT_PARAMS else 'rural'
        self.a2g = AlHouraniA2GModel(a2g_env)
        self.rice = RiceFadingModel(k_factor=6.0)
        self.environment = environment
    
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
        Rice K-factor as a function of altitude.
        Higher altitude → more LOS → higher K-factor.

        Range calibrated to Khawaja (2019) Table VI (arXiv:1801.01656):
          Urban 2.4 GHz interpolated: K ≈ 18 dB at h=50m, 22 dB at h=200m.
          Lower bound reflects dense-urban near-ground (Ref[47]: K=-5..10 dB).
        """
        k_min = 10.0   # dB near ground — Khawaja (2019) Table VI Ref[47] upper end
        k_max = 24.0   # dB at free-space altitude — Khawaja Table VI Ref[54] extrapolated
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
        
        # A2G component: Al-Hourani model (correct domain for UAV altitude scenarios)
        l_friis = self.friis.path_loss(distance_3d, frequency_mhz)
        l_a2g = self.a2g.path_loss(distance_m, altitude_m, frequency_mhz)

        # Blended model: A2G at low altitude, Friis at high altitude
        # (Al-Hourani already approaches Friis at high elevations, so this
        # blending adds conservative consistency at extreme altitudes)
        path_loss = l_a2g * (1 - alpha) + l_friis * alpha
        
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


class AtmosphericAbsorption:
    """
    Atmospheric absorption losses based on ITU-R P.676.
    Significant for long-range scenarios (>5 km) at 2.4/5.8 GHz.

    Reference: ITU-R P.676-12 (Attenuation by atmospheric gases and related effects)
    """

    @staticmethod
    def attenuation_db_per_km(frequency_mhz: float,
                               temperature_c: float = 15.0,
                               humidity_percent: float = 50.0,
                               pressure_hpa: float = 1013.0) -> float:
        """
        Calculate atmospheric attenuation per km.

        Simplified model for 1-100 GHz based on ITU-R P.676.

        Args:
            frequency_mhz: Frequency in MHz
            temperature_c: Temperature in Celsius
            humidity_percent: Relative humidity (0-100)
            pressure_hpa: Atmospheric pressure (hPa)

        Returns:
            Attenuation in dB/km
        """
        f_ghz = frequency_mhz / 1000.0

        # Water vapor density (g/m^3) - simplified Magnus formula
        T_k = temperature_c + 273.15
        es = 6.1121 * np.exp((17.502 * temperature_c) / (240.97 + temperature_c))
        e = es * humidity_percent / 100.0
        rho_water = 216.7 * e / T_k  # g/m^3

        # Oxygen absorption (dominant below 30 GHz)
        if f_ghz < 57:
            gamma_o = (7.2 * f_ghz**2 / (f_ghz**2 + 0.34) +
                       0.62 * f_ghz**2 / ((54 - f_ghz)**1.16 + 0.83)) * 1e-3
        else:
            gamma_o = 0.0

        # Water vapor absorption
        gamma_w = (3.98 * np.exp(-((f_ghz - 22.235) / 9.42)**2) +
                   11.96 * np.exp(-((f_ghz - 183.31) / 11.14)**2) +
                   0.081 * f_ghz**2) * rho_water * 1e-4

        return float(gamma_o + gamma_w)

    @staticmethod
    def total_loss_db(distance_m: float, frequency_mhz: float,
                       temperature_c: float = 15.0,
                       humidity_percent: float = 50.0) -> float:
        """Total atmospheric absorption over given distance."""
        gamma = AtmosphericAbsorption.attenuation_db_per_km(
            frequency_mhz, temperature_c, humidity_percent
        )
        return gamma * distance_m / 1000.0


class UrbanMultiPathCorrection:
    """
    Urban multi-path correction for long-range NLOS scenarios.
    Models additional losses from multiple reflections, scattering, and diffraction.

    Based on:
    - ITU-R P.1411-12 (2019) §4.2.1 dual-slope outdoor path loss model
    - Saunders & Aragón-Zavala, "Antennas and Propagation for Wireless
      Communication Systems", 2nd ed. (2007), §4.5 (Fresnel breakpoint)
    - Al-Hourani et al. (2014) for altitude-dependent LOS correction
    """

    @staticmethod
    def multi_path_loss_db(distance_m: float, environment: str = 'urban',
                            altitude_m: float = 100.0,
                            h_jammer_m: float = 2.0) -> float:
        """
        Additional NLOS loss beyond Friis due to multipath effects.

        Uses the ITU-R P.1411-12 dual-slope model with published path-loss
        exponents (n1 below Fresnel breakpoint, n2 above). The Fresnel
        breakpoint d_BP = 4·h_tx·h_rx·f/c (Saunders §4.5) depends on antenna
        heights and frequency. The excess over free-space (Friis) is returned
        so the correction composes correctly with the base Al-Hourani model.

        Args:
            distance_m:  Distance in meters
            environment: dense_urban | urban | suburban | rural
            altitude_m:  UAV altitude in meters
            h_jammer_m:  Jammer antenna height above ground in meters (default 2 m
                         for ground-mounted portable system; use higher values for
                         vehicle-mounted or elevated jammers)

        Returns:
            Additional loss in dB (always ≥ 0)
        """
        d_km = max(distance_m / 1000.0, 0.01)

        # Dual-slope path-loss exponents for A2G residual correction at 2.4 GHz
        # (beyond the Al-Hourani base model).
        # n1: exponent below Fresnel breakpoint; n2: exponent above breakpoint.
        # Derived from WINNER+ D1.1.2 (2008) Table 4-2 UMa/UMi NLOS exponents,
        # scaled to 2.4 GHz residual role. Ordering is physically correct:
        # dense_urban > urban > suburban > rural (more clutter → more excess loss).
        ITU_COEFFS = {
            'dense_urban': {'n1': 2.25, 'n2': 3.75, 'sigma': 5.5},
            'urban':       {'n1': 2.18, 'n2': 3.48, 'sigma': 4.6},
            'suburban':    {'n1': 2.10, 'n2': 2.75, 'sigma': 4.5},
            'rural':       {'n1': 2.00, 'n2': 2.30, 'sigma': 3.0},
        }
        c = ITU_COEFFS.get(environment, ITU_COEFFS['urban'])

        # Fresnel breakpoint: d_BP = 4·h_jammer·h_uav·f/c (Saunders §4.5)
        freq_ghz = 2.437       # 2.4 GHz ISM band centre
        d_break_km = 4.0 * h_jammer_m * max(altitude_m, 1.0) * freq_ghz / 300.0

        # Reference free-space loss at 1 m (ITU-R P.525-4 at freq_ghz)
        L_ref_db = 32.45 + 20.0 * np.log10(freq_ghz * 1000.0) - 60.0

        # Dual-slope total path loss (ITU-R P.1411-12 eq. 1)
        if d_km <= d_break_km:
            L_total_db = L_ref_db + 10.0 * c['n1'] * np.log10(d_km / 0.001)
        else:
            L_break_db = L_ref_db + 10.0 * c['n1'] * np.log10(d_break_km / 0.001)
            L_total_db = L_break_db + 10.0 * c['n2'] * np.log10(d_km / d_break_km)

        # Friis baseline at the same distance (ITU-R P.525-4)
        L_friis_db = 32.45 + 20.0 * np.log10(freq_ghz * 1000.0) + 20.0 * np.log10(d_km)

        # Excess loss = multipath correction (always ≥ 0 by physics)
        excess_loss_db = max(0.0, L_total_db - L_friis_db)

        # Altitude correction: P_LOS → 1 as altitude ↑, reducing multipath
        # (Al-Hourani 2014: elevation angle increases → fewer obstructions)
        altitude_factor = max(0.1, 1.0 - min(0.9, altitude_m / 500.0))

        return max(0.0, excess_loss_db * altitude_factor)


class ModulationType(Enum):
    """Modulation schemes for BER calculation."""
    BPSK = "bpsk"
    QPSK = "qpsk"
    QAM16 = "16qam"
    QAM64 = "64qam"


class BERModel:
    """
    Bit Error Rate and Packet Error Rate models.
    Maps J/S ratio to communication link quality.

    Based on:
    - Proakis, "Digital Communications", 5th ed., 2007
    - IEEE 802.11 OFDM physical layer specifications
    """

    @staticmethod
    def ber(sinr_db: float, modulation: ModulationType = ModulationType.QPSK) -> float:
        """
        Calculate Bit Error Rate for given SINR and modulation.

        SINR = S / (N + J), where the J/S ratio determines the interference.
        For jamming analysis: SINR_db ≈ -J/S_db (when jammer dominates noise).

        Args:
            sinr_db: Signal-to-Interference-plus-Noise Ratio in dB
            modulation: Modulation type

        Returns:
            Bit Error Rate (0-0.5)
        """
        sinr_linear = 10 ** (sinr_db / 10)
        sinr_linear = max(sinr_linear, 1e-10)

        if modulation == ModulationType.BPSK:
            return 0.5 * erfc(np.sqrt(sinr_linear))
        elif modulation == ModulationType.QPSK:
            return 0.5 * erfc(np.sqrt(sinr_linear / 2))
        elif modulation == ModulationType.QAM16:
            # Approximate BER for 16-QAM
            return (3 / 8) * erfc(np.sqrt(sinr_linear / 10))
        elif modulation == ModulationType.QAM64:
            # Approximate BER for 64-QAM
            return (7 / 24) * erfc(np.sqrt(sinr_linear / 42))
        return 0.5

    @staticmethod
    def packet_error_rate(ber: float, packet_length_bits: int = 8192) -> float:
        """
        Calculate Packet Error Rate from BER.
        PER = 1 - (1 - BER)^L

        Args:
            ber: Bit Error Rate
            packet_length_bits: Packet length in bits (default 8192 = 1KB)

        Returns:
            Packet Error Rate (0-1)
        """
        if ber <= 0:
            return 0.0
        if ber >= 0.5:
            return 1.0
        return 1.0 - (1.0 - ber) ** packet_length_bits

    @staticmethod
    def jamming_success_probability(js_ratio_db: float,
                                    noise_floor_db: float = -100.0,
                                    signal_power_db: float = -70.0,
                                    modulation: ModulationType = ModulationType.QPSK,
                                    packet_length_bits: int = 8192) -> float:
        """
        Calculate probability of successful jamming (communication disruption)
        based on BER/PER model instead of hard threshold.

        Args:
            js_ratio_db: Jamming-to-Signal ratio in dB
            noise_floor_db: Receiver noise floor in dBm
            signal_power_db: Received signal power in dBm
            modulation: Modulation type used by the link
            packet_length_bits: Packet length in bits

        Returns:
            Probability of successful jamming (0-1)
        """
        # SINR = S / (N + J) in dB
        # J = S + J/S (in dB domain: jammer_power = signal_power + js_ratio)
        jammer_power_linear = 10 ** ((signal_power_db + js_ratio_db) / 10)
        noise_linear = 10 ** (noise_floor_db / 10)
        signal_linear = 10 ** (signal_power_db / 10)

        sinr_linear = signal_linear / (noise_linear + jammer_power_linear)
        sinr_db = 10 * np.log10(max(sinr_linear, 1e-20))

        ber_val = BERModel.ber(sinr_db, modulation)
        per_val = BERModel.packet_error_rate(ber_val, packet_length_bits)

        return per_val


class DopplerModel:
    """
    Doppler effect model for UAV communications.

    Based on:
    - Rappaport, "Wireless Communications", 2nd ed., 2002
    - Khawaja et al., "A Survey of Air-to-Ground Propagation Channel Modeling"
    """

    SPEED_OF_LIGHT = 3e8  # m/s

    @staticmethod
    def doppler_shift_hz(frequency_mhz: float, velocity_ms: float,
                         angle_deg: float = 0.0) -> float:
        """
        Calculate Doppler frequency shift.

        f_d = (v / c) * f * cos(theta)

        Args:
            frequency_mhz: Carrier frequency in MHz
            velocity_ms: Relative velocity in m/s
            angle_deg: Angle between velocity vector and propagation direction

        Returns:
            Doppler shift in Hz
        """
        f_hz = frequency_mhz * 1e6
        return (velocity_ms / DopplerModel.SPEED_OF_LIGHT) * f_hz * np.cos(np.radians(angle_deg))

    @staticmethod
    def max_doppler_spread_hz(frequency_mhz: float, velocity_ms: float) -> float:
        """
        Calculate maximum Doppler spread (worst case, angle = 0).

        Args:
            frequency_mhz: Carrier frequency in MHz
            velocity_ms: Velocity in m/s

        Returns:
            Maximum Doppler spread in Hz
        """
        f_hz = frequency_mhz * 1e6
        return velocity_ms * f_hz / DopplerModel.SPEED_OF_LIGHT

    @staticmethod
    def coherence_time_s(frequency_mhz: float, velocity_ms: float) -> float:
        """
        Calculate channel coherence time.
        T_c ≈ 0.423 / f_d_max

        Args:
            frequency_mhz: Carrier frequency in MHz
            velocity_ms: Velocity in m/s

        Returns:
            Coherence time in seconds
        """
        f_d_max = DopplerModel.max_doppler_spread_hz(frequency_mhz, velocity_ms)
        if f_d_max < 1e-6:
            return float('inf')
        return 0.423 / f_d_max

    @staticmethod
    def hop_sync_degradation(frequency_mhz: float, velocity_ms: float,
                             channel_bandwidth_mhz: float = 2.0,
                             degradation_threshold: float = 0.1) -> float:
        """
        Calculate FHSS hop synchronization degradation probability.

        When Doppler shift > threshold fraction of channel bandwidth,
        there is a probability of synchronization failure.

        Args:
            frequency_mhz: Carrier frequency in MHz
            velocity_ms: Velocity in m/s
            channel_bandwidth_mhz: Channel bandwidth in MHz
            degradation_threshold: Fraction of bandwidth that causes issues

        Returns:
            Probability of hop sync degradation (0-1)
        """
        f_d = abs(DopplerModel.max_doppler_spread_hz(frequency_mhz, velocity_ms))
        bw_hz = channel_bandwidth_mhz * 1e6
        ratio = f_d / bw_hz

        if ratio < degradation_threshold:
            return 0.0
        # Soft degradation above threshold
        return min(1.0, (ratio - degradation_threshold) / (1.0 - degradation_threshold))


class AlHouraniA2GModel(PropagationModel):
    """
    Al-Hourani et al. probabilistic Air-to-Ground propagation model.

    Based on:
    - Al-Hourani et al., "Optimal LAP Altitude for Maximum Coverage", IEEE WCL, 2014
    - ITU-R P.1410 (propagation for aeronautical mobile systems)

    The model computes LOS probability as a function of elevation angle
    and environment-dependent parameters, then weights LOS and NLOS path losses.

    P_LOS(theta) = 1 / (1 + a * exp(-b * (theta - a)))
    L = P_LOS * L_LOS + (1 - P_LOS) * L_NLOS
    """

    # Environment-dependent parameters from Al-Hourani et al., Table II
    ENVIRONMENT_PARAMS = {
        'dense_urban': {'a': 12.08, 'b': 0.11, 'eta_los': 1.6, 'eta_nlos': 23.0},
        'urban':       {'a': 9.61,  'b': 0.16, 'eta_los': 1.0, 'eta_nlos': 20.0},
        'suburban':    {'a': 4.88,  'b': 0.43, 'eta_los': 0.1, 'eta_nlos': 21.0},
        'rural':       {'a': 2.00,  'b': 0.60, 'eta_los': 0.1, 'eta_nlos': 20.0},
    }

    def __init__(self, environment: str = 'urban'):
        """
        Args:
            environment: 'dense_urban', 'urban', 'suburban', or 'rural'
        """
        if environment not in self.ENVIRONMENT_PARAMS:
            raise ValueError(f"Unknown environment: {environment}. "
                             f"Choose from: {list(self.ENVIRONMENT_PARAMS.keys())}")
        self.environment = environment
        self.env_params = self.ENVIRONMENT_PARAMS[environment]
        self.friis = FriisModel()

    def los_probability(self, altitude_m: float, horizontal_distance_m: float) -> float:
        """
        Calculate Line-of-Sight probability.

        P_LOS = 1 / (1 + a * exp(-b * (theta_deg - a)))

        Args:
            altitude_m: UAV altitude in meters
            horizontal_distance_m: Horizontal distance in meters

        Returns:
            LOS probability (0-1)
        """
        if horizontal_distance_m <= 0:
            return 1.0
        theta_deg = np.degrees(np.arctan(altitude_m / horizontal_distance_m))
        a = self.env_params['a']
        b = self.env_params['b']
        return 1.0 / (1.0 + a * np.exp(-b * (theta_deg - a)))

    def path_loss(self, distance_m: float, altitude_m: float = 100.0,
                  frequency_mhz: float = 2437.0, **kwargs) -> float:
        """
        Calculate probabilistic A2G path loss.

        L = P_LOS * L_LOS + (1 - P_LOS) * L_NLOS
        L_LOS  = L_fs + eta_LOS
        L_NLOS = L_fs + eta_NLOS

        Args:
            distance_m: Horizontal distance in meters
            altitude_m: UAV altitude in meters
            frequency_mhz: Frequency in MHz

        Returns:
            Expected path loss in dB
        """
        distance_3d = np.sqrt(distance_m ** 2 + altitude_m ** 2)
        if distance_3d <= 0:
            return 0.0

        l_fs = self.friis.path_loss(distance_3d, frequency_mhz)
        p_los = self.los_probability(altitude_m, distance_m)

        l_los = l_fs + self.env_params['eta_los']
        l_nlos = l_fs + self.env_params['eta_nlos']

        return p_los * l_los + (1 - p_los) * l_nlos


class AntennaPattern(ABC):
    """Abstract base class for antenna radiation patterns."""

    @abstractmethod
    def gain_dbi(self, theta_deg: float, phi_deg: float = 0.0) -> float:
        """
        Calculate antenna gain at given angles.

        Args:
            theta_deg: Off-boresight angle in degrees (elevation)
            phi_deg: Azimuth angle in degrees

        Returns:
            Gain in dBi
        """
        pass


class OmnidirectionalPattern(AntennaPattern):
    """Omnidirectional antenna (isotropic approximation)."""

    def __init__(self, nominal_gain_dbi: float = 2.0):
        self.nominal_gain_dbi = nominal_gain_dbi

    def gain_dbi(self, theta_deg: float, phi_deg: float = 0.0) -> float:
        return self.nominal_gain_dbi


class CosinePattern(AntennaPattern):
    """
    Cosine-taper antenna pattern (common approximation for directional antennas).

    G(theta) = G_max * cos^n(theta)  for |theta| <= 90 deg
    where n is chosen to match the specified -3dB beamwidth.
    """

    def __init__(self, beam_width_deg: float = 60.0, max_gain_dbi: float = 6.0):
        """
        Args:
            beam_width_deg: -3dB beamwidth in degrees
            max_gain_dbi: Maximum (boresight) gain in dBi
        """
        self.beam_width_deg = beam_width_deg
        self.max_gain_dbi = max_gain_dbi
        # Solve for exponent n: cos^n(BW/2) = 0.5 => n = log(0.5) / log(cos(BW/2))
        half_bw_rad = np.radians(beam_width_deg / 2)
        cos_half = np.cos(half_bw_rad)
        if cos_half > 0 and cos_half < 1:
            self.exponent = np.log(0.5) / np.log(cos_half)
        else:
            self.exponent = 1.0

    def gain_dbi(self, theta_deg: float, phi_deg: float = 0.0) -> float:
        if abs(theta_deg) >= 90:
            return -30.0  # Back-lobe suppression
        cos_theta = np.cos(np.radians(theta_deg))
        if cos_theta <= 0:
            return -30.0
        relative_gain_db = 10 * self.exponent * np.log10(cos_theta)
        return self.max_gain_dbi + relative_gain_db


class ShadowFading:
    """
    Shadow fading model with optional heavy-tailed distributions.
    Standard log-normal (Gaussian in dB) for typical scenarios,
    Student's t for heavy-tailed extreme events.
    """

    def __init__(self, sigma_db: float = 8.0, distribution: str = 'normal',
                 df: float = 5.0):
        """
        Args:
            sigma_db: Standard deviation of shadow fading in dB (4-12 typical)
            distribution: 'normal' | 'student_t' | 'mixture'
            df: degrees of freedom for Student's t (lower = heavier tails)
        """
        self.sigma_db = sigma_db
        self.distribution = distribution
        self.df = df

    def generate_fading(self, n_samples: int = 1) -> np.ndarray:
        """Generate shadow fading samples in dB using selected distribution."""
        if self.distribution == 'normal':
            return np.random.randn(n_samples) * self.sigma_db
        elif self.distribution == 'student_t':
            from scipy.stats import t
            # Scale Student's t to match desired std
            samples = t.rvs(self.df, size=n_samples)
            scale = self.sigma_db / np.sqrt(self.df / (self.df - 2))
            return samples * scale
        elif self.distribution == 'mixture':
            # 90% normal + 10% extreme events with 3x larger std
            mask = np.random.random(n_samples) < 0.9
            samples = np.zeros(n_samples)
            samples[mask] = np.random.randn(np.sum(mask)) * self.sigma_db
            samples[~mask] = np.random.randn(np.sum(~mask)) * self.sigma_db * 3.0
            return samples
        else:
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
