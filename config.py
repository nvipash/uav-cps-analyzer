#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: Configuration Module
System parameters, UAV database, and default settings.

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import json
import os


class DroneType(Enum):
    """Types of commercial drones."""
    DJI_MAVIC_3 = "dji_mavic_3"
    DJI_MINI_4 = "dji_mini_4"
    DJI_FPV = "dji_fpv"
    ANALOG_FPV = "analog_fpv"
    AUTEL_EVO = "autel_evo"
    PARROT_ANAFI = "parrot_anafi"
    CUSTOM_FIBER = "fiber_optic"


@dataclass
class DroneSpecification:
    """Specifications for a drone model."""
    name: str
    manufacturer: str
    
    # Communication
    frequency_mhz: float = 2437.0  # Primary frequency
    frequency_band: str = "2.4GHz"  # Band
    tx_power_dbm: float = 20.0     # Transmit power
    protocol: str = "OcuSync"       # Communication protocol
    fhss_channels: int = 40        # Number of FHSS channels (0 = no FHSS)
    fhss_hop_rate_hz: float = 500  # Hop rate
    
    # Performance
    max_range_m: float = 15000     # Maximum range
    max_speed_ms: float = 20.0     # Maximum speed
    max_altitude_m: float = 500    # Maximum altitude
    
    # Physical
    weight_kg: float = 0.9
    rcs_m2: float = 0.01          # Radar cross section
    acoustic_signature_db: float = 65  # Noise level at 1m
    
    # Antenna
    antenna_gain_dbi: float = 2.0
    antenna_pattern: str = "omnidirectional"


# Database of drone specifications
DRONE_DATABASE: Dict[DroneType, DroneSpecification] = {
    DroneType.DJI_MAVIC_3: DroneSpecification(
        name="Mavic 3",
        manufacturer="DJI",
        frequency_mhz=2437,
        frequency_band="2.4/5.8 GHz",
        tx_power_dbm=20,
        protocol="OcuSync 3+",
        fhss_channels=40,
        fhss_hop_rate_hz=500,
        max_range_m=15000,
        max_speed_ms=21,
        max_altitude_m=6000,
        weight_kg=0.895,
        rcs_m2=0.01,
        acoustic_signature_db=68
    ),
    DroneType.DJI_MINI_4: DroneSpecification(
        name="Mini 4 Pro",
        manufacturer="DJI",
        frequency_mhz=2437,
        frequency_band="2.4/5.8 GHz",
        tx_power_dbm=18,
        protocol="OcuSync 4",
        fhss_channels=40,
        fhss_hop_rate_hz=500,
        max_range_m=20000,
        max_speed_ms=16,
        max_altitude_m=4000,
        weight_kg=0.249,
        rcs_m2=0.005,
        acoustic_signature_db=60
    ),
    DroneType.DJI_FPV: DroneSpecification(
        name="FPV Drone",
        manufacturer="DJI",
        frequency_mhz=5800,
        frequency_band="5.8 GHz",
        tx_power_dbm=25,
        protocol="OcuSync 3",
        fhss_channels=40,
        fhss_hop_rate_hz=500,
        max_range_m=10000,
        max_speed_ms=40,  # 144 km/h
        max_altitude_m=6000,
        weight_kg=0.795,
        rcs_m2=0.015,
        acoustic_signature_db=75
    ),
    DroneType.ANALOG_FPV: DroneSpecification(
        name="Analog FPV (Custom)",
        manufacturer="Various",
        frequency_mhz=5800,
        frequency_band="5.8 GHz",
        tx_power_dbm=25,
        protocol="Analog",
        fhss_channels=0,  # No FHSS
        fhss_hop_rate_hz=0,
        max_range_m=2000,
        max_speed_ms=50,
        max_altitude_m=500,
        weight_kg=0.5,
        rcs_m2=0.02,
        acoustic_signature_db=80
    ),
    DroneType.CUSTOM_FIBER: DroneSpecification(
        name="Fiber-Optic UAV",
        manufacturer="Custom",
        frequency_mhz=0,  # No RF
        frequency_band="Fiber-Optic",
        tx_power_dbm=0,
        protocol="Ethernet-over-Fiber",
        fhss_channels=0,
        fhss_hop_rate_hz=0,
        max_range_m=30000,  # Fiber length limited
        max_speed_ms=30,
        max_altitude_m=500,
        weight_kg=2.0,
        rcs_m2=0.02,
        acoustic_signature_db=70
    )
}


@dataclass
class JammerSpecification:
    """Specifications for a jamming system."""
    name: str
    type: str  # portable, mobile, stationary
    
    # Power
    power_w: float
    power_dbm: float = field(init=False)
    
    # Frequency
    frequency_bands: List[str] = field(default_factory=lambda: ["2.4GHz", "5.8GHz"])
    bandwidth_mhz: float = 100  # Total bandwidth
    
    # Antenna
    antenna_gain_dbi: float = 6.0
    antenna_type: str = "directional"
    beam_width_deg: float = 60
    
    # Physical
    weight_kg: float = 5.0
    battery_hours: float = 2.0
    
    # Cost
    cost_usd: float = 5000
    
    def __post_init__(self):
        self.power_dbm = 10 * (3 + __import__('math').log10(self.power_w))  # W to dBm


# Database of jammer systems
JAMMER_DATABASE = {
    "portable_10w": JammerSpecification(
        name="Portable System",
        type="portable",
        power_w=10,
        antenna_gain_dbi=6,
        antenna_type="directional",
        beam_width_deg=60,
        weight_kg=5,
        battery_hours=2,
        cost_usd=5000
    ),
    "mobile_100w": JammerSpecification(
        name="Mobile System",
        type="mobile",
        power_w=100,
        antenna_gain_dbi=10,
        antenna_type="directional",
        beam_width_deg=30,
        weight_kg=50,
        battery_hours=8,
        cost_usd=50000
    ),
    "stationary_500w": JammerSpecification(
        name="Stationary System",
        type="stationary",
        power_w=500,
        antenna_gain_dbi=15,
        antenna_type="phased_array",
        beam_width_deg=10,
        weight_kg=200,
        battery_hours=24,  # Grid powered
        cost_usd=200000
    )
}


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""
    n_iterations: int = 10000
    n_processes: int = -1  # -1 = auto (CPU count - 1)
    random_seed: Optional[int] = None
    
    # Output
    output_dir: str = "output"
    save_figures: bool = True
    figure_format: str = "png"  # png, pdf, svg
    
    # Confidence intervals
    ci_level: float = 0.95  # 95% CI
    
    # Convergence
    check_convergence: bool = True
    convergence_threshold: float = 0.01  # 1% change


@dataclass
class PropagationConfig:
    """Configuration for propagation models."""
    # Altitude thresholds
    h_urban: float = 100.0      # Below: urban model
    h_freespace: float = 500.0  # Above: free-space model
    
    # COST 231 parameters
    city_size: str = "medium"   # small, medium, large
    
    # Fading parameters
    shadow_fading_std_db: float = 8.0
    rice_k_factor_db: float = 6.0
    
    # Frequency
    default_frequency_mhz: float = 2437.0


@dataclass 
class FHSSConfig:
    """Configuration for FHSS emulation."""
    # OcuSync defaults
    n_channels: int = 40
    hop_rate_hz: float = 500
    band_start_mhz: float = 2400
    band_end_mhz: float = 2483.5
    
    # LFSR parameters
    lfsr_seed: int = 0x1234
    lfsr_bits: int = 16


class Config:
    """
    Main configuration class.
    Loads and manages all configuration settings.
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to JSON config file (optional)
        """
        self.simulation = SimulationConfig()
        self.propagation = PropagationConfig()
        self.fhss = FHSSConfig()
        
        if config_file and os.path.exists(config_file):
            self.load(config_file)
    
    def load(self, config_file: str):
        """Load configuration from JSON file."""
        with open(config_file, 'r') as f:
            data = json.load(f)
        
        if 'simulation' in data:
            for key, value in data['simulation'].items():
                if hasattr(self.simulation, key):
                    setattr(self.simulation, key, value)
        
        if 'propagation' in data:
            for key, value in data['propagation'].items():
                if hasattr(self.propagation, key):
                    setattr(self.propagation, key, value)
        
        if 'fhss' in data:
            for key, value in data['fhss'].items():
                if hasattr(self.fhss, key):
                    setattr(self.fhss, key, value)
    
    def save(self, config_file: str):
        """Save configuration to JSON file."""
        data = {
            'simulation': {
                'n_iterations': self.simulation.n_iterations,
                'n_processes': self.simulation.n_processes,
                'random_seed': self.simulation.random_seed,
                'output_dir': self.simulation.output_dir,
                'save_figures': self.simulation.save_figures,
                'figure_format': self.simulation.figure_format,
                'ci_level': self.simulation.ci_level
            },
            'propagation': {
                'h_urban': self.propagation.h_urban,
                'h_freespace': self.propagation.h_freespace,
                'city_size': self.propagation.city_size,
                'shadow_fading_std_db': self.propagation.shadow_fading_std_db,
                'rice_k_factor_db': self.propagation.rice_k_factor_db,
                'default_frequency_mhz': self.propagation.default_frequency_mhz
            },
            'fhss': {
                'n_channels': self.fhss.n_channels,
                'hop_rate_hz': self.fhss.hop_rate_hz,
                'band_start_mhz': self.fhss.band_start_mhz,
                'band_end_mhz': self.fhss.band_end_mhz
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_drone(self, drone_type: DroneType) -> DroneSpecification:
        """Get drone specification by type."""
        return DRONE_DATABASE.get(drone_type)
    
    def get_jammer(self, jammer_id: str) -> JammerSpecification:
        """Get jammer specification by ID."""
        return JAMMER_DATABASE.get(jammer_id)


# Default configuration instance
DEFAULT_CONFIG = Config()


if __name__ == "__main__":
    print("=" * 60)
    print("UAV-CPS-Analyzer: Configuration Module Test")
    print("=" * 60)
    
    # Test configuration
    config = Config()
    
    print("\n1. Simulation Config:")
    print(f"   Iterations: {config.simulation.n_iterations}")
    print(f"   CI Level: {config.simulation.ci_level}")
    
    print("\n2. Propagation Config:")
    print(f"   Urban threshold: {config.propagation.h_urban}m")
    print(f"   Free-space threshold: {config.propagation.h_freespace}m")
    
    print("\n3. Drone Database:")
    for drone_type, spec in DRONE_DATABASE.items():
        print(f"   {spec.name}: {spec.protocol}, {spec.fhss_channels} channels, "
              f"max {spec.max_range_m/1000:.0f}km")
    
    print("\n4. Jammer Database:")
    for jammer_id, spec in JAMMER_DATABASE.items():
        print(f"   {spec.name}: {spec.power_w}W ({spec.power_dbm:.1f}dBm), "
              f"${spec.cost_usd:,.0f}")
    
    # Save default config
    config.save("/home/claude/UAV-CPS-Analyzer/config.json")
    print("\n5. Config saved to config.json")
    
    print("\n" + "=" * 60)
    print("Configuration test complete!")
