#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer
================

A software complex for modeling and analysis of cyber-physical systems 
of unmanned aerial vehicles (UAVs).

Modules:
    - propagation_models: Signal propagation models (COST 231-Hata, Friis, Rice)
    - fhss_emulator: FHSS protocol emulation (OcuSync)
    - monte_carlo_engine: Monte Carlo simulation engine
    - cps_analyzer: Cyber-Physical Systems analyzer with sensor fusion
    - visualization: Publication-quality figure generation
    - config: Configuration and UAV/jammer databases

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

__version__ = "1.0.0"
__author__ = "Novitskyi P.S., Stepaniak M.V."
__email__ = "novitskyi@lp.edu.ua"
__institution__ = "Lviv Polytechnic National University"

from .propagation_models import (
    PropagationModel,
    FriisModel,
    COST231HataModel,
    RiceFadingModel,
    AltitudeDependentModel,
    calculate_js_ratio
)

from .fhss_emulator import (
    FHSSProtocol,
    OcuSyncProtocol,
    JammingStrategy,
    JammingEffectivenessAnalyzer,
    ChannelSimulator
)

from .monte_carlo_engine import (
    SimulationParams,
    SimulationResult,
    MCResult,
    MonteCarloEngine,
    ScenarioSimulator
)

from .cps_analyzer import (
    ThreatType,
    SensorType,
    NeutralizationMethod,
    Sensor,
    SensorFusion,
    CUASArchitecture,
    DempsterShafer
)

from .visualization import UAVCPSVisualizer

from .config import (
    Config,
    DroneType,
    DroneSpecification,
    JammerSpecification,
    DRONE_DATABASE,
    JAMMER_DATABASE
)

__all__ = [
    # Version info
    '__version__',
    '__author__',
    
    # Propagation models
    'PropagationModel',
    'FriisModel',
    'COST231HataModel',
    'RiceFadingModel',
    'AltitudeDependentModel',
    'calculate_js_ratio',
    
    # FHSS emulation
    'FHSSProtocol',
    'OcuSyncProtocol',
    'JammingStrategy',
    'JammingEffectivenessAnalyzer',
    'ChannelSimulator',
    
    # Monte Carlo
    'SimulationParams',
    'SimulationResult',
    'MCResult',
    'MonteCarloEngine',
    'ScenarioSimulator',
    
    # CPS Analysis
    'ThreatType',
    'SensorType',
    'NeutralizationMethod',
    'Sensor',
    'SensorFusion',
    'CUASArchitecture',
    'DempsterShafer',
    
    # Visualization
    'UAVCPSVisualizer',
    
    # Configuration
    'Config',
    'DroneType',
    'DroneSpecification',
    'JammerSpecification',
    'DRONE_DATABASE',
    'JAMMER_DATABASE'
]
