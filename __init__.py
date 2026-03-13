#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer
================

A software complex for modeling and analysis of cyber-physical systems
of unmanned aerial vehicles (UAVs).

Modules:
    - propagation_models: Signal propagation models (COST 231-Hata, Friis, Rice,
                          Al-Hourani A2G, BER, Doppler, antenna patterns)
    - fhss_emulator: FHSS protocol emulation (OcuSync)
    - monte_carlo_engine: Monte Carlo simulation engine with bootstrap CIs and convergence
    - cps_analyzer: Cyber-Physical Systems analyzer with sensor fusion
    - sensitivity: Global sensitivity analysis (Sobol indices, Morris method)
    - validation: Formal validation framework (ASME V&V 20) — includes GROUP H field data
    - reporting: LaTeX table generation and statistical summary reports
    - visualization: Publication-quality figure generation
    - config: Configuration and UAV/jammer databases
    - literature_dataset: A2G channel data from Khawaja (2019) arXiv:1801.01656
                          Tables V/VI (path loss exponents, Rice K-factors) for
                          ML calibration; values labeled by data_origin (measured /
                          interpolated / physics-estimated)

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
    AlHouraniA2GModel,
    BERModel,
    ModulationType,
    DopplerModel,
    AntennaPattern,
    OmnidirectionalPattern,
    CosinePattern,
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

from .sensitivity import (
    sobol_analysis,
    morris_screening,
    SobolResult,
    MorrisResult
)

from .validation import (
    ValidationEngine,
    ValidationCase,
    ValidationReport,
    InternalConsistencyChecker
)

from .reporting import (
    generate_latex_table,
    table3_latex,
    table4_latex,
    table7_latex,
    sobol_latex,
    generate_summary_report
)

from .visualization import UAVCPSVisualizer

from .config import (
    Config,
    DroneType,
    DroneSpecification,
    JammerSpecification,
    DRONE_DATABASE,
    JAMMER_DATABASE,
    ENVIRONMENT_PRESETS
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
    'AlHouraniA2GModel',
    'BERModel',
    'ModulationType',
    'DopplerModel',
    'AntennaPattern',
    'OmnidirectionalPattern',
    'CosinePattern',
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

    # Sensitivity analysis
    'sobol_analysis',
    'morris_screening',
    'SobolResult',
    'MorrisResult',

    # Validation
    'ValidationEngine',
    'ValidationCase',
    'ValidationReport',
    'InternalConsistencyChecker',

    # Reporting
    'generate_latex_table',
    'table3_latex',
    'table4_latex',
    'table7_latex',
    'sobol_latex',
    'generate_summary_report',

    # Visualization
    'UAVCPSVisualizer',

    # Configuration
    'Config',
    'DroneType',
    'DroneSpecification',
    'JammerSpecification',
    'DRONE_DATABASE',
    'JAMMER_DATABASE',
    'ENVIRONMENT_PRESETS',
]
