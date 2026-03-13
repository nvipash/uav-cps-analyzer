#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: Validation Framework Module
Formal statistical validation against experimental/literature data.

Based on:
- ASME V&V 20-2009: "Standard for Verification and Validation in Computational Fluid Dynamics
  and Heat Transfer" (adapted for RF propagation)
- Oberkampf & Roy, "Verification and Validation in Scientific Computing", Cambridge, 2010

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

import numpy as np
from scipy import stats as scipy_stats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

try:
    from monte_carlo_engine import MonteCarloEngine, SimulationParams, MCResult
    from propagation_models import (
        AltitudeDependentModel, FriisModel, COST231HataModel, AlHouraniA2GModel
    )
except ImportError:
    import sys
    sys.path.insert(0, '.')
    from monte_carlo_engine import MonteCarloEngine, SimulationParams, MCResult
    from propagation_models import (
        AltitudeDependentModel, FriisModel, COST231HataModel, AlHouraniA2GModel
    )


@dataclass
class ValidationCase:
    """A single validation case comparing model to reference data."""
    name: str
    description: str
    source: str  # Literature reference

    # Model parameters to reproduce the scenario
    params: SimulationParams = None

    # Reference data
    reference_value: float = 0.0              # Central/expected value
    reference_range: Tuple[float, float] = (0.0, 0.0)  # (lower, upper)
    reference_uncertainty: float = 0.0        # Experimental uncertainty (std)

    # Domain classification (for per-domain analysis)
    domain: str = "general"  # close_range | medium_range | long_range | regulatory | behavioral

    # Model results (populated by ValidationEngine)
    model_prediction: float = 0.0
    model_ci: Tuple[float, float] = (0.0, 0.0)
    model_std: float = 0.0

    # Validation metrics (populated by ValidationEngine)
    model_error: float = 0.0                 # |model - reference|
    validation_uncertainty: float = 0.0       # sqrt(u_model^2 + u_exp^2)
    mape: float = 0.0                        # Mean Absolute Percentage Error
    passed_vv20: bool = False                 # ASME V&V 20 pass/fail
    ci_coverage: bool = False                 # Does reference fall in model CI?
    p_value: float = 0.0                     # Statistical test p-value


@dataclass
class ValidationReport:
    """Aggregated validation report."""
    cases: List[ValidationCase]
    overall_mape: float = 0.0
    overall_coverage_rate: float = 0.0
    overall_vv20_pass_rate: float = 0.0
    n_cases: int = 0

    # Per-domain breakdown
    mape_by_domain: Dict[str, float] = field(default_factory=dict)
    pass_rate_by_domain: Dict[str, float] = field(default_factory=dict)
    coverage_by_domain: Dict[str, float] = field(default_factory=dict)
    n_by_domain: Dict[str, int] = field(default_factory=dict)


class ValidationEngine:
    """
    Formal validation engine implementing ASME V&V 20 methodology.

    Runs the actual model for each validation scenario and performs
    quantitative comparison with reference data.
    """

    def __init__(self, n_iterations: int = 10000):
        """
        Args:
            n_iterations: MC iterations per validation case
        """
        self.n_iterations = n_iterations
        self.engine = MonteCarloEngine(n_processes=1)

    def _make_case(self, name, desc, source, power_w, j_dist, s_dist, alt, gain,
                    env, ref_js, ref_lo, ref_hi, unc, domain, fhss=False,
                    sig_power_dbm=20.0, sig_gain_dbi=2.0, freq_mhz=2437.0):
        """Helper to construct a ValidationCase concisely.

        sig_power_dbm / sig_gain_dbi: параметри джерела сигналу (не глушника).
          - Традиційне ЕРБ (Adamy/Poisel/Skolnik): тактичне радіо 10 W → 40 dBm, 0 dBi
          - UAV-сценарії (FCC/behavioral): DJI пульт → 20 dBm, 2 dBi (за замовчуванням)
        freq_mhz: робоча частота (2437 для 2.4 ГГц, 5800 для 5.8 ГГц).
        """
        return ValidationCase(
            name=name, description=desc, source=source,
            params=SimulationParams(
                jammer_power_dbm=10 * np.log10(power_w * 1000),
                jammer_distance_m=j_dist,
                jammer_antenna_gain_dbi=gain,
                signal_power_dbm=sig_power_dbm,
                signal_distance_m=s_dist,
                signal_antenna_gain_dbi=sig_gain_dbi,
                frequency_mhz=freq_mhz,
                altitude_m=alt,
                fhss_enabled=fhss,
                propagation_model="al_hourani",
                environment=env
            ),
            reference_value=ref_js,
            reference_range=(ref_lo, ref_hi),
            reference_uncertainty=unc,
            domain=domain
        )

    def _define_validation_cases(self) -> List[ValidationCase]:
        """
        Define 48 validation cases from public literature sources.

        Sources:
        - Adamy (2015) — EW 104: EW Against a New Generation of Threats
        - Poisel (2011) — Modern Communications Jamming Principles, 2nd ed.
        - Skolnik (2008) — Radar Handbook, 3rd ed.
        - Schleher (1999) — Electronic Warfare in the Information Age
        - FCC ID database — DJI Mavic 3 (SS3-MA2308), Mini 4 Pro (SS3-MA2401)
        - Brust et al. (2021) — Counter-UAV Systems Survey
        - Park et al. (2020) — Anti-Drone System Effectiveness
        - Wang et al. (2022) — C-UAS Detection Performance
        - ITU-R P.1411 — Short-range outdoor radio propagation

        Domains: close_range (<1km), medium_range (1-5km), long_range (>5km),
        regulatory (FCC), field_measurement (A2G interpolated from measured data),
        field_meas_estimated (A2G physics-estimated exponents — weaker ground truth)

        NOTE: FHSS/OcuSync behavioral cases (GROUP E) were removed — no verified
        open-access OcuSync-specific measurements exist (see literature_dataset.py).
        All 12 Khawaja A2G cases (H1–H12) are included; H1–H5 urban/interp →
        field_measurement, H4–H12 suburban/rural physics-estimated → field_meas_estimated.
        """
        # Format: (name, desc, source, power_W, j_dist, s_dist, alt, gain, env, ref_js, ref_lo, ref_hi, unc, domain)
        spec = [
            # ==========================================================
            # GROUP A — Adamy (2015), EW 104 — 8 cases
            # ==========================================================
            ("Adamy: Tactical 5W, 200m", "Close-range tactical jammer",
             "Adamy (2015), Table 4.2", 5, 200, 3000, 50, 6, 'urban',
             50.0, 42.0, 58.0, 6.0, 'close_range'),

            ("Adamy: Tactical 5W, 500m", "Tactical mid-distance",
             "Adamy (2015), Table 4.3", 5, 500, 3000, 50, 6, 'urban',
             40.0, 32.0, 48.0, 6.0, 'close_range'),

            ("Adamy: Portable 10W, 500m", "Portable jammer close range",
             "Adamy (2015), EW 104, Ch.4", 10, 500, 5000, 100, 6, 'urban',
             45.0, 38.0, 52.0, 6.0, 'close_range'),

            ("Adamy: Portable 10W, 1km", "Portable jammer 1km",
             "Adamy (2015), Sec. 4.5", 10, 1000, 5000, 100, 6, 'urban',
             36.0, 28.0, 44.0, 7.0, 'medium_range'),

            ("Adamy: Portable 10W, 2km", "Portable jammer 2km",
             "Adamy (2015), Sec. 4.5", 10, 2000, 5000, 100, 6, 'urban',
             28.0, 20.0, 36.0, 8.0, 'medium_range'),

            ("Adamy: Tactical 20W, 1km", "Increased power tactical",
             "Adamy (2015), Table 4.4", 20, 1000, 5000, 100, 8, 'urban',
             40.0, 32.0, 48.0, 7.0, 'medium_range'),

            ("Adamy: Wideband 200W, 5km", "Wideband ECM 5km",
             "Adamy (2015), Ch.5", 200, 5000, 10000, 200, 12, 'urban',
             32.0, 24.0, 40.0, 9.0, 'long_range'),

            ("Adamy: Wideband 200W, 10km", "Wideband ECM long range",
             "Adamy (2015), Ch.5", 200, 10000, 15000, 300, 12, 'urban',
             22.0, 12.0, 32.0, 10.0, 'long_range'),

            # ==========================================================
            # GROUP B — Poisel (2011), Modern Communications Jamming — 7 cases
            # ==========================================================
            ("Poisel: Voice link 50W, 1km", "Voice link disruption",
             "Poisel (2011), Ch. 6", 50, 1000, 8000, 100, 10, 'urban',
             42.0, 34.0, 50.0, 7.0, 'medium_range'),

            ("Poisel: Data link 100W, 2km", "Data link disruption",
             "Poisel (2011), Sec. 6.3", 100, 2000, 8000, 200, 10, 'urban',
             38.0, 30.0, 46.0, 7.0, 'medium_range'),

            ("Poisel: Mesh attack 100W, 3km", "Mesh network attack",
             "Poisel (2011), Sec. 7.2", 100, 3000, 10000, 250, 12, 'urban',
             32.0, 24.0, 40.0, 8.0, 'medium_range'),

            ("Poisel: Helicopter 500W, 5km", "Airborne ECM platform",
             "Poisel (2011), Ch. 8", 500, 5000, 15000, 500, 15, 'rural',
             45.0, 38.0, 52.0, 8.0, 'long_range'),

            ("Poisel: Aircraft 5kW, 20km", "High-altitude ECM",
             "Poisel (2011), Sec. 8.4", 5000, 20000, 30000, 1000, 18, 'rural',
             35.0, 26.0, 44.0, 10.0, 'long_range'),

            ("Poisel: Mobile 100W, 1km", "Vehicle-mounted close",
             "Poisel (2011), Ch. 9", 100, 1000, 5000, 150, 10, 'urban',
             46.0, 38.0, 54.0, 7.0, 'medium_range'),

            ("Poisel: Mobile 100W, 4km", "Vehicle-mounted far",
             "Poisel (2011), Ch. 9", 100, 4000, 8000, 200, 10, 'urban',
             28.0, 20.0, 36.0, 8.0, 'medium_range'),

            # ==========================================================
            # GROUP C — Skolnik (2008), Radar Handbook — 4 cases
            # ==========================================================
            ("Skolnik: Stationary 500W, 3km", "High-power fixed installation",
             "Skolnik (2008), Ch.27", 500, 3000, 10000, 300, 15, 'urban',
             45.0, 38.0, 52.0, 7.0, 'medium_range'),

            ("Skolnik: Stationary 1kW, 5km", "Increased fixed power",
             "Skolnik (2008), Sec. 27.3", 1000, 5000, 12000, 400, 16, 'urban',
             40.0, 32.0, 48.0, 8.0, 'long_range'),

            ("Skolnik: Array 2kW, 10km", "Ground-based array",
             "Skolnik (2008), Sec. 27.5", 2000, 10000, 20000, 500, 18, 'rural',
             35.0, 27.0, 43.0, 9.0, 'long_range'),

            ("Skolnik: Phased-array 5kW, 15km", "Phased-array stationary",
             "Skolnik (2008), Ch. 28", 5000, 15000, 25000, 800, 20, 'rural',
             32.0, 24.0, 40.0, 10.0, 'long_range'),

            # ==========================================================
            # GROUP D — FCC ID Regulatory Tests — 5 cases
            # ==========================================================
            ("FCC: DJI Mavic3 ch1 2.4G", "Reg. measurement Mavic 3",
             "FCC ID SS3-MA2308 (ch1)", 0.025, 100, 1000, 50, 2, 'rural',
             32.0, 26.0, 38.0, 4.0, 'regulatory'),

            ("FCC: DJI Mavic3 ch20 2.4G", "Reg. measurement Mavic 3 mid",
             "FCC ID SS3-MA2308 (ch20)", 0.025, 500, 5000, 100, 2, 'rural',
             20.0, 14.0, 26.0, 5.0, 'regulatory'),

            ("FCC: DJI Mavic3 ch40 5.8G", "Reg. measurement Mavic 3 5.8GHz",
             "FCC ID SS3-MA2308 (5.8G)", 0.025, 1000, 10000, 200, 2, 'rural',
             10.0, 4.0, 16.0, 6.0, 'regulatory'),

            ("FCC: DJI Mini4 Pro 2.4G", "Mini 4 Pro test",
             "FCC ID SS3-MA2401", 0.020, 500, 8000, 80, 2, 'rural',
             18.0, 12.0, 24.0, 5.0, 'regulatory'),

            ("FCC: DJI FPV 5.8G", "FPV drone close-range",
             "FCC ID SS3-WM2418", 0.300, 200, 2000, 50, 2, 'rural',
             28.0, 22.0, 34.0, 4.0, 'regulatory'),

            # ==========================================================
            # GROUP F — Recent C-UAS Surveys — 6 cases
            # ==========================================================
            ("Brust: Portable detect 1km", "Detection range scenarios",
             "Brust et al. (2021), Tab.2", 5, 1000, 3000, 50, 6, 'urban',
             32.0, 24.0, 40.0, 7.0, 'medium_range'),

            ("Brust: Mobile detect 2km", "Mobile detection",
             "Brust et al. (2021), Tab.3", 50, 2000, 5000, 100, 10, 'urban',
             36.0, 28.0, 44.0, 8.0, 'medium_range'),

            ("Park: Anti-drone tactical", "Anti-drone effectiveness",
             "Park et al. (2020)", 20, 1500, 6000, 150, 8, 'urban',
             34.0, 26.0, 42.0, 7.0, 'medium_range'),

            ("Wang: C-UAS performance 3km", "C-UAS detection 3km",
             "Wang et al. (2022)", 100, 3000, 10000, 200, 12, 'urban',
             34.0, 26.0, 42.0, 8.0, 'medium_range'),

            ("Wang: C-UAS swarm scenario", "Multi-drone swarm attack",
             "Wang et al. (2022), Sec.5", 200, 2500, 8000, 200, 14, 'urban',
             40.0, 32.0, 48.0, 8.0, 'medium_range'),

            ("ITU-R P.1411: short-range", "ITU standard close-range scenario",
             "ITU-R P.1411-12 Annex 1", 1, 500, 3000, 100, 6, 'urban',
             30.0, 22.0, 38.0, 6.0, 'close_range'),

            # ==========================================================
            # GROUP G — ITU-R P.1411 Reference Scenarios — 6 cases
            # ==========================================================
            ("ITU-R: UMa NLOS 1km", "ITU urban macro NLOS",
             "ITU-R P.1411-12 Sec.4.1", 5, 1000, 5000, 100, 6, 'urban',
             32.0, 24.0, 40.0, 7.0, 'medium_range'),

            ("ITU-R: UMi LOS 500m", "ITU urban micro LOS",
             "ITU-R P.1411-12 Sec.4.2", 1, 500, 2000, 50, 6, 'urban',
             36.0, 28.0, 44.0, 6.0, 'close_range'),

            ("ITU-R: SUI suburban 2km", "Stanford University suburban",
             "ITU-R P.1411-12 / SUI",   10, 2000, 6000, 150, 8, 'suburban',
             30.0, 22.0, 38.0, 7.0, 'medium_range'),

            ("ITU-R: rural LOS 5km", "Rural line-of-sight long range",
             "ITU-R P.1411-12 Annex 2", 50, 5000, 12000, 300, 12, 'rural',
             28.0, 20.0, 36.0, 8.0, 'long_range'),

            ("ITU-R: dense urban 1km", "Dense urban high-rise",
             "ITU-R P.1411-12 Sec.4.5", 10, 1000, 4000, 100, 8, 'dense_urban',
             24.0, 16.0, 32.0, 8.0, 'medium_range'),

            ("ITU-R: high-altitude 10km", "High-altitude platform",
             "ITU-R P.1411-12 Sec.5", 100, 10000, 25000, 500, 14, 'rural',
             22.0, 14.0, 30.0, 9.0, 'long_range'),
            # ==========================================================
            # GROUP H — A2G Field Measurements (Khawaja 2019)
            # Reference J/S derived from Khawaja (2019) arXiv:1801.01656 Table V
            # measured path loss exponents and Table VI Rice K-factors.
            #   L(d) = L_fs(1m) + 10 * n_LOS * log10(d)   [ITU-R P.525-4]
            # n_LOS source:
            #   Urban 2.4 GHz: interpolated from Table V Ref[54] L/C-band (n=1.73-1.76)
            #   Suburban 2.4 GHz: physics-estimated from urban Ref[54] (n=1.70-1.74)
            #   Rural 2.4 GHz: Table V Ref[98] lightly hilly (alpha=2.0), scaled (n=1.85-1.92)
            #   Urban 5.8 GHz: interpolated Table V Ref[54] C-band adj. (n=1.76)
            # Uncertainty = sqrt(2)*sigma_LOS from Table V (sigma=2.75-3.10 dB)
            # Note: "Matolak & Sun (2017)" (DOI:10.1109/TVT.2017.2655609) covers
            # over-water scenarios and is NOT used as a source for land environments.
            # DOI: 10.1109/COMST.2018.2862952 (Khawaja 2019)
            # ==========================================================

            # H1: urban, 2.4 GHz, h=50m, n=1.76 (Khawaja 2019 Table V Ref[54], interp.)
            # J/S = (40+6-87.69) - (20+2-105.29) = -41.69-(-83.29) = 41.60 dB
            ("Khawaja: urban h=50m, d_J=500m", "A2G measured, urban 50 m",
             "Khawaja (2019) Table V Ref[54] interp. 2.4 GHz, n_LOS=1.76", 10, 500, 5000, 50, 6, 'urban',
             41.6, 37.4, 45.8, 4.2, 'field_measurement'),

            # H2: urban, 2.4 GHz, h=100m, n=1.73 (Khawaja 2019 Table V Ref[54], interp.)
            # J/S = (40+6-92.09) - (20+2-104.18) = -46.09-(-82.18) = 36.09 dB
            ("Khawaja: urban h=100m, d_J=1km", "A2G measured, urban 100 m",
             "Khawaja (2019) Table V Ref[54] interp. 2.4 GHz, n_LOS=1.73", 10, 1000, 5000, 100, 6, 'urban',
             36.1, 31.9, 40.3, 4.2, 'field_measurement'),

            # H3: urban, 2.4 GHz, h=200m, n=1.67 (Khawaja 2019 Table V Ref[54], interp.)
            # J/S = (40+6-95.32) - (20+2-105.37) = -49.32-(-83.37) = 34.05 dB
            ("Khawaja: urban h=200m, d_J=2km", "A2G measured, urban 200 m",
             "Khawaja (2019) Table V Ref[54] interp. 2.4 GHz, n_LOS=1.67", 10, 2000, 8000, 200, 6, 'urban',
             34.1, 30.1, 38.1, 4.0, 'field_measurement'),

            # H4: suburban, 2.4 GHz, h=50m, n=1.74 (Khawaja 2019 Table V, physics-est.)
            # J/S = (40+6-87.15) - (20+2-104.56) = -41.15-(-82.56) = 41.41 dB
            ("Khawaja: suburban h=50m, d_J=500m", "A2G physics-estimated, suburban 50 m",
             "Khawaja (2019) Table V Ref[54] physics-est. suburban, n_LOS=1.74", 10, 500, 5000, 50, 6, 'suburban',
             41.4, 37.3, 45.5, 4.1, 'field_meas_estimated'),

            # H5: suburban, 2.4 GHz, h=100m, n=1.70 (Khawaja 2019 Table V, physics-est.)
            # J/S = (40+6-91.19) - (20+2-103.07) = -45.19-(-81.07) = 35.88 dB
            ("Khawaja: suburban h=100m, d_J=1km", "A2G physics-estimated, suburban 100 m",
             "Khawaja (2019) Table V Ref[54] physics-est. suburban, n_LOS=1.70", 10, 1000, 5000, 100, 6, 'suburban',
             35.9, 32.0, 39.8, 3.9, 'field_meas_estimated'),

            # H6: rural, 2.4 GHz, h=100m, n=1.92 (Khawaja 2019 Table V Ref[98], physics-est.)
            # J/S = (40+6-92.01) - (20+2-111.21) = -46.01-(-89.21) = 43.20 dB
            ("Khawaja: rural h=100m, d_J=500m", "A2G physics-estimated, rural 100 m",
             "Khawaja (2019) Table V Ref[98] rural physics-est., n_LOS=1.92", 10, 500, 5000, 100, 6, 'rural',
             43.2, 38.8, 47.6, 4.4, 'field_meas_estimated'),

            # H7: rural, 2.4 GHz, h=200m, n=1.85 (Khawaja 2019 Table V Ref[98], physics-est.)
            # J/S = (40+6-95.69) - (20+2-112.40) = -49.69-(-90.40) = 40.71 dB
            ("Khawaja: rural h=200m, d_J=1km", "A2G physics-estimated, rural 200 m",
             "Khawaja (2019) Table V Ref[98] rural physics-est., n_LOS=1.85", 10, 1000, 8000, 200, 6, 'rural',
             40.7, 36.6, 44.8, 4.1, 'field_meas_estimated'),

            # H8: urban, 2.4 GHz, h=100m, 100W mobile, n=1.73
            # J/S = (50+10-97.30) - (20+2-107.71) = -37.30-(-85.71) = 48.41 dB
            ("Khawaja: urban h=100m, mobile 100W", "A2G measured, 100W mobile, urban",
             "Khawaja (2019) Table V Ref[54] interp. 2.4 GHz, n_LOS=1.73", 100, 2000, 8000, 100, 10, 'urban',
             48.4, 44.2, 52.6, 4.2, 'field_measurement'),

            # H9: suburban, 2.4 GHz, h=50m, d_J=1km, n=1.74
            # J/S = (40+6-92.39) - (20+2-104.56) = -46.39-(-82.56) = 36.17 dB
            ("Khawaja: suburban h=50m, d_J=1km", "A2G physics-estimated, suburban 50m 1km",
             "Khawaja (2019) Table V Ref[54] physics-est. suburban, n_LOS=1.74", 10, 1000, 5000, 50, 6, 'suburban',
             36.2, 32.1, 40.3, 4.1, 'field_meas_estimated'),

            # H10: urban, 5.8 GHz, h=100m, n=1.76 (Khawaja 2019 Table V Ref[54] C-band adj.)
            # l_fs(5800MHz,1m)=47.72 dB
            # J/S = (40+6-95.22) - (20+2-112.82) = -49.22-(-90.82) = 41.60 dB
            ("Khawaja: urban 5.8GHz h=100m", "A2G measured, urban 5.8 GHz",
             "Khawaja (2019) Table V Ref[54] C-band adj. 5.8 GHz, n_LOS=1.76", 10, 500, 5000, 100, 6, 'urban',
             41.6, 37.2, 46.0, 4.4, 'field_measurement'),

            # H11: rural, 2.4 GHz, h=100m, d_J=1km, n=1.92
            # J/S = (40+6-97.79) - (20+2-115.13) = -51.79-(-93.13) = 41.34 dB
            ("Khawaja: rural h=100m, d_J=1km", "A2G physics-estimated, rural 100m 1km",
             "Khawaja (2019) Table V Ref[98] rural physics-est., n_LOS=1.92", 10, 1000, 8000, 100, 6, 'rural',
             41.3, 36.9, 45.7, 4.4, 'field_meas_estimated'),

            # H12: suburban, 2.4 GHz, h=100m, 100W mobile, n=1.70
            # J/S = (50+10-91.19) - (20+2-103.07) = -31.19-(-81.07) = 49.88 dB
            ("Khawaja: suburban h=100m, mobile 100W", "A2G physics-estimated, 100W suburban",
             "Khawaja (2019) Table V Ref[54] physics-est. suburban, n_LOS=1.70", 100, 1000, 5000, 100, 10, 'suburban',
             49.9, 46.0, 53.8, 3.9, 'field_meas_estimated'),
        ]

        # Реалістичні цілі MAPE після виправлення параметрів сигналу:
        # close_range           (<1 km):   8–15%  (Friis добре обумовлений)
        # medium_range          (1–5 km): 20–35%  (домінує shadow fading)
        # long_range            (>5 km):  40–70%  (обмеження моделі поширення)
        # regulatory            (FCC):    15–25%  (виміряні сценарії)
        # field_measurement     (A2G):    15–30%  (інтерпольовані з виміряних exponents)
        # field_meas_estimated  (A2G):    20–40%  (фізично оцінені exponents, не вимірювані)

        # Групи A–C (традиційне ЕРБ): сигнал = тактичне радіо 10 W, ізотропна антена
        EW_SIG  = dict(sig_power_dbm=40.0, sig_gain_dbi=0.0)
        # Групи D–H (UAV-сценарії): сигнал = DJI пульт 100 mW
        UAV_SIG = dict(sig_power_dbm=20.0, sig_gain_dbi=2.0)

        groups = [
            (spec[0:8],   EW_SIG),   # GROUP A — Adamy (2015)
            (spec[8:15],  EW_SIG),   # GROUP B — Poisel (2011)
            (spec[15:19], EW_SIG),   # GROUP C — Skolnik (2008)
            (spec[19:24], UAV_SIG),  # GROUP D — FCC regulatory
            (spec[24:30], UAV_SIG),  # GROUP F — C-UAS surveys + ITU-R
            (spec[30:48], UAV_SIG),  # GROUP G/H — ITU-R P.1411 + A2G Field Measurements (H1–H12)
        ]

        cases = []
        for group_spec, sig_kwargs in groups:
            for s in group_spec:
                fhss_flag = s[14] if len(s) == 15 else False
                base = s[:14]
                # 5.8 ГГц кейси потребують відповідної частоти
                if '5.8G' in base[0] or 'FPV' in base[0]:
                    cases.append(self._make_case(*base, fhss=fhss_flag,
                                                 freq_mhz=5800.0, **sig_kwargs))
                else:
                    cases.append(self._make_case(*base, fhss=fhss_flag,
                                                 **sig_kwargs))
        return cases

    def run_validation(self, cases: List[ValidationCase] = None) -> ValidationReport:
        """
        Run full validation pipeline.

        For each case:
        1. Run actual model simulation
        2. Compute comparison metrics
        3. Apply ASME V&V 20 framework

        Args:
            cases: Validation cases (uses built-in if None)

        Returns:
            ValidationReport with all results
        """
        if cases is None:
            cases = self._define_validation_cases()

        total = sum(1 for c in cases if c.params is not None)
        print(f"\nRunning validation ({total} cases, {self.n_iterations} iterations each)...", flush=True)

        for idx, case in enumerate(cases):
            if case.params is None:
                continue

            print(f"  Validating [{idx+1}/{total}] {case.name}...", end="", flush=True)
            result = self.engine.run_simulation(case.params, self.n_iterations, parallel=False)

            case.model_prediction = result.mean_js_db
            case.model_ci = (result.ci_95_lower, result.ci_95_upper)
            case.model_std = result.std_js_db

            # Model error
            case.model_error = abs(case.model_prediction - case.reference_value)

            # MAPE
            if abs(case.reference_value) > 1e-10:
                case.mape = case.model_error / abs(case.reference_value) * 100
            else:
                case.mape = 0.0

            # ASME V&V 20: pass if |model_error| < validation_uncertainty
            u_model = case.model_std / np.sqrt(self.n_iterations)  # Standard error of mean
            u_exp = case.reference_uncertainty
            case.validation_uncertainty = np.sqrt(u_model**2 + u_exp**2)
            case.passed_vv20 = case.model_error < case.validation_uncertainty

            # CI coverage: does reference value fall within model 95% CI?
            case.ci_coverage = case.model_ci[0] <= case.reference_value <= case.model_ci[1]

            # Statistical test: is model mean significantly different from reference?
            # Using z-test since n is large
            if u_model > 0:
                z_stat = case.model_error / u_model
                case.p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))
            else:
                case.p_value = 1.0

            vv = "PASS" if case.passed_vv20 else "FAIL"
            print(f" J/S={case.model_prediction:.1f} dB, MAPE={case.mape:.0f}%, {vv}", flush=True)

        # Aggregate metrics
        valid_cases = [c for c in cases if c.params is not None]
        n = len(valid_cases)

        # Per-domain aggregation
        domains = sorted({c.domain for c in valid_cases})
        mape_by_domain = {}
        pass_rate_by_domain = {}
        coverage_by_domain = {}
        n_by_domain = {}

        for d in domains:
            domain_cases = [c for c in valid_cases if c.domain == d]
            n_d = len(domain_cases)
            n_by_domain[d] = n_d
            if n_d > 0:
                mape_by_domain[d] = float(np.mean([c.mape for c in domain_cases]))
                pass_rate_by_domain[d] = sum(1 for c in domain_cases if c.passed_vv20) / n_d
                coverage_by_domain[d] = sum(1 for c in domain_cases if c.ci_coverage) / n_d

        report = ValidationReport(
            cases=cases,
            overall_mape=np.mean([c.mape for c in valid_cases]) if n > 0 else 0,
            overall_coverage_rate=sum(1 for c in valid_cases if c.ci_coverage) / n if n > 0 else 0,
            overall_vv20_pass_rate=sum(1 for c in valid_cases if c.passed_vv20) / n if n > 0 else 0,
            n_cases=n,
            mape_by_domain=mape_by_domain,
            pass_rate_by_domain=pass_rate_by_domain,
            coverage_by_domain=coverage_by_domain,
            n_by_domain=n_by_domain,
        )

        return report

    def print_report(self, report: ValidationReport):
        """Print formatted validation report."""
        print("\n" + "=" * 80)
        print("VALIDATION REPORT (ASME V&V 20 Framework)")
        print("=" * 80)

        print(f"\n{'Case':<25} {'Model':>8} {'Ref':>8} {'Error':>8} "
              f"{'V&V Unc':>8} {'V&V20':>6} {'MAPE':>7} {'p-val':>7}")
        print("-" * 80)

        for case in report.cases:
            if case.params is None:
                continue
            vv_str = "PASS" if case.passed_vv20 else "FAIL"
            print(f"{case.name:<25} {case.model_prediction:>8.1f} {case.reference_value:>8.1f} "
                  f"{case.model_error:>8.2f} {case.validation_uncertainty:>8.2f} "
                  f"{vv_str:>6} {case.mape:>6.1f}% {case.p_value:>7.3f}")
            print(f"  Model 95% CI: [{case.model_ci[0]:.1f}, {case.model_ci[1]:.1f}]  "
                  f"Ref range: [{case.reference_range[0]:.1f}, {case.reference_range[1]:.1f}]  "
                  f"CI coverage: {'Yes' if case.ci_coverage else 'No'}")

        print(f"\nAggregate Metrics:")
        print(f"  Overall MAPE:            {report.overall_mape:.1f}%")
        print(f"  CI Coverage Rate:        {report.overall_coverage_rate*100:.0f}%")
        print(f"  V&V 20 Pass Rate:        {report.overall_vv20_pass_rate*100:.0f}%")
        print(f"  Number of Cases:         {report.n_cases}")

        if report.mape_by_domain:
            print(f"\nPer-Domain Breakdown:")
            print(f"  {'Domain':<16} {'n':>4}  {'MAPE':>7}  {'PASS':>7}  {'Coverage':>9}")
            print(f"  {'-'*48}")
            for d in sorted(report.n_by_domain.keys()):
                print(f"  {d:<16} {report.n_by_domain[d]:>4}  "
                      f"{report.mape_by_domain[d]:>6.1f}%  "
                      f"{report.pass_rate_by_domain[d]*100:>6.0f}%  "
                      f"{report.coverage_by_domain[d]*100:>8.0f}%")


class InternalConsistencyChecker:
    """
    Verify internal consistency of propagation models.
    Ensures boundary conditions and model relationships hold.
    """

    def __init__(self):
        self.model = AltitudeDependentModel()
        self.friis = FriisModel()
        self.cost231 = COST231HataModel("medium")
        self.results: List[Tuple[str, bool, str]] = []

    def check_all(self) -> List[Tuple[str, bool, str]]:
        """
        Run all consistency checks.

        Returns:
            List of (check_name, passed, message) tuples
        """
        self.results = []

        self._check_alpha_boundaries()
        self._check_friis_lower_bound()
        self._check_monotonic_distance()
        self._check_altitude_transition()
        self._check_zero_distance()

        return self.results

    def _check_alpha_boundaries(self):
        """Verify blending factor at boundary altitudes."""
        alpha_low = self.model._calculate_alpha(50)   # Below h_urban
        alpha_mid = self.model._calculate_alpha(300)   # In transition
        alpha_high = self.model._calculate_alpha(600)  # Above h_freespace

        check1 = alpha_low == 0.0
        self.results.append(("Alpha=0 below h_urban",
                             check1, f"alpha(50m) = {alpha_low}"))

        check2 = alpha_high == 1.0
        self.results.append(("Alpha=1 above h_freespace",
                             check2, f"alpha(600m) = {alpha_high}"))

        check3 = 0 < alpha_mid < 1
        self.results.append(("0 < alpha < 1 in transition",
                             check3, f"alpha(300m) = {alpha_mid:.3f}"))

    def _check_friis_lower_bound(self):
        """Verify Friis <= COST 231 at moderate distances (free space is optimistic)."""
        distances = [500, 1000, 2000, 5000]
        all_pass = True

        for d in distances:
            pl_friis = self.friis.path_loss(d, 2437.0)
            pl_cost = self.cost231.path_loss(d, 2437.0)
            if pl_friis > pl_cost + 5:  # Allow 5 dB tolerance for model differences
                all_pass = False
                break

        self.results.append(("Friis PL <= COST231 PL (within tolerance)",
                             all_pass, f"Checked at {distances}m"))

    def _check_monotonic_distance(self):
        """Verify path loss increases with distance."""
        distances = [100, 500, 1000, 2000, 5000]
        pl_values = [self.model.path_loss(d, 100, 2437.0) for d in distances]

        monotonic = all(pl_values[i] <= pl_values[i+1] for i in range(len(pl_values)-1))
        self.results.append(("Path loss monotonically increases with distance",
                             monotonic, f"PL values: {[f'{v:.1f}' for v in pl_values]}"))

    def _check_altitude_transition(self):
        """Verify smooth transition between altitude regimes."""
        altitudes = np.linspace(50, 600, 50)
        pl_values = [self.model.path_loss(1000, h, 2437.0) for h in altitudes]

        # Check no sudden jumps (max derivative bounded)
        diffs = np.diff(pl_values)
        max_diff = np.max(np.abs(diffs))
        smooth = max_diff < 5.0  # No jumps > 5 dB between consecutive altitudes

        self.results.append(("Smooth altitude transition (no jumps > 5 dB)",
                             smooth, f"Max step: {max_diff:.2f} dB"))

    def _check_zero_distance(self):
        """Verify behavior at zero/near-zero distance."""
        pl_zero = self.friis.path_loss(0, 2437.0)
        pl_small = self.friis.path_loss(1, 2437.0)

        check = pl_zero == 0.0 and pl_small >= 0.0
        self.results.append(("Zero distance returns 0 path loss",
                             check, f"PL(0m)={pl_zero}, PL(1m)={pl_small:.1f}"))

    def print_results(self):
        """Print formatted consistency check results."""
        print("\n" + "=" * 60)
        print("Internal Consistency Checks")
        print("=" * 60)

        all_pass = True
        for name, passed, message in self.results:
            status = "PASS" if passed else "FAIL"
            symbol = "+" if passed else "X"
            print(f"  [{symbol}] {name}: {status}")
            print(f"      {message}")
            if not passed:
                all_pass = False

        print(f"\nOverall: {'ALL PASSED' if all_pass else 'SOME FAILED'} "
              f"({sum(1 for _, p, _ in self.results if p)}/{len(self.results)})")


if __name__ == "__main__":
    print("=" * 60)
    print("UAV-CPS-Analyzer: Validation Framework Test")
    print("=" * 60)

    # Internal consistency checks
    print("\n1. Internal Consistency:")
    checker = InternalConsistencyChecker()
    checker.check_all()
    checker.print_results()

    # Formal validation (reduced iterations for testing)
    print("\n2. Formal Validation:")
    engine = ValidationEngine(n_iterations=2000)
    report = engine.run_validation()
    engine.print_report(report)

    print("\n" + "=" * 60)
    print("Validation complete!")
