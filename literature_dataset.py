#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: Literature Dataset Module

Provides A2G channel parameters extracted from peer-reviewed publications.
Each entry is labeled by data_origin:
  "measured"          — value taken verbatim from a published table
  "interpolated"      — derived from measured values by frequency interpolation
  "physics-estimated" — physics-motivated estimate; no direct measurement

Primary source:
  Khawaja et al. (2019) — "A Survey of Air-to-Ground Propagation Channel
  Modeling for Unmanned Aerial Vehicles", IEEE Commun. Surveys Tuts. 21(3),
  pp. 2361-2391. DOI: 10.1109/COMST.2018.2862952. arXiv: 1801.01656.
  → Table V: path loss exponents (n/gamma), shadow fading std (sigma_dB)
  → Table VI: Rice K-factors

IMPORTANT DATA LIMITATIONS:
  1. Khawaja Tables V/VI aggregate field campaigns across altitude ranges
     (typically 50–300 m). The paper does NOT provide altitude-specific
     breakdowns. altitude_m entries below use physics-motivated scaling.
  2. Measured bands are L-band (~970 MHz) and C-band (~5.3 GHz). Entries
     at 2.4 GHz are frequency-interpolated (log-linear, weight=0.534).
  3. n_nlos and sigma_nlos_db values are NOT the focus of these tables and
     are labeled "physics-estimated" where used.

FHSS NOTE:
  No verified open-access measurements of DJI OcuSync jamming effectiveness
  were found. References "Beason et al. (2021)" and "Schiller et al. (2023)"
  could not be confirmed as real publications and have been removed.
  Parlin et al. (2018) IEEE ICMCIS is real but tests FASST/ACCST (not OcuSync)
  and is excluded to prevent cross-protocol extrapolation.

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ChannelParameters:
    """A2G channel parameters from published field campaigns."""
    source: str           # Full citation: paper + table + original reference
    data_origin: str      # "measured" | "interpolated" | "physics-estimated"
    environment: str      # dense_urban, urban, suburban, rural, open_field
    frequency_ghz: float  # Representative carrier frequency (GHz)
    altitude_m: float     # Representative UAV altitude for this entry (m)

    # Log-distance model: L(d) = L_fs(1m) + 10*n*log10(d)
    n_los: float          # LOS path loss exponent
    n_nlos: float         # NLOS path loss exponent
    sigma_los_db: float   # LOS shadow fading std (dB)
    sigma_nlos_db: float  # NLOS shadow fading std (dB)
    rice_k_db: float      # Rice K-factor, LOS conditions (dB)

    alt_min_m: float = 0.0
    alt_max_m: float = 1000.0


@dataclass
class FHSSVulnerabilityPoint:
    """FHSS jamming effectiveness from published experiments."""
    source: str
    strategy: str
    js_db: float
    effectiveness: float
    frequency_ghz: float = 2.437
    n_channels: int = 40
    hop_rate_hz: float = 500.0
    notes: str = ""


# Al-Hourani (2014), Table II: environment (a, b) parameters for LOS probability
_ENV_AB: Dict[str, Tuple[float, float]] = {
    'dense_urban': (12.08, 0.11),
    'urban':       (9.61,  0.16),
    'suburban':    (4.88,  0.43),
    'rural':       (2.00,  0.60),
    'open_field':  (1.00,  0.80),
}


class LiteratureDataset:
    """
    Repository of A2G channel parameters from published literature.
    All values sourced from Khawaja et al. (2019) Tables V/VI (arXiv 1801.01656).
    """

    @staticmethod
    def get_a2g_channel_params() -> List[ChannelParameters]:
        """
        A2G channel parameters from Khawaja (2019) Tables V and VI.

        Frequency interpolation for 2.4 GHz entries:
          weight = log10(2.4/0.97) / log10(5.3/0.97) = 0.534
          n(2.4)   = 1.70 + (1.75 - 1.70) * 0.534 = 1.73
          sigma(2.4) = 2.85 + (3.05 - 2.85) * 0.534 = 2.96 dB
          K(2.4)   linear: 10^(1.2+(2.74-1.2)*0.534) = 10^(1.2+0.822) = 10^2.022 ≈ 105 → 20.2 dB

        Altitude scaling (physics-motivated, not measured):
          Δn ≈ -0.03 per +100 m (higher altitude → more LOS → lower n)
          ΔK ≈ +2 dB per +100 m  (higher altitude → stronger LOS component)
        """
        return [
            # ================================================================
            # MEASURED: Urban/Suburban, L-band (~970 MHz)
            # Khawaja (2019) Table V, Table VI — original field campaign Ref[54]
            # Ref[54] measured A2G channels in urban/suburban environments.
            # n_los = 1.7 (Table V); sigma_los = midpoint [2.6, 3.1] = 2.85 dB
            # K = 12 dB at L-band (Table VI, Ref[54])
            # ================================================================
            ChannelParameters(
                source="Khawaja (2019) Table V/VI Ref[54] — urban/suburban L-band ~970 MHz",
                data_origin="measured",
                environment="urban", frequency_ghz=0.97, altitude_m=100,
                n_los=1.70, n_nlos=3.50,
                sigma_los_db=2.85,   # midpoint [2.6, 3.1] dB, Table V Ref[54]
                sigma_nlos_db=5.50,  # physics-estimated (not in Table V)
                rice_k_db=12.0,      # Table VI, Ref[54], L-band
                alt_min_m=30, alt_max_m=300,
            ),
            # ================================================================
            # MEASURED: Urban/Suburban, C-band (~5.3 GHz)
            # Khawaja (2019) Table V, Table VI — original field campaign Ref[54]
            # n_los = midpoint [1.5, 2.0] = 1.75; sigma_los = midpoint [2.9, 3.2] = 3.05 dB
            # K = 27.4 dB at C-band (Table VI, Ref[54])
            # ================================================================
            ChannelParameters(
                source="Khawaja (2019) Table V/VI Ref[54] — urban/suburban C-band ~5.3 GHz",
                data_origin="measured",
                environment="urban", frequency_ghz=5.3, altitude_m=100,
                n_los=1.75, n_nlos=3.50,
                sigma_los_db=3.05,   # midpoint [2.9, 3.2] dB, Table V Ref[54]
                sigma_nlos_db=5.50,
                rice_k_db=27.4,      # Table VI, Ref[54], C-band
                alt_min_m=30, alt_max_m=300,
            ),
            # ================================================================
            # INTERPOLATED: Urban, 2.4 GHz — primary UAV-CPS-Analyzer frequency
            # Log-freq interpolation from Khawaja (2019) Table V/VI Ref[54] L/C-band
            # Base (h=100m): n=1.73, sigma=2.96 dB, K=20 dB
            # Altitude entries use Δn=-0.03/+100m, ΔK=+2dB/+100m scaling
            # ================================================================
            ChannelParameters(
                source="Interpolated from Khawaja (2019) Table V/VI Ref[54] L/C-band → 2.4 GHz, h=50m",
                data_origin="interpolated",
                environment="urban", frequency_ghz=2.4, altitude_m=50,
                n_los=1.76, n_nlos=3.60,   # 1.73 + 0.03 (lower altitude)
                sigma_los_db=3.00,
                sigma_nlos_db=5.50,
                rice_k_db=18.0,             # 20 - 2 dB (lower altitude)
                alt_min_m=20, alt_max_m=80,
            ),
            ChannelParameters(
                source="Interpolated from Khawaja (2019) Table V/VI Ref[54] L/C-band → 2.4 GHz, h=100m",
                data_origin="interpolated",
                environment="urban", frequency_ghz=2.4, altitude_m=100,
                n_los=1.73, n_nlos=3.50,   # log-freq interp between 1.70 and 1.75
                sigma_los_db=2.96,          # log-freq interp between 2.85 and 3.05
                sigma_nlos_db=5.50,
                rice_k_db=20.0,             # log-power interp between 12 and 27.4 dB
                alt_min_m=80, alt_max_m=150,
            ),
            ChannelParameters(
                source="Interpolated from Khawaja (2019) Table V/VI Ref[54] L/C-band → 2.4 GHz, h=200m",
                data_origin="interpolated",
                environment="urban", frequency_ghz=2.4, altitude_m=200,
                n_los=1.67, n_nlos=3.40,   # 1.73 - 0.06 (higher altitude)
                sigma_los_db=2.85,
                sigma_nlos_db=5.20,
                rice_k_db=22.0,             # 20 + 2 dB (higher altitude)
                alt_min_m=150, alt_max_m=300,
            ),
            # ================================================================
            # INTERPOLATED: Urban, 5.8 GHz (DJI FPV / analog band)
            # Small extrapolation from C-band (5.3→5.8 GHz, Δf/f ≈ 9%)
            # ================================================================
            ChannelParameters(
                source="Interpolated from Khawaja (2019) Table V/VI Ref[54] C-band → 5.8 GHz",
                data_origin="interpolated",
                environment="urban", frequency_ghz=5.8, altitude_m=100,
                n_los=1.76, n_nlos=3.55,
                sigma_los_db=3.10,
                sigma_nlos_db=5.50,
                rice_k_db=27.0,             # close to measured 27.4 dB at 5.3 GHz
                alt_min_m=30, alt_max_m=300,
            ),
            # ================================================================
            # PHYSICS-ESTIMATED: Dense urban, 2.4 GHz
            # Khawaja (2019) Table VI Ref[47] (urban near airports): K = -5 to 10 dB.
            # Ref[47] documents heavily cluttered urban with intermittent LOS.
            # n_los set slightly above urban based on additional clutter;
            # K taken at the upper end of Ref[47] range (10 dB at h≥80m).
            # ================================================================
            ChannelParameters(
                source="Physics-estimated dense_urban — Khawaja (2019) Table VI Ref[47] K range, urban base",
                data_origin="physics-estimated",
                environment="dense_urban", frequency_ghz=2.4, altitude_m=50,
                n_los=1.82, n_nlos=3.80,
                sigma_los_db=3.20,
                sigma_nlos_db=6.00,
                rice_k_db=8.0,              # Ref[47]: K=-5..10 dB, lower end at h=50m
                alt_min_m=20, alt_max_m=80,
            ),
            ChannelParameters(
                source="Physics-estimated dense_urban — Khawaja (2019) Table VI Ref[47] K range, urban base",
                data_origin="physics-estimated",
                environment="dense_urban", frequency_ghz=2.4, altitude_m=100,
                n_los=1.78, n_nlos=3.70,
                sigma_los_db=3.10,
                sigma_nlos_db=5.80,
                rice_k_db=10.0,             # Ref[47]: K=-5..10 dB, upper end at h=100m
                alt_min_m=80, alt_max_m=200,
            ),
            # ================================================================
            # PHYSICS-ESTIMATED: Suburban, 2.4 GHz
            # Derived from urban Ref[54] 2.4 GHz interpolated values with reduced
            # clutter (suburban has fewer high-rise obstacles).
            # Ref[48] in Khawaja Table V (suburban/open, 3.1-5.3 GHz): n=2.54-3.04,
            # sigma=2.8-5.3 dB — appears to reflect low-altitude or near-ground
            # geometry; not used directly for elevated UAV scenarios.
            # ================================================================
            ChannelParameters(
                source="Physics-estimated suburban 2.4 GHz — scaled from Khawaja (2019) urban Ref[54]",
                data_origin="physics-estimated",
                environment="suburban", frequency_ghz=2.4, altitude_m=50,
                n_los=1.74, n_nlos=3.30,
                sigma_los_db=2.90,
                sigma_nlos_db=5.00,
                rice_k_db=20.0,
                alt_min_m=20, alt_max_m=80,
            ),
            ChannelParameters(
                source="Physics-estimated suburban 2.4 GHz — scaled from Khawaja (2019) urban Ref[54]",
                data_origin="physics-estimated",
                environment="suburban", frequency_ghz=2.4, altitude_m=100,
                n_los=1.70, n_nlos=3.10,
                sigma_los_db=2.75,
                sigma_nlos_db=4.80,
                rice_k_db=22.0,
                alt_min_m=80, alt_max_m=200,
            ),
            # ================================================================
            # PHYSICS-ESTIMATED: Rural, 2.4 GHz
            # Khawaja (2019) Table V Ref[98]: lightly hilly rural, L-band, h≈120m:
            # path loss model alpha=2.0 (free-space exponent), sigma=3.4 dB.
            # Khawaja Table V Ref[52]: hilly/mountain L-band n=1.3-1.8,
            # C-band n=1.0-1.8, sigma=2.2-3.9 dB (lower n due to reflecting terrain).
            # K-factor for rural not in Khawaja Table VI; estimated above urban
            # based on LOS-dominant open terrain.
            # ================================================================
            ChannelParameters(
                source="Physics-estimated rural 2.4 GHz — Khawaja (2019) Table V Ref[98] n=2.0 base",
                data_origin="physics-estimated",
                environment="rural", frequency_ghz=2.4, altitude_m=100,
                n_los=1.92, n_nlos=2.80,   # Ref[98] α=2.0 at h=120m; slight reduction for 2.4 GHz
                sigma_los_db=3.10,          # Ref[98] sigma=3.4 dB; slightly lower for open
                sigma_nlos_db=4.50,
                rice_k_db=24.0,             # estimated above urban (more LOS in rural)
                alt_min_m=50, alt_max_m=200,
            ),
            ChannelParameters(
                source="Physics-estimated rural 2.4 GHz — Khawaja (2019) Table V Ref[98], altitude-scaled",
                data_origin="physics-estimated",
                environment="rural", frequency_ghz=2.4, altitude_m=200,
                n_los=1.85, n_nlos=2.60,
                sigma_los_db=2.90,
                sigma_nlos_db=4.20,
                rice_k_db=26.0,
                alt_min_m=150, alt_max_m=400,
            ),
        ]

    @staticmethod
    def get_fhss_vulnerability_points() -> List[FHSSVulnerabilityPoint]:
        """
        Measured FHSS jamming effectiveness from published experiments.

        No verified open-access OcuSync-specific measurements found.
        Previously included "Beason et al. (2021)" and "Schiller et al. (2023)"
        have been removed — these references could not be confirmed as real
        publications.

        Parlin et al. (2018) IEEE ICMCIS (verified real) characterises FASST/ACCST
        protocols (Futaba/FrSky), not DJI OcuSync, and is excluded to prevent
        cross-protocol extrapolation.

        Returns empty list pending verified OcuSync-specific data.
        """
        return []

    @classmethod
    def get_best_channel_params(cls, environment: str, altitude_m: float,
                                frequency_ghz: float = 2.4) -> Optional[ChannelParameters]:
        """
        Return the most applicable channel parameters for given conditions.
        Priority: (1) environment + frequency match, (2) frequency match only,
        (3) any entry. Within each tier, closest frequency then closest altitude.
        """
        all_params = cls.get_a2g_channel_params()

        candidates = [p for p in all_params
                      if p.environment == environment
                      and abs(p.frequency_ghz - frequency_ghz) < 0.5]
        if not candidates:
            candidates = [p for p in all_params
                          if abs(p.frequency_ghz - frequency_ghz) < 0.5]
        if not candidates:
            candidates = all_params
        if not candidates:
            return None

        return min(candidates,
                   key=lambda p: (abs(p.frequency_ghz - frequency_ghz),
                                  abs(p.altitude_m - altitude_m)))

    @classmethod
    def compute_js_from_measurement(cls,
                                    power_w: float,
                                    jammer_gain_dbi: float,
                                    jammer_dist_m: float,
                                    signal_power_dbm: float,
                                    signal_gain_dbi: float,
                                    signal_dist_m: float,
                                    altitude_m: float,
                                    frequency_mhz: float = 2437.0,
                                    environment: str = 'urban',
                                    ) -> Tuple[float, float]:
        """
        Compute reference J/S using published A2G path loss exponents.

        Model: L(d) = L_fs(1m) + 10 * n_los * log10(d)  (ITU-R P.525-4)
        n_los is taken directly from Khawaja (2019) Table V — it represents
        the field-measured effective exponent for elevated UAV scenarios
        (LOS-dominant above ~50 m) and is NOT further weighted by LOS
        probability (doing so would double-count the altitude effect).

        Returns:
            (js_db, uncertainty_db) where uncertainty_db = sqrt(2)*sigma_los,
            representing independent shadow fading on jammer and signal paths.
        """
        freq_ghz = frequency_mhz / 1000.0
        power_dbm = 10.0 * np.log10(power_w * 1000.0)

        params = cls.get_best_channel_params(environment, altitude_m, freq_ghz)
        if params is None:
            n_eff, sigma = 2.0, 3.0
        else:
            n_eff = params.n_los
            sigma = params.sigma_los_db

        # ITU-R P.525-4: free-space path loss at 1 m reference distance
        l_fs_1m = 32.45 + 20.0 * np.log10(frequency_mhz) + 20.0 * np.log10(1e-3)

        l_j = l_fs_1m + 10.0 * n_eff * np.log10(max(jammer_dist_m, 1.0))
        l_s = l_fs_1m + 10.0 * n_eff * np.log10(max(signal_dist_m, 1.0))

        js_db = ((power_dbm + jammer_gain_dbi - l_j)
                 - (signal_power_dbm + signal_gain_dbi - l_s))
        uncertainty_db = np.sqrt(2.0) * sigma

        return float(js_db), float(uncertainty_db)

    @classmethod
    def get_a2g_training_scenarios(cls) -> List[dict]:
        """
        Training scenarios for the ML propagation correction model.
        Reference J/S values are derived from Khawaja (2019) Table V measured
        path loss exponents, not from the Friis free-space formula.

        Returns list of dicts:
            power_w, j_dist, s_dist, alt, gain, env,
            ref_js (dB), uncertainty_db (dB), source (citation)
        """
        SIG_POWER_DBM = 20.0
        SIG_GAIN_DBI  = 2.0
        FREQ_MHZ      = 2437.0

        base_scenarios = [
            # === Urban 2.4 GHz ===
            (10,    200,  2000,   50,  6, 'urban'),
            (10,    500,  5000,   50,  6, 'urban'),
            (10,    500,  5000,  100,  6, 'urban'),
            (10,   1000,  5000,  100,  6, 'urban'),
            (10,   1000,  8000,  150,  6, 'urban'),
            (10,   1500,  8000,  150,  6, 'urban'),
            (10,   2000,  8000,  200,  6, 'urban'),
            (10,   2000, 10000,  200,  6, 'urban'),
            (100,   500,  5000,  100, 10, 'urban'),
            (100,  1000,  8000,  100, 10, 'urban'),
            (100,  1000,  8000,  200, 10, 'urban'),
            (100,  2000,  8000,  200, 10, 'urban'),
            (100,  3000, 10000,  200, 10, 'urban'),
            (100,  4000, 10000,  250, 10, 'urban'),
            (500,  1000, 10000,  200, 15, 'urban'),
            (500,  2000, 10000,  300, 15, 'urban'),
            (500,  3000, 10000,  300, 15, 'urban'),
            (500,  5000, 15000,  400, 15, 'urban'),
            # === Suburban ===
            (10,    500,  5000,   50,  6, 'suburban'),
            (10,    500,  5000,  100,  6, 'suburban'),
            (10,   1000,  5000,  100,  6, 'suburban'),
            (100,  1000,  8000,  100, 10, 'suburban'),
            (100,  2000,  8000,  200, 10, 'suburban'),
            (500,  3000, 10000,  300, 15, 'suburban'),
            # === Rural ===
            (10,    500,  5000,  100,  6, 'rural'),
            (10,   1000,  5000,  100,  6, 'rural'),
            (10,   2000,  8000,  200,  6, 'rural'),
            (100,  1000,  8000,  100, 10, 'rural'),
            (100,  2000,  8000,  200, 10, 'rural'),
            (500,  3000, 10000,  300, 15, 'rural'),
            (500,  5000, 15000,  400, 15, 'rural'),
            # === Dense urban ===
            (10,    500,  5000,   50,  6, 'dense_urban'),
            (100,  1000,  8000,  100, 10, 'dense_urban'),
            (500,  2000, 10000,  200, 15, 'dense_urban'),
        ]

        result = []
        for (power_w, j_dist, s_dist, alt, gain, env) in base_scenarios:
            js_db, unc = cls.compute_js_from_measurement(
                power_w=power_w, jammer_gain_dbi=gain, jammer_dist_m=j_dist,
                signal_power_dbm=SIG_POWER_DBM, signal_gain_dbi=SIG_GAIN_DBI,
                signal_dist_m=s_dist, altitude_m=alt,
                frequency_mhz=FREQ_MHZ, environment=env,
            )
            params = cls.get_best_channel_params(env, alt, 2.4)
            source = params.source if params else "Khawaja (2019) Table V — default"
            result.append({
                'power_w': power_w, 'j_dist': j_dist, 's_dist': s_dist,
                'alt': alt, 'gain': gain, 'env': env,
                'ref_js': js_db, 'uncertainty_db': unc, 'source': source,
            })
        return result

    @classmethod
    def get_surrogate_anchor_points(cls,
                                    signal_power_dbm: float = 20.0,
                                    signal_gain_dbi: float = 2.0,
                                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (X, Y) anchor data for surrogate model seeding.

        X shape: (n, 6) — [jammer_power_dbm, jammer_distance_m,
                            jammer_antenna_gain_dbi, signal_distance_m,
                            altitude_m, path_loss_std]
        Y shape: (n, 3) — [mean_js_db, std_js_db, success_probability]
        """
        scenarios = cls.get_a2g_training_scenarios()
        X_rows, Y_rows = [], []

        for s in scenarios:
            power_dbm = 10.0 * np.log10(s['power_w'] * 1000.0)
            js_db, unc = cls.compute_js_from_measurement(
                power_w=s['power_w'], jammer_gain_dbi=s['gain'],
                jammer_dist_m=s['j_dist'],
                signal_power_dbm=signal_power_dbm, signal_gain_dbi=signal_gain_dbi,
                signal_dist_m=s['s_dist'], altitude_m=s['alt'],
                frequency_mhz=2437.0, environment=s['env'],
            )
            params = cls.get_best_channel_params(s['env'], s['alt'], 2.4)
            sigma = params.sigma_los_db if params else 3.0

            X_rows.append([
                power_dbm, s['j_dist'], s['gain'], s['s_dist'], s['alt'], sigma,
            ])
            std_js = unc / np.sqrt(2.0)
            success_prob = float(np.clip(0.5 + 0.04 * (js_db - 20.0), 0.05, 0.99))
            Y_rows.append([js_db, std_js, success_prob])

        return np.array(X_rows, dtype=float), np.array(Y_rows, dtype=float)


if __name__ == "__main__":
    print("=" * 72)
    print("Literature Dataset — Summary (Khawaja 2019 arXiv:1801.01656)")
    print("=" * 72)

    channel_params = LiteratureDataset.get_a2g_channel_params()
    print(f"\nChannel parameter sets: {len(channel_params)}")
    fmt = "  [{:5s}] {:12s} {:5.2f} GHz  h={:4.0f}m  n={:.2f}  K={:5.1f}dB  s={:.2f}dB"
    for p in channel_params:
        print(fmt.format(
            p.data_origin[:5], p.environment, p.frequency_ghz,
            p.altitude_m, p.n_los, p.rice_k_db, p.sigma_los_db))

    fhss_pts = LiteratureDataset.get_fhss_vulnerability_points()
    print(f"\nFHSS vulnerability points: {len(fhss_pts)}"
          " (no verified OcuSync data — see module docstring)")

    scenarios = LiteratureDataset.get_a2g_training_scenarios()
    print(f"\nTraining scenarios: {len(scenarios)}")
    for s in scenarios[:5]:
        print(f"  {s['env']:12s}  P={s['power_w']:4.0f}W"
              f"  d_J={s['j_dist']:5.0f}m  h={s['alt']:3.0f}m"
              f"  J/S={s['ref_js']:6.1f}dB  unc={s['uncertainty_db']:.1f}dB")
    print(f"  ... ({len(scenarios) - 5} more)")
