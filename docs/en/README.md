# UAV-CPS-Analyzer — English Documentation

## Overview

UAV-CPS-Analyzer is a comprehensive software complex for modeling and analyzing cyber-physical systems of unmanned aerial vehicles (UAVs) under electromagnetic jamming. It combines classical RF propagation models, formal statistical methodology, AI/ML techniques, multi-jammer coordination, swarm scenario analysis, and adversarial co-evolution into a single integrated tool.

**Authors:** Novitskyi P.S., Stepaniak M.V., Lviv Polytechnic National University, 2025-2026
**Version:** 1.2.0
**License:** MIT
**Language:** Python 3.9+

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [Core Modules](#core-modules)
4. [AI/ML Modules](#aiml-modules)
5. [Advanced Modules](#advanced-modules)
6. [Running the Analysis](#running-the-analysis)
7. [Output Artifacts](#output-artifacts)
8. [Validation Framework](#validation-framework)
9. [Extension Guide](#extension-guide)

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib scikit-learn

# Run the comprehensive analysis pipeline
python run_analysis.py

# Run individual modules
python validation.py
python ai_threat_classifier.py
python multi_jammer_coordination.py
```

Expected runtime: 140-200 seconds depending on hardware.

## Architecture

The system has three layers:

### Layer 1: Physics Foundation (5 modules)

- **propagation_models.py** — RF propagation: Friis, COST 231-Hata, Rice fading, shadow fading, Al-Hourani A2G probabilistic model, BER/PER soft degradation, Doppler effects, atmospheric absorption (ITU-R P.676), urban multi-path correction, antenna patterns (Cosine), heavy-tailed distributions (Student's t, mixture)
- **fhss_emulator.py** — FHSS protocol emulation: LFSR-based hop sequences mimicking DJI OcuSync 2.0/3.0/3+ behavior; 5 jamming strategies (broadband, narrowband, sweep, follower, protocol-aware)
- **monte_carlo_engine.py** — Parallel Monte Carlo with bootstrap CIs on CI bounds, formal convergence criterion (5% relative half-width), QMC (Sobol) and antithetic variance reduction, spatial correlation (Gudmundson model), reproducibility via fixed seeds
- **cps_analyzer.py** — Cyber-Physical Systems analysis: Dempster-Shafer sensor fusion (RF/Radar/Acoustic/EO-IR), C-UAS architecture, cost-effectiveness Pareto with multi-constraint optimization, stochastic dominance ranking
- **config.py** — Configuration: drone database (DJI Mavic 3, Mini 4 Pro, FPV, fiber-optic), jammer specifications, 5 environment presets (dense_urban, urban, suburban, rural, open_field)

### Layer 2: Statistical & Validation (3 modules)

- **sensitivity.py** — Global sensitivity analysis: Sobol indices (first-order S1, total-order ST), Morris method elementary effects screening, parallelized via multiprocessing
- **validation.py** — ASME V&V 20 framework: 40 validation cases from 7 sources (Adamy 2015, Poisel 2011, Skolnik 2008, FCC ID, Beason 2021, Schiller 2023, ITU-R P.1411); per-domain MAPE breakdown (close/medium/long range, regulatory, behavioral); 7 internal consistency checks
- **reporting.py** — LaTeX table generation (booktabs format), Markdown summary reports with timestamps and reproducibility metadata

### Layer 3: AI/ML & Advanced (8 modules)

- **ai_surrogate.py** — Neural network surrogate model (MLP/GP/ensemble) trained on Monte Carlo outputs; achieves R²≥0.96 with 200 training points and 2000-3000× speedup
- **ai_optimizer.py** — Bayesian optimization (differential evolution + surrogate) for jammer/sensor placement; sensor suite optimization with budget constraints
- **ai_propagation_correction.py** — ML correction of propagation model: Gradient Boosting on residuals from 20+ reference points
- **ai_adaptive_jamming.py** — Reinforcement learning agent (Q-learning) for jamming strategy selection; supports adversarial environment and multi-objective reward (jam_rate + power_efficiency + stealth)
- **ai_threat_classifier.py** — Ensemble NN (MLP + Random Forest + Gradient Boosting) for threat classification; 24-feature input including time-series, spectral, behavioral; achieves 99.6% accuracy vs 32% for pure Dempster-Shafer
- **ai_uncertainty_active.py** — Multi-output uncertainty propagation with proper Polynomial Chaos Expansion (Hermite polynomials, order 3); active learning loop with GP uncertainty sampling
- **multi_jammer_coordination.py** *(new)* — Coordinated multi-jammer networks: power combining at target, coverage heatmaps, genetic algorithm optimizer for placement
- **swarm_scenarios.py** *(new)* — UAV swarm attacks: 4 types (cooperative, decentralized, kamikaze, decoy+strike); saturation curve analysis vs jammer network capacity
- **ai_coevolution.py** *(new)* — Adversarial AI-vs-AI training: simultaneous Q-learning of jammer and FHSS defender; Nash equilibrium estimation; cycle detection
- **trajectory_scenarios.py** *(new)* — Time-varying UAV trajectories: linear approach, fly-by, circular orbit, Bezier evasive; J/S evolution over time
- **visualization.py** — Publication-quality figures (5 plots, 300 DPI, PDF + PNG)

## Core Modules

### `propagation_models.py`

Implements 13 propagation/RF models:

| Model | Standard | Range |
|-------|----------|-------|
| FriisModel | ITU-R P.525-4 | Free-space, all distances |
| COST231HataModel | COST 231 | Urban/suburban, 1.5-2.4 GHz |
| RiceFadingModel | Rice 1944 | LOS multipath |
| ShadowFading | Log-normal/Student-t/mixture | Large-scale fading |
| AltitudeDependentModel | Linear blending | Combined urban/free-space |
| AlHouraniA2GModel | Al-Hourani 2014 | Probabilistic A2G with LOS prob |
| BERModel | Proakis 2007 | BPSK/QPSK/16-QAM/64-QAM |
| DopplerModel | Classical | Frequency shift, coherence time |
| AtmosphericAbsorption | ITU-R P.676 | Oxygen + water vapor |
| UrbanMultiPathCorrection | Empirical | NLOS additional loss |
| ModulationType | Enum | Modulation schemes |
| CosinePattern | Cosine taper | Directional antennas |
| OmnidirectionalPattern | Constant | Omnidirectional antennas |

### `fhss_emulator.py`

FHSS emulation parameters:
- 40 channels in 2.4 GHz band (2400-2483.5 MHz)
- Channel bandwidth: 2 MHz (2.5 MHz for OcuSync 3+)
- Hop rate: 500 Hz (dwell time 2 ms)
- LFSR-16 with maximum-length taps [16, 15, 13, 4]

Note: This is a behavioral emulation based on public regulatory data (FCC ID),
manufacturer specifications (DJI), and reverse-engineering studies (Beason 2021,
Schiller 2023). The actual OcuSync protocol is proprietary.

### `monte_carlo_engine.py`

Three sampling methods:
- **Standard MC** (`run_simulation`) — basic pseudo-random
- **QMC (Sobol)** (`run_simulation_qmc`) — low-discrepancy quasi-random for better convergence
- **Antithetic variates** (`run_simulation_antithetic`) — paired sampling for variance reduction

Adaptive convergence (`run_simulation_adaptive`) auto-stops when relative CI half-width drops below threshold (default 5%).

## AI/ML Modules

### Surrogate Model (`ai_surrogate.py`)

Trains neural network on Monte Carlo outputs to enable instant J/S prediction:

```python
from ai_surrogate import SurrogateTrainer, SurrogatePredictor

trainer = SurrogateTrainer(mc_iterations_per_point=500)
X, Y = trainer.generate_training_data(n_points=200)
model, sx, sy, metrics = trainer.train(X, Y, model_type='ensemble')
predictor = SurrogatePredictor(model, sx, sy)
result = predictor.predict(power_dbm=40, distance=500, ...)
# Result in <1ms vs 1000ms for full MC
```

### Threat Classifier (`ai_threat_classifier.py`)

24-feature ensemble classifier:
- Classical sensor features (10): RF, Radar, Acoustic, EO/IR detections + parameters
- Behavioral features (6): RCS variation, velocity variation, altitude variation, heading variation, flight smoothness, reaction to jamming
- Spectral RF features (5): bandwidth, modulation complexity, hop rate, burstiness, spectral kurtosis
- Acoustic spectral features (3): fundamental frequency, harmonic ratio, Doppler shift

### Co-evolution (`ai_coevolution.py`)

Multi-agent RL where both jammer and FHSS defender learn simultaneously:
- Jammer actions: 4 strategies (broadband, narrowband, sweep, follower)
- FHSS defenses: 5 postures (uniform_hopping, avoid_jammed_channels, high_modulation, spread_pattern, frequency_diversity)
- Zero-sum game: Q-learning for both agents
- Outputs: Nash equilibrium strategy mix, convergence dynamics, cycle detection

## Advanced Modules

### Multi-Jammer Coordination

`MultiJammerSimulator` combines power from multiple jammers in linear domain:

```python
from multi_jammer_coordination import (
    MultiJammerSimulator, JammerNetwork, JammerNode, create_default_networks
)

networks = create_default_networks()
sim = MultiJammerSimulator()
coverage = sim.coverage_map(networks['5_mesh'])
# Returns 2D heatmap of P(jam) over 6km×6km area
```

Pre-defined networks:
- `1_portable` — single 10W jammer (baseline)
- `3_triangle` — 3 jammers in triangle (perimeter defense)
- `5_mesh` — 5-jammer mesh (central + 4 directional)
- `2_stationary` — 2 high-power phased arrays

`JammerNetworkOptimizer` uses genetic algorithm to find optimal placement.

### Swarm Scenarios

4 swarm types implemented:
- **COOPERATIVE** — coordinated, requires shared comm; jammed → lost
- **DECENTRALIZED** — independent agents, more resilient
- **KAMIKAZE** — autonomous, harder to disable
- **DECOY_STRIKE** — decoys absorb jamming for real attackers

```python
from swarm_scenarios import SwarmAttackSimulator, SwarmConfig, SwarmType

config = SwarmConfig(swarm_type=SwarmType.COOPERATIVE, n_uavs=20)
sim = SwarmAttackSimulator()
result = sim.simulate_attack(swarm, jammer_network)
# Returns survival rate, time to breach, peak active count, saturation flag
```

### Trajectory Scenarios

4 standard trajectories:
- **linear_approach** — straight-line attack toward jammer
- **fly_by** — pass at offset distance
- **circular_orbit** — sustained surveillance
- **bezier_evasive** — curved evasive maneuver

## Running the Analysis

The `run_analysis.py` script runs all 23 analytical steps:

```bash
python run_analysis.py
```

Steps:
1. Table 3 — Core MC scenarios (QMC variance reduction)
2. Propagation model comparison (Al-Hourani vs legacy)
3. Table 4 — FHSS strategy effectiveness
4. BER/PER soft degradation curve
5. Doppler effect analysis
6. Antenna pattern impact
7. Multi-environment comparison
8. Sensor fusion & detection rates
9. Formal validation (ASME V&V 20) + per-domain
10. OAT sensitivity + cost-effectiveness Pareto
11. AI Surrogate Model
12. AI Bayesian Optimization
13. AI Propagation Correction
14. AI Adaptive Jamming (RL)
15. AI Threat Classifier (ensemble)
16. Long-range correction impact
17. Adversarial RL + Multi-objective
18. Multi-output uncertainty + Pareto constraints
19. Variance reduction comparison (MC vs QMC vs Antithetic)
20. Spatial correlation impact
21. PCE-based tail estimation
22. Active learning for ML correction
23. Time-varying trajectory scenarios

## Output Artifacts

Generated in `output/`:
- 5 publication figures (PDF + PNG, 300 DPI)
- 3 LaTeX tables (table3.tex, table4.tex, table7.tex)
- Markdown summary report (summary_report.md)
- Validation case-by-case results

## Validation Framework

**ASME V&V 20** methodology:
- E = |model_prediction - reference_value|
- u_val = sqrt(u_model² + u_exp²)
- PASS if |E| < u_val

40 validation cases organized into 5 domains:
- close_range (4 cases): MAPE 7-12%, PASS rate 100%
- medium_range (14 cases): MAPE 30-40%
- long_range (7 cases): MAPE 80-100% — needs calibration
- regulatory (5 cases): FCC ID measurements
- behavioral (4 cases): FHSS protocol verification

## Extension Guide

To add a new propagation model:
1. Inherit from `PropagationModel` ABC in `propagation_models.py`
2. Implement `path_loss(distance_m, **kwargs)` method
3. Add to factory function `get_model()`
4. Add validation cases in `validation.py`

To add a new validation case:
1. Add tuple to `_define_validation_cases()` in `validation.py`
2. Specify domain field for per-domain analysis
3. Use `_make_case()` helper

To add a new AI module:
1. Create `ai_<name>.py` file
2. Export classes via `__init__.py`
3. Add Step in `run_analysis.py`

## Citation

```bibtex
@software{uav_cps_analyzer_2025,
  author = {Novitskyi, Pavlo and Stepaniak, Maksym},
  title = {UAV-CPS-Analyzer: Software Complex for Cyber-Physical Systems Analysis
           of UAV Communication Reliability},
  year = {2025},
  institution = {Lviv Polytechnic National University},
  url = {https://github.com/your-username/uav-cps-analyzer},
  version = {1.2.0}
}
```

## License

MIT License — see LICENSE file.

## Contact

- **Author:** Pavlo Novitskyi (novitskyi@lp.edu.ua)
- **Co-author:** Maksym Stepaniak
- **Institution:** Lviv Polytechnic National University, Ukraine
