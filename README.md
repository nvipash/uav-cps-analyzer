# UAV-CPS-Analyzer

**Програмний комплекс моделювання кіберфізичних систем безпілотних літальних апаратів**

*Software Complex for Modeling Cyber-Physical Systems of Unmanned Aerial Vehicles*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Version](https://img.shields.io/badge/version-1.2.0-green.svg)]()

## Опис

UAV-CPS-Analyzer — комплексний програмний інструмент для моделювання та аналізу надійності кіберфізичних систем безпілотних літальних апаратів в умовах електромагнітних завад. Інтегрує 13 фізичних моделей розповсюдження, 11 модулів штучного інтелекту, формальну валідацію за ASME V&V 20 та 40 еталонних сценаріїв з відкритих науково-технічних джерел.

**Автори:** Новицький П.С., Степаняк М.В., Національний університет «Львівська політехніка», 2025-2026

## Основні можливості

- **13 моделей розповсюдження** — Friis, COST 231-Hata, Rice fading, Al-Hourani A2G, BER/PER soft degradation, Doppler, ITU-R P.676 atmospheric absorption, urban multi-path correction
- **Монте-Карло симуляція** — QMC (Sobol), antithetic variates, адаптивна збіжність, просторова кореляція (Gudmundson), N=10 000
- **Емуляція FHSS** — DJI OcuSync 2.0/3.0/3+, 5 стратегій глушіння, LFSR-16
- **Аналіз C-UAS** — Dempster-Shafer 4-сенсорна фузія, Парето-оптимізація сенсорних комплексів
- **11 AI/ML модулів** — сурогатна модель (2400× прискорення), RL-глушник, ансамблевий класифікатор (99.6%), ко-еволюція, PCE, мережі глушників, роєві сценарії, траєкторії
- **Мережі глушників** — координоване об'єднання потужностей, генетична оптимізація розміщення
- **Сценарії роїв** — 4 типи атак (cooperative/decentralized/kamikaze/decoy+strike)
- **Формальна валідація** — ASME V&V 20, 40 еталонних випадків з 7 джерел

## Встановлення

```bash
pip install numpy scipy matplotlib scikit-learn
```

Python 3.9+, без додаткових залежностей.

## Запуск

```bash
# Повний аналіз (23 кроки, ~130 с)
python run_analysis.py

# Окремі модулі
python validation.py
python ai_threat_classifier.py
python multi_jammer_coordination.py
```

## Структура проекту

```
UAV-CPS-Analyzer/
│
├── Core modules (Layer 1 — Physics)
│   ├── propagation_models.py   # 13 RF/propagation models
│   ├── fhss_emulator.py        # FHSS protocol emulation
│   ├── monte_carlo_engine.py   # MC + QMC + antithetic, N=10 000
│   ├── cps_analyzer.py         # Dempster-Shafer sensor fusion, C-UAS
│   ├── config.py               # Drone DB, jammer specs, environments
│   └── visualization.py        # 5 publication figures (300 DPI PDF+PNG)
│
├── Statistical & Validation (Layer 2)
│   ├── sensitivity.py          # Sobol S1/ST, Morris elementary effects
│   ├── validation.py           # ASME V&V 20, 40 cases, per-domain MAPE
│   └── reporting.py            # LaTeX booktabs tables, Markdown reports
│
├── AI/ML & Advanced (Layer 3)
│   ├── ai_surrogate.py         # MLP/GP/ensemble surrogate, R²=0.965, 2400×
│   ├── ai_optimizer.py         # Bayesian optimization (DE + surrogate)
│   ├── ai_propagation_correction.py  # Gradient Boosting residual correction
│   ├── ai_adaptive_jamming.py  # Q-learning RL jammer agent
│   ├── ai_threat_classifier.py # MLP+RF+GBT ensemble, 99.6% accuracy
│   ├── ai_coevolution.py       # Adversarial AI-vs-AI, Nash equilibrium
│   ├── ai_uncertainty_active.py # PCE (Hermite order-3), active learning
│   ├── multi_jammer_coordination.py  # Jammer networks, genetic optimizer
│   ├── swarm_scenarios.py      # UAV swarm attacks (4 types)
│   └── trajectory_scenarios.py # Time-varying trajectories (4 types)
│
├── run_analysis.py             # 23-step integrated pipeline
│
├── docs/
│   ├── en/README.md            # Full English documentation
│   └── ua/README.md            # Повна українська документація
│
└── output_v1.2/                # Generated artifacts
    ├── fig_*.pdf / fig_*.png   # 5 publication figures (300 DPI)
    ├── table3.tex / table4.tex / table7.tex
    └── summary_report.md
```

## Ключові результати (v1.2)

| Модуль | Результат |
|--------|-----------|
| Surrogate model | R²=0.965, RMSE=3.32 dB, **2400× speedup** |
| BER/PER threshold | Реальний поріг J/S ≈ −10 dB (не +10 dB) → **100× менша потужність** |
| FHSS protection | Narrowband втрачає **97% ефективності** проти FHSS |
| Nash equilibrium | frequency_diversity **43.5%** — домінантна стратегія захисника |
| Threat classifier | **99.6%** accuracy (vs 32% Dempster-Shafer) |
| RL jammer (adversarial) | **+112%** vs best fixed strategy |
| Spatial correlation | Ігнорування → **+9–30%** завищення ширини CI |
| Validation (close range) | MAPE **7.9%**, PASS rate **100%** (ASME V&V 20) |

## Документація

- [English Documentation](docs/en/README.md)
- [Українська документація](docs/ua/README.md)

## Цитування

```bibtex
@software{uav_cps_analyzer_2025,
  author    = {Novitskyi, Pavlo and Stepaniak, Maksym},
  title     = {UAV-CPS-Analyzer: Software Complex for Cyber-Physical Systems
               Analysis of UAV Communication Reliability},
  year      = {2025},
  institution = {Lviv Polytechnic National University},
  version   = {1.2.0}
}
```

## Ліцензія

GNU General Public License v3.0 — see LICENSE file.
