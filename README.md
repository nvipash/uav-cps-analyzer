# UAV-CPS-Analyzer

**Програмний комплекс моделювання кіберфізичних систем безпілотних літальних апаратів**

*Software Complex for Modeling Cyber-Physical Systems of Unmanned Aerial Vehicles*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Опис

UAV-CPS-Analyzer — це програмний комплекс для моделювання та аналізу надійності кіберфізичних систем безпілотних літальних апаратів в умовах електромагнітних завад.

### Основні можливості:

- **Моделювання поширення сигналу** — COST 231-Hata, Friis, висотно-залежна комбінована модель
- **Емуляція FHSS-протоколів** — OcuSync 2.0/3.0/3+, аналіз стратегій jamming
- **Монте-Карло симуляція** — паралельні обчислення, статистичний аналіз, довірчі інтервали
- **Аналіз C-UAS архітектури** — sensor fusion (Dempster-Shafer), багаторівнева система захисту
- **Візуалізація** — публікаційні графіки для наукових статей

## Встановлення

### Вимоги

- Python 3.9+
- NumPy 1.21+
- SciPy 1.7+
- Matplotlib 3.4+

### Інсталяція

```bash
# Клонування репозиторію
git clone https://github.com/your-repo/UAV-CPS-Analyzer.git
cd UAV-CPS-Analyzer

# Встановлення залежностей
pip install numpy scipy matplotlib

# Запуск тестів
python -m pytest tests/
```

## Використання

### Швидкий старт

```bash
# Повний аналіз з генерацією графіків
python src/main.py

# Тільки сценарії з Таблиці 3 (Monte Carlo)
python src/main.py --scenario table3 -n 10000

# Тільки FHSS аналіз
python src/main.py --fhss

# Генерація графіків
python src/main.py --figures -o output/
```

### Програмне використання

```python
from src import (
    MonteCarloEngine, SimulationParams,
    OcuSyncProtocol, JammingEffectivenessAnalyzer,
    SensorFusion, ThreatType
)

# Monte Carlo симуляція
engine = MonteCarloEngine()
params = SimulationParams(
    jammer_power_dbm=40,      # 10W
    jammer_distance_m=500,
    signal_distance_m=5000,
    altitude_m=100
)
result = engine.run_simulation(params, n_iterations=10000)

print(f"J/S: {result.mean_js_db:.1f} ± {result.std_js_db:.1f} dB")
print(f"95% CI: [{result.ci_95_lower:.1f}, {result.ci_95_upper:.1f}]")

# FHSS аналіз
protocol = OcuSyncProtocol(version="3.0")
analyzer = JammingEffectivenessAnalyzer(protocol)
results = analyzer.compare_strategies()

# Sensor fusion
fusion = SensorFusion()
detection_rate = fusion.get_detection_rate(ThreatType.RF_CONTROLLED, 1000)
```

## Структура проекту

```
UAV-CPS-Analyzer/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── main.py               # Main entry point
│   ├── propagation_models.py # Signal propagation models
│   ├── fhss_emulator.py      # FHSS protocol emulation
│   ├── monte_carlo_engine.py # Monte Carlo simulation
│   ├── cps_analyzer.py       # CPS analysis, sensor fusion
│   ├── visualization.py      # Figure generation
│   └── config.py             # Configuration, databases
├── tests/                    # Unit tests
├── docs/                     # Documentation
├── output/                   # Generated figures
├── README.md
└── config.json              # Configuration file
```

## Модулі

### propagation_models.py

Моделі поширення радіосигналу:
- `FriisModel` — втрати у вільному просторі (ITU-R P.525-4)
- `COST231HataModel` — міська модель (COST 231)
- `RiceFadingModel` — модель замирань Райса
- `AltitudeDependentModel` — комбінована висотно-залежна модель

### fhss_emulator.py

Емуляція FHSS-протоколів:
- `OcuSyncProtocol` — емуляція DJI OcuSync (40 каналів, 500 Гц)
- `JammingEffectivenessAnalyzer` — аналіз ефективності стратегій jamming
- `ChannelSimulator` — симуляція каналу зв'язку

### monte_carlo_engine.py

Статистичний аналіз:
- `MonteCarloEngine` — паралельна симуляція Монте-Карло
- `ScenarioSimulator` — симуляція сценаріїв зі статті
- Аналіз чутливості (tornado diagram)

### cps_analyzer.py

Аналіз кіберфізичних систем:
- `SensorFusion` — злиття даних сенсорів (Dempster-Shafer)
- `CUASArchitecture` — багаторівнева архітектура C-UAS
- Моделі сенсорів: RF, Radar, Acoustic, EO/IR

## Результати

Програмний комплекс відтворює результати з наукової статті:

| Сценарій | J/S (дБ) | 95% ДІ | P(success) |
|----------|----------|--------|------------|
| Portable 10W, 500m | 38.2 | [28.8, 47.3] | 100% |
| Portable 10W, 1000m | 27.6 | [18.3, 36.5] | 100% |
| Mobile 100W, 2000m | 27.1 | [17.9, 36.2] | 100% |
| Stationary 500W, 3000m | 29.1 | [19.7, 38.3] | 100% |

## Автори

- **Новіцький П.С.** — Львівська політехніка
- **Степаняк М.В.** — науковий керівник

## Література

1. Menouar H. et al., "UAV-Enabled Intelligent Transportation Systems," IEEE Commun. Mag., 2017.
2. Torrieri D., Principles of Spread-Spectrum Communication Systems, Springer, 2018.
3. Metropolis N., Ulam S., "The Monte Carlo Method," J. Amer. Stat. Assoc., 1949.
4. De Miguel-Vela C. et al., "Counter-UAS Sensors," Sensors, 2021.

## Ліцензія

MIT License. Див. [LICENSE](LICENSE) для деталей.

## Цитування

```bibtex
@article{novitskyi2025uav,
  title={Software Complex for Modeling Cyber-Physical Systems of UAVs},
  author={Novitskyi, P.S. and Stepaniak, M.V.},
  journal={Lviv Polytechnic National University},
  year={2025}
}
```
