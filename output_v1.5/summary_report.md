# UAV-CPS-Analyzer: Statistical Summary Report
Generated: 2026-05-03 18:28:50
Software version: 1.0.0

## Monte Carlo Simulation Results

| Scenario | Mean J/S (dB) | Std (dB) | 95% CI (dB) | P(jam) | Converged | N |
|----------|--------------|---------|-------------|--------|-----------|---|
| Portable 10W, 500m | 51.0 | 5.6 | [40.6, 62.4] | 100% | No | 10,000 |
| Portable 10W, 1000m | 43.4 | 5.6 | [32.9, 54.7] | 100% | No | 10,000 |
| Mobile 100W, 2000m | 51.0 | 5.6 | [40.5, 62.3] | 100% | No | 10,000 |
| Stationary 500W, 3000m | 56.0 | 5.7 | [45.5, 67.5] | 100% | No | 10,000 |

### Bootstrap CI Uncertainty
- **Portable 10W, 500m**: CI lower [40.28, 40.76], CI upper [62.00, 62.76]
- **Portable 10W, 1000m**: CI lower [32.68, 33.10], CI upper [54.48, 55.09]
- **Mobile 100W, 2000m**: CI lower [40.28, 40.67], CI upper [62.03, 62.69]
- **Stationary 500W, 3000m**: CI lower [45.28, 45.75], CI upper [67.26, 68.02]

### Sample Size Justification
- **Portable 10W, 500m**: Required N = 9,604 (for 0.1 dB precision at 95% confidence), used N = 10,000
- **Portable 10W, 1000m**: Required N = 9,604 (for 0.1 dB precision at 95% confidence), used N = 10,000
- **Mobile 100W, 2000m**: Required N = 9,604 (for 0.1 dB precision at 95% confidence), used N = 10,000
- **Stationary 500W, 3000m**: Required N = 9,604 (for 0.1 dB precision at 95% confidence), used N = 10,000

## FHSS Effectiveness Analysis

| Strategy | Static | FHSS | Power Multiplier |
|----------|--------|------|-----------------|
| broadband | 65% | 100% | x1 |
| narrowband | 95% | 2% | x40 |
| sweep | 80% | 12% | x5 |
| follower | 98% | 75% | x3 |
| protocol | 85% | 85% | x2 |

## Model Validation (ASME V&V 20)

| Case | Model (dB) | Reference (dB) | Error | MAPE | V&V 20 |
|------|-----------|---------------|-------|------|--------|
| Adamy: Tactical 5W, 200m | 35.0 | 50.0 | 15.04 | 30.1% | FAIL |
| Adamy: Tactical 5W, 500m | 24.3 | 40.0 | 15.70 | 39.3% | FAIL |
| Adamy: Portable 10W, 500m | 33.1 | 45.0 | 11.92 | 26.5% | FAIL |
| Adamy: Portable 10W, 1km | 25.5 | 36.0 | 10.51 | 29.2% | FAIL |
| Adamy: Portable 10W, 2km | 18.8 | 28.0 | 9.25 | 33.0% | FAIL |
| Adamy: Tactical 20W, 1km | 30.5 | 40.0 | 9.46 | 23.6% | FAIL |
| Adamy: Wideband 200W, 5km | 35.4 | 32.0 | 3.43 | 10.7% | PASS |
| Adamy: Wideband 200W, 10km | 32.6 | 22.0 | 10.64 | 48.4% | FAIL |
| Poisel: Voice link 50W, 1km | 41.7 | 42.0 | 0.26 | 0.6% | PASS |
| Poisel: Data link 100W, 2km | 37.1 | 38.0 | 0.90 | 2.4% | PASS |
| Poisel: Mesh attack 100W, 3km | 37.2 | 32.0 | 5.19 | 16.2% | PASS |
| Poisel: Helicopter 500W, 5km | 55.4 | 45.0 | 10.41 | 23.1% | FAIL |
| Poisel: Aircraft 5kW, 20km | 55.2 | 35.0 | 20.20 | 57.7% | FAIL |
| Poisel: Mobile 100W, 1km | 39.7 | 46.0 | 6.25 | 13.6% | PASS |
| Poisel: Mobile 100W, 4km | 30.5 | 28.0 | 2.50 | 8.9% | PASS |
| Skolnik: Stationary 500W, 3km | 47.3 | 45.0 | 2.28 | 5.1% | PASS |
| Skolnik: Stationary 1kW, 5km | 48.1 | 40.0 | 8.06 | 20.1% | FAIL |
| Skolnik: Array 2kW, 10km | 54.9 | 35.0 | 19.92 | 56.9% | FAIL |
| Skolnik: Phased-array 5kW, 15km | 58.9 | 32.0 | 26.90 | 84.1% | FAIL |
| FCC: DJI Mavic3 ch1 2.4G | 27.9 | 32.0 | 4.13 | 12.9% | FAIL |
| FCC: DJI Mavic3 ch20 2.4G | 33.0 | 20.0 | 12.99 | 65.0% | FAIL |
| FCC: DJI Mavic3 ch40 5.8G | 33.0 | 10.0 | 23.00 | 230.0% | FAIL |
| FCC: DJI Mini4 Pro 2.4G | 37.3 | 18.0 | 19.28 | 107.1% | FAIL |
| FCC: DJI FPV 5.8G | 43.1 | 28.0 | 15.08 | 53.9% | FAIL |
| Brust: Portable detect 1km | 35.4 | 32.0 | 3.38 | 10.6% | PASS |
| Brust: Mobile detect 2km | 47.6 | 36.0 | 11.60 | 32.2% | FAIL |
| Park: Anti-drone tactical | 46.2 | 34.0 | 12.19 | 35.8% | FAIL |
| Wang: C-UAS performance 3km | 55.2 | 34.0 | 21.25 | 62.5% | FAIL |
| Wang: C-UAS swarm scenario | 59.9 | 40.0 | 19.93 | 49.8% | FAIL |
| ITU-R P.1411: short-range | 36.2 | 30.0 | 6.20 | 20.7% | FAIL |
| ITU-R: UMa NLOS 1km | 40.4 | 32.0 | 8.40 | 26.3% | FAIL |
| ITU-R: UMi LOS 500m | 31.4 | 36.0 | 4.59 | 12.8% | PASS |
| ITU-R: SUI suburban 2km | 41.7 | 30.0 | 11.74 | 39.1% | FAIL |
| ITU-R: rural LOS 5km | 54.1 | 28.0 | 26.12 | 93.3% | FAIL |
| ITU-R: dense urban 1km | 43.4 | 24.0 | 19.44 | 81.0% | FAIL |
| ITU-R: high-altitude 10km | 58.4 | 22.0 | 36.41 | 165.5% | FAIL |
| Khawaja: urban h=50m, d_J=500m | 52.3 | 41.6 | 10.73 | 25.8% | FAIL |
| Khawaja: urban h=100m, d_J=1km | 43.5 | 36.1 | 7.42 | 20.5% | FAIL |
| Khawaja: urban h=200m, d_J=2km | 41.2 | 34.1 | 7.14 | 20.9% | FAIL |
| Khawaja: suburban h=50m, d_J=500m | 54.0 | 41.4 | 12.64 | 30.5% | FAIL |
| Khawaja: suburban h=100m, d_J=1km | 46.4 | 35.9 | 10.49 | 29.2% | FAIL |
| Khawaja: rural h=100m, d_J=500m | 63.0 | 43.2 | 19.77 | 45.8% | FAIL |
| Khawaja: rural h=200m, d_J=1km | 60.5 | 40.7 | 19.79 | 48.6% | FAIL |
| Khawaja: urban h=100m, mobile 100W | 56.1 | 48.4 | 7.68 | 15.9% | FAIL |
| Khawaja: suburban h=50m, d_J=1km | 44.8 | 36.2 | 8.63 | 23.8% | FAIL |
| Khawaja: urban 5.8GHz h=100m | 51.2 | 41.6 | 9.57 | 23.0% | FAIL |
| Khawaja: rural h=100m, d_J=1km | 59.0 | 41.3 | 17.68 | 42.8% | FAIL |
| Khawaja: suburban h=100m, mobile 100W | 60.5 | 49.9 | 10.57 | 21.2% | FAIL |

- Overall MAPE: 41.2%
- CI Coverage Rate: 46%
- V&V 20 Pass Rate: 19%
- Total Cases: 48

### Per-Domain Validation Metrics

| Domain | n | MAPE | V&V 20 PASS | CI Coverage |
|--------|---|------|-------------|-------------|
| close_range | 5 | 25.8% | 20% | 40% |
| field_meas_estimated | 7 | 34.6% | 0% | 29% |
| field_measurement | 5 | 21.2% | 0% | 80% |
| long_range | 9 | 62.2% | 11% | 22% |
| medium_range | 17 | 27.6% | 41% | 65% |
| regulatory | 5 | 93.8% | 0% | 20% |

---
Report generated by UAV-CPS-Analyzer v1.0.0
Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025