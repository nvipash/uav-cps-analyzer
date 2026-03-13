#!/usr/bin/env python3
"""
Full analysis script for scientific audit of UAV-CPS-Analyzer results.
Runs all analyses with progress tracking and elapsed time per step.
"""

import sys, os, time, warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress only expected, benign warnings produced during normal operation.
# A blanket filterwarnings('ignore') was removed so that unexpected numerical
# issues remain visible.

# sklearn GP / MLP fits on limited synthetic data often reach max_iter before
# the tolerance is met; the result is still usable for the surrogate model.
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# NumPy overflow in exp() and invalid (NaN) values occur in the tails of the
# Rice/log-normal distributions during Monte Carlo sampling of extreme
# parameter combinations; the affected samples are clipped downstream.
warnings.filterwarnings('ignore', category=RuntimeWarning,
                        message='overflow encountered in exp')
warnings.filterwarnings('ignore', category=RuntimeWarning,
                        message='invalid value encountered')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ != "__main__":
    sys.exit(0)

from monte_carlo_engine import MonteCarloEngine, SimulationParams, MCResult
from propagation_models import BERModel, DopplerModel, AlHouraniA2GModel, ModulationType
from fhss_emulator import OcuSyncProtocol, JammingEffectivenessAnalyzer, JammingStrategy
from cps_analyzer import SensorFusion, ThreatType, CUASArchitecture, CostEffectivenessAnalyzer
from validation import ValidationEngine, InternalConsistencyChecker
from config import ENVIRONMENT_PRESETS
from ai_surrogate import SurrogateTrainer, SurrogatePredictor, train_and_evaluate
from ai_optimizer import JammerOptimizer, SensorSuiteOptimizer
from ai_propagation_correction import PropagationCorrector
from ai_adaptive_jamming import AdaptiveJammingSimulator, ACTIONS
from ai_threat_classifier import run_comparison as run_threat_comparison, CLASS_NAMES
from ai_uncertainty_active import (
    MultiOutputUncertaintyPropagator, ActiveLearningCorrector
)
from cps_analyzer import CostEffectivenessAnalyzer
from trajectory_scenarios import TrajectorySimulator, print_trajectory_summary
from multi_jammer_coordination import (
    MultiJammerSimulator, JammerNetworkOptimizer, create_default_networks
)
from swarm_scenarios import (
    SwarmAttackSimulator, SwarmConfig, SwarmType, SwarmGenerator, print_swarm_results
)
from ai_coevolution import CoevolutionTrainer
import numpy as np

# ---------- config ----------
N = 10000       # primary MC iterations (meets 9604 theoretical min)
N_SEC = 5000    # secondary sweeps / comparisons
SEED = 42       # reproducible
TOTAL_STEPS = 26  # 10 physics + 5 AI + 3 advanced + 5 +PCE + 3 (multi-jammer/swarm/coevol)
# -----------------------------

t_global = time.time()
engine = MonteCarloEngine(n_processes=1)  # sequential — avoids Windows spawn/DLL crashes

def step(num, label):
    pct = int(num / TOTAL_STEPS * 100)
    bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
    print(f"\n[{bar}] {pct:>3}%  Step {num}/{TOTAL_STEPS}: {label}", flush=True)
    return time.time()

def done(t0):
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)


# ================================================================
t0 = step(1, "Table 3 — Core MC jamming scenarios (QMC variance reduction)")
# ================================================================
scenarios = [
    ("Portable 10W, 500m",   10,  500, 5000,  100, 6),
    ("Portable 10W, 1000m",  10, 1000, 5000,  100, 6),
    ("Mobile 100W, 2000m",  100, 2000, 8000,  200, 10),
    ("Stationary 500W, 3km", 500, 3000, 10000, 300, 15),
]
table3 = {}
print(f"  {'Scenario':<26} {'J/S (dB)':<11} {'Std':<8} {'95% CI':<22} {'P(jam)':<8} {'Conv'}", flush=True)
print("  " + "-" * 73, flush=True)
for name, pw, jd, sd, alt, ga in scenarios:
    p = SimulationParams(
        jammer_power_dbm=10*np.log10(pw*1000), jammer_distance_m=jd,
        jammer_antenna_gain_dbi=ga, signal_distance_m=sd,
        altitude_m=alt, fhss_enabled=False,
        shadow_correlation=0.3,  # spatial correlation typical urban
    )
    # Use Quasi-Monte Carlo (Sobol) for variance reduction at N=10000
    r = engine.run_simulation_qmc(p, N, random_seed=SEED)
    ci = f"[{r.ci_95_lower:.1f}, {r.ci_95_upper:.1f}]"
    conv = "Yes" if r.converged else "No"
    print(f"  {name:<26} {r.mean_js_db:<11.1f} {r.std_js_db:<8.1f} {ci:<22} {r.success_probability*100:<8.0f}% {conv}", flush=True)
    table3[name] = r
done(t0)


# ================================================================
t0 = step(2, "Propagation model comparison — Legacy vs Al-Hourani")
# ================================================================
print(f"  {'Model':<25} {'J/S (dB)':<11} {'Std':<8} {'95% CI':<22}", flush=True)
print("  " + "-" * 63, flush=True)
for model_name, env in [("Altitude-Dependent", None), ("Al-Hourani (urban)", "urban"),
                         ("Al-Hourani (suburban)", "suburban"), ("Al-Hourani (rural)", "rural")]:
    p = SimulationParams(
        jammer_power_dbm=40, jammer_distance_m=500, signal_distance_m=5000,
        altitude_m=100, fhss_enabled=False,
        propagation_model="al_hourani" if env else "altitude",
        environment=env or "urban"
    )
    r = engine.run_simulation(p, N_SEC, parallel=False, random_seed=SEED)
    ci = f"[{r.ci_95_lower:.1f}, {r.ci_95_upper:.1f}]"
    print(f"  {model_name:<25} {r.mean_js_db:<11.1f} {r.std_js_db:<8.1f} {ci:<22}", flush=True)
done(t0)


# ================================================================
t0 = step(3, "Table 4 — FHSS jamming strategy effectiveness")
# ================================================================
protocol = OcuSyncProtocol(version="3.0")
analyzer = JammingEffectivenessAnalyzer(protocol)
results = analyzer.compare_strategies()
print(f"  {'Strategy':<16} {'Static':<10} {'FHSS':<10} {'Power x':<10} {'FHSS Protection'}", flush=True)
print("  " + "-" * 58, flush=True)
for s, d in results.items():
    static_e = d["effectiveness_static"]
    fhss_e = d["effectiveness_fhss"]
    protection = (1 - fhss_e / static_e) * 100 if static_e > 0 else 0
    print(f"  {s:<16} {static_e*100:>6.0f}%   {fhss_e*100:>6.0f}%   x{d['power_multiplier']:<6.0f}   {protection:>5.0f}% reduction", flush=True)
done(t0)


# ================================================================
t0 = step(4, "BER/PER soft degradation curve")
# ================================================================
print(f"  {'J/S (dB)':<12} {'SINR (dB)':<12} {'BER (QPSK)':<14} {'PER (1KB)':<14} {'P(jam)'}", flush=True)
print("  " + "-" * 63, flush=True)
for js in [-15, -10, -5, -3, 0, 3, 5, 10, 15, 20]:
    sig, noise = -70.0, -100.0
    j_lin = 10**((sig + js) / 10)
    n_lin = 10**(noise / 10)
    s_lin = 10**(sig / 10)
    sinr = 10 * np.log10(s_lin / (n_lin + j_lin))
    ber = BERModel.ber(sinr, ModulationType.QPSK)
    per = BERModel.packet_error_rate(ber, 8192)
    prob = BERModel.jamming_success_probability(js)
    print(f"  {js:<12} {sinr:<12.1f} {ber:<14.6f} {per:<14.6f} {prob:.4f}", flush=True)
done(t0)


# ================================================================
t0 = step(5, "Doppler effect analysis")
# ================================================================
print(f"  {'Scenario':<30} {'f_d (Hz)':<12} {'T_c (ms)':<12} {'FHSS Degrad.':<14} {'Impact'}", flush=True)
print("  " + "-" * 73, flush=True)
for label, freq, vel in [("DJI Mavic (2.4G, 21m/s)", 2437, 21),
                          ("DJI FPV (5.8G, 40m/s)", 5800, 40),
                          ("High-speed (2.4G, 80m/s)", 2437, 80),
                          ("Missile (5.8G, 300m/s)", 5800, 300)]:
    fd = DopplerModel.max_doppler_spread_hz(freq, vel)
    tc = DopplerModel.coherence_time_s(freq, vel) * 1000
    deg = DopplerModel.hop_sync_degradation(freq, vel, 2.0)
    impact = "Negligible" if deg < 0.01 else ("Moderate" if deg < 0.1 else "Significant")
    print(f"  {label:<30} {fd:<12.1f} {tc:<12.2f} {deg:<14.4f} {impact}", flush=True)
done(t0)


# ================================================================
t0 = step(6, "Antenna pattern impact")
# ================================================================
print(f"  {'Configuration':<30} {'J/S (dB)':<11} {'Delta vs Omni'}", flush=True)
print("  " + "-" * 53, flush=True)
base_r = engine.run_simulation(SimulationParams(
    jammer_power_dbm=40, jammer_distance_m=500, signal_distance_m=5000,
    altitude_m=100, fhss_enabled=False, jammer_beam_width_deg=0
), N_SEC, parallel=False, random_seed=SEED)
print(f"  {'Omnidirectional':<30} {base_r.mean_js_db:<11.1f} {'---'}", flush=True)
for bw in [60, 30, 10]:
    r = engine.run_simulation(SimulationParams(
        jammer_power_dbm=40, jammer_distance_m=500, signal_distance_m=5000,
        altitude_m=100, fhss_enabled=False, jammer_beam_width_deg=bw
    ), N_SEC, parallel=False, random_seed=SEED)
    delta = r.mean_js_db - base_r.mean_js_db
    label = f"Directional ({bw} deg)"
    print(f"  {label:<30} {r.mean_js_db:<11.1f} {delta:+.1f} dB", flush=True)
done(t0)


# ================================================================
t0 = step(7, "Multi-environment comparison")
# ================================================================
print(f"  {'Environment':<16} {'Shadow std':<12} {'J/S (dB)':<11} {'Std':<8} {'95% CI':<22}", flush=True)
print("  " + "-" * 68, flush=True)
for env, preset in ENVIRONMENT_PRESETS.items():
    p = SimulationParams(
        jammer_power_dbm=40, jammer_distance_m=500, signal_distance_m=5000,
        altitude_m=100, fhss_enabled=False,
        path_loss_std=preset["shadow_fading_std_db"]
    )
    r = engine.run_simulation(p, N_SEC, parallel=False, random_seed=SEED)
    ci = f"[{r.ci_95_lower:.1f}, {r.ci_95_upper:.1f}]"
    print(f"  {env:<16} {preset['shadow_fading_std_db']:<12.1f} {r.mean_js_db:<11.1f} {r.std_js_db:<8.1f} {ci:<22}", flush=True)
done(t0)


# ================================================================
t0 = step(8, "Sensor fusion & detection rates")
# ================================================================
fusion = SensorFusion()
for tt in [ThreatType.RF_CONTROLLED, ThreatType.FIBER_OPTIC, ThreatType.AUTONOMOUS]:
    rate = fusion.get_detection_rate(tt, 1000, n_trials=500)
    print(f"  {tt.value:<20}: {rate*100:.1f}%", flush=True)
done(t0)


# ================================================================
t0 = step(9, "Formal validation (ASME V&V 20) + consistency checks")
# ================================================================
checker = InternalConsistencyChecker()
checker.check_all()
passed = sum(1 for _, p, _ in checker.results if p)
total_checks = len(checker.results)
print(f"  Internal consistency: {passed}/{total_checks} passed", flush=True)

val = ValidationEngine(n_iterations=2000)
report = val.run_validation()
print(f"  Total cases: {report.n_cases}", flush=True)
print(f"  Overall MAPE:        {report.overall_mape:.1f}%", flush=True)
print(f"  V&V 20 PASS rate:    {report.overall_vv20_pass_rate*100:.0f}%", flush=True)
print(f"  CI coverage rate:    {report.overall_coverage_rate*100:.0f}%", flush=True)

if report.mape_by_domain:
    print(f"\n  Per-domain breakdown:", flush=True)
    print(f"    {'Domain':<16} {'n':>4}  {'MAPE':>7}  {'PASS':>7}  {'Coverage':>9}", flush=True)
    print(f"    {'-'*48}", flush=True)
    for d in sorted(report.n_by_domain.keys()):
        print(f"    {d:<16} {report.n_by_domain[d]:>4}  "
              f"{report.mape_by_domain[d]:>6.1f}%  "
              f"{report.pass_rate_by_domain[d]*100:>6.0f}%  "
              f"{report.coverage_by_domain[d]*100:>8.0f}%", flush=True)
done(t0)


# ================================================================
t0 = step(10, "OAT sensitivity + cost-effectiveness Pareto")
# ================================================================
print("  Sensitivity analysis...", flush=True)
eng2 = MonteCarloEngine(n_processes=1)
sens = eng2.sensitivity_analysis(SimulationParams(
    jammer_power_dbm=40, jammer_distance_m=500, signal_distance_m=5000,
    altitude_m=100, fhss_enabled=False
), n_iterations=500)
sorted_sens = sorted(sens.items(), key=lambda x: x[1], reverse=True)
for param, val in sorted_sens:
    bar = "#" * int(val * 5)
    print(f"    {param:<20} +/-{val:>5.1f} dB  {bar}", flush=True)

print("  Cost-effectiveness Pareto...", flush=True)
cea = CostEffectivenessAnalyzer(n_trials=200)
configs = cea.enumerate_configurations(range_m=1000)
pareto = cea.pareto_front(configs)
pareto_names = {c["name"] for c in pareto}
print(f"  {len(configs)} configurations, {len(pareto)} Pareto-optimal:", flush=True)
for c in pareto:
    print(f"    {c['name']:<35} ${c['cost_usd']:>9,}  RF:{c['detection_rf']*100:.0f}%  Fiber:{c['detection_fiber']*100:.0f}%", flush=True)
done(t0)


# ================================================================
t0 = step(11, "AI Surrogate Model — train NN to replace MC")
# ================================================================
trainer = SurrogateTrainer(mc_iterations_per_point=500)
X_surr, Y_surr = trainer.generate_training_data(n_points=200, seed=SEED)
model_surr, scaler_X, scaler_Y, surr_metrics = trainer.train(X_surr, Y_surr)
predictor = SurrogatePredictor(model_surr, scaler_X, scaler_Y)

# Speedup test
import time as _t
mc_t0 = _t.time()
engine.run_simulation(SimulationParams(jammer_power_dbm=40, jammer_distance_m=500,
    signal_distance_m=5000, altitude_m=100, fhss_enabled=False), 5000, parallel=False, random_seed=99)
mc_time = _t.time() - mc_t0
sr_t0 = _t.time()
for _ in range(100):
    predictor.predict(40, 500, 6, 5000, 100, 3)
sr_time = (_t.time() - sr_t0) / 100

print(f"  R2 (mean J/S):  {surr_metrics['r2_mean_js']:.4f}", flush=True)
print(f"  RMSE:           {surr_metrics['rmse_mean_js']:.2f} dB", flush=True)
print(f"  MAE:            {surr_metrics['mae_mean_js']:.2f} dB", flush=True)
print(f"  MC time:        {mc_time:.3f}s per run", flush=True)
print(f"  Surrogate time: {sr_time*1000:.3f}ms per prediction", flush=True)
print(f"  Speedup:        {mc_time/sr_time:.0f}x", flush=True)
done(t0)


# ================================================================
t0 = step(12, "AI Bayesian Optimization — jammer & sensor placement")
# ================================================================
jopt = JammerOptimizer(predictor)
for budget_w, label in [(10, "10W portable"), (100, "100W mobile"), (500, "500W stationary")]:
    result = jopt.optimize_power_distance(budget_watts=budget_w)
    p = result.optimal_params
    print(f"  {label}: J/S={p['predicted_js_db']:.1f} dB @ {p['distance_m']:.0f}m, "
          f"{p['power_watts']:.1f}W, {p['antenna_gain_dbi']:.1f} dBi "
          f"({result.n_evaluations} evals)", flush=True)

print("  Sensor suite optimization:", flush=True)
sopt = SensorSuiteOptimizer()
for budget_k in [20, 50, 100]:
    r = sopt.optimize_for_budget(budget_usd=budget_k * 1000, n_trials=200)
    print(f"    ${budget_k}K: {'+'.join(r['sensors'])} -> RF:{r['detection_rf']*100:.0f}%, "
          f"Fiber:{r['detection_fiber']*100:.0f}%, utilization:{r['budget_utilization']:.0f}%",
          flush=True)
done(t0)


# ================================================================
t0 = step(13, "AI Propagation Correction — ML reduces model MAPE")
# ================================================================
corrector = PropagationCorrector()
corr_metrics = corrector.train()
print(f"  Original MAPE:  {corr_metrics['original_mape']:.1f}%", flush=True)
print(f"  Corrected MAPE: {corr_metrics['corrected_mape']:.1f}%", flush=True)
print(f"  Improvement:    {corr_metrics['mape_improvement']:.1f} percentage points", flush=True)
print(f"  CV R2:          {corr_metrics['cv_r2_mean']:.3f} +/- {corr_metrics['cv_r2_std']:.3f}",
      flush=True)
print(f"  Mean correction: {corr_metrics['mean_correction_db']:.1f} dB", flush=True)
print(f"  Feature importance:", flush=True)
for feat, imp in sorted(corr_metrics['feature_importance'].items(), key=lambda x: -x[1]):
    bar = "#" * int(imp * 50)
    print(f"    {feat:<12} {imp:.3f}  {bar}", flush=True)
done(t0)


# ================================================================
t0 = step(14, "AI Adaptive Jamming — RL agent vs fixed strategies")
# ================================================================
sim = AdaptiveJammingSimulator(js_ratio_db=30.0)
rl_results = sim.run_comparison(n_train_episodes=200)
print(f"  Fixed strategies:", flush=True)
for name, rate in rl_results['fixed_results'].items():
    print(f"    {name:<16}: {rate*100:.1f}%", flush=True)
print(f"  RL adaptive:       {rl_results['rl_jam_rate']*100:.1f}%", flush=True)
print(f"  Best fixed:        {rl_results['best_fixed']} ({rl_results['best_fixed_rate']*100:.1f}%)",
      flush=True)
print(f"  RL improvement:    {rl_results['rl_improvement']:+.1f}% over best fixed", flush=True)
done(t0)


# ================================================================
t0 = step(15, "AI Threat Classifier — NN vs D-S vs Hybrid")
# ================================================================
cls_results = run_threat_comparison(n_samples=300)
print(f"  Pure Dempster-Shafer:  {cls_results['ds_accuracy']*100:.1f}%", flush=True)
print(f"  Pure Neural Network:   {cls_results['nn_accuracy']*100:.1f}% "
      f"(F1={cls_results['nn_f1']:.3f}, conf={cls_results['nn_confidence']:.2f})", flush=True)
print(f"  Hybrid NN + D-S:       {cls_results['hybrid_accuracy']*100:.1f}%", flush=True)
print(f"  NN F1 per class:", flush=True)
for cls, f1 in cls_results['nn_f1_per_class'].items():
    print(f"    {cls:<20}: {f1:.3f}", flush=True)
print(f"  Confusion matrix (NN):", flush=True)
for i, row in enumerate(cls_results['nn_confusion']):
    print(f"    {CLASS_NAMES[i]:<15} {row}", flush=True)
done(t0)


# ================================================================
t0 = step(16, "Long-range correction impact (atmospheric + multi-path)")
# ================================================================
print("  Effect of atmospheric absorption + multi-path on long-range J/S:", flush=True)
print(f"  {'Distance':<12} {'No corr.':<12} {'With corr.':<14} {'Delta':<8}", flush=True)
print("  " + "-" * 46, flush=True)
for jd_km in [1, 3, 5, 10, 15]:
    p_none = SimulationParams(
        jammer_power_dbm=50, jammer_distance_m=jd_km*1000, signal_distance_m=10000,
        altitude_m=200, fhss_enabled=False, propagation_model='al_hourani',
        environment='urban', enable_atmospheric=False, enable_multipath=False
    )
    p_corr = SimulationParams(
        jammer_power_dbm=50, jammer_distance_m=jd_km*1000, signal_distance_m=10000,
        altitude_m=200, fhss_enabled=False, propagation_model='al_hourani',
        environment='urban', enable_atmospheric=True, enable_multipath=True
    )
    r_none = engine.run_simulation(p_none, 2000, parallel=False, random_seed=SEED)
    r_corr = engine.run_simulation(p_corr, 2000, parallel=False, random_seed=SEED)
    delta = r_corr.mean_js_db - r_none.mean_js_db
    print(f"  {jd_km}km{'':<8} {r_none.mean_js_db:<12.1f} {r_corr.mean_js_db:<14.1f} {delta:+.1f} dB", flush=True)
done(t0)


# ================================================================
t0 = step(17, "Adversarial RL + Multi-objective reward")
# ================================================================
from ai_adaptive_jamming import AdaptiveJammingSimulator

print("  Standard RL vs Adversarial environment vs Multi-objective:", flush=True)

# Standard
sim1 = AdaptiveJammingSimulator(js_ratio_db=30.0, adversarial=False, multi_objective=False)
r1 = sim1.run_comparison(n_train_episodes=150)

# Adversarial
sim2 = AdaptiveJammingSimulator(js_ratio_db=30.0, adversarial=True, multi_objective=False)
r2 = sim2.run_comparison(n_train_episodes=150)

# Multi-objective
sim3 = AdaptiveJammingSimulator(js_ratio_db=30.0, adversarial=True, multi_objective=True)
r3 = sim3.run_comparison(n_train_episodes=150)

print(f"    {'Mode':<28} {'RL rate':<10} {'Best fixed':<14} {'Improvement'}", flush=True)
print(f"    {'-'*65}", flush=True)
print(f"    {'Standard (no adversary)':<28} {r1['rl_jam_rate']*100:<10.1f}% "
      f"{r1['best_fixed_rate']*100:<14.1f}% {r1['rl_improvement']:+.1f}%", flush=True)
print(f"    {'Adversarial environment':<28} {r2['rl_jam_rate']*100:<10.1f}% "
      f"{r2['best_fixed_rate']*100:<14.1f}% {r2['rl_improvement']:+.1f}%", flush=True)
print(f"    {'Adversarial + multi-obj':<28} {r3['rl_jam_rate']*100:<10.1f}% "
      f"{r3['best_fixed_rate']*100:<14.1f}% {r3['rl_improvement']:+.1f}%", flush=True)
done(t0)


# ================================================================
t0 = step(18, "Multi-output uncertainty propagation + Pareto constraints")
# ================================================================
print("  Multi-output uncertainty (J/S, P_jam, std, CI width):", flush=True)
prop = MultiOutputUncertaintyPropagator(n_samples=64)
base = SimulationParams(
    jammer_power_dbm=40, jammer_distance_m=500, signal_distance_m=5000,
    altitude_m=100, fhss_enabled=False
)
mo = prop.propagate(base)
print(f"    {'Output':<15} {'mean':<10} {'std':<10} {'p5':<10} {'p95':<10}", flush=True)
print(f"    {'-'*55}", flush=True)
for name in mo.output_names:
    print(f"    {name:<15} {mo.means[name]:<10.2f} {mo.stds[name]:<10.2f} "
          f"{mo.p5[name]:<10.2f} {mo.p95[name]:<10.2f}", flush=True)

print("\n  Cross-correlations:", flush=True)
for (k1, k2), corr in list(mo.correlations.items())[:3]:
    print(f"    {k1} <-> {k2}: r={corr:+.3f}", flush=True)

print("\n  Multi-constraint Pareto (max_cost=$60K, min_RF=85%, min_fiber=50%):", flush=True)
cea = CostEffectivenessAnalyzer(n_trials=200)
configs = cea.enumerate_configurations(range_m=1000)
constrained = cea.multi_constraint_pareto(
    configs, max_cost=60000, min_rf_detection=0.85, min_fiber_detection=0.50
)
print(f"    Feasible Pareto-optimal configs: {len(constrained)}", flush=True)
for c in constrained:
    print(f"      {c['name']:<35} ${c['cost_usd']:>9,}  RF:{c['detection_rf']*100:.0f}%  "
          f"Fiber:{c['detection_fiber']*100:.0f}%", flush=True)
done(t0)


# ================================================================
t0 = step(19, "Variance reduction comparison (Standard MC vs QMC vs Antithetic)")
# ================================================================
print(f"  Comparing CI widths at N={N} for same scenario:", flush=True)
test_p = SimulationParams(
    jammer_power_dbm=40, jammer_distance_m=500, signal_distance_m=5000,
    altitude_m=100, fhss_enabled=False
)
methods = [
    ("Standard MC", lambda: engine.run_simulation(test_p, N, parallel=False, random_seed=SEED)),
    ("QMC (Sobol) ", lambda: engine.run_simulation_qmc(test_p, N, random_seed=SEED)),
    ("Antithetic  ", lambda: engine.run_simulation_antithetic(test_p, N, random_seed=SEED)),
]
print(f"  {'Method':<14} {'Mean':<10} {'Std':<8} {'CI width':<10} {'Boot CI lo':<14} {'Boot CI hi'}", flush=True)
print("  " + "-" * 70, flush=True)
for name, fn in methods:
    r = fn()
    ci_w = r.ci_95_upper - r.ci_95_lower
    bl = r.ci_lower_uncertainty
    bh = r.ci_upper_uncertainty
    print(f"  {name:<14} {r.mean_js_db:<10.3f} {r.std_js_db:<8.3f} {ci_w:<10.3f} "
          f"[{bl[0]:.2f},{bl[1]:.2f}]  [{bh[0]:.2f},{bh[1]:.2f}]", flush=True)
done(t0)


# ================================================================
t0 = step(20, "Spatial correlation impact on shadow fading")
# ================================================================
print("  Effect of shadow fading correlation between signal and jammer paths:", flush=True)
print(f"  {'Correlation':<14} {'Mean J/S':<11} {'Std':<8} {'CI width':<11}", flush=True)
print("  " + "-" * 47, flush=True)
for rho in [0.0, 0.3, 0.6, 0.9]:
    p = SimulationParams(
        jammer_power_dbm=40, jammer_distance_m=500, signal_distance_m=5000,
        altitude_m=100, fhss_enabled=False, shadow_correlation=rho
    )
    r = engine.run_simulation(p, N, parallel=False, random_seed=SEED)
    ci_w = r.ci_95_upper - r.ci_95_lower
    print(f"  rho={rho:<10} {r.mean_js_db:<11.2f} {r.std_js_db:<8.2f} {ci_w:<11.2f}", flush=True)
print("\n  Note: Higher correlation (paths share obstacles) reduces CI width — "
      "common shadow effect cancels in J/S = J - S.", flush=True)
done(t0)


# ================================================================
t0 = step(21, "PCE-based tail estimation (Polynomial Chaos Expansion)")
# ================================================================
print("  Hermite-polynomial PCE for tail risk estimation (rare events):", flush=True)
prop = MultiOutputUncertaintyPropagator(n_samples=64)
base = SimulationParams(
    jammer_power_dbm=40, jammer_distance_m=500, signal_distance_m=5000,
    altitude_m=100, fhss_enabled=False
)
mo = prop.propagate(base)
print(f"  {'Output':<13} {'mean':<10} {'std':<8} {'p1':<10} {'p5':<10} {'p95':<10} {'p99'}", flush=True)
print("  " + "-" * 68, flush=True)
for name in mo.output_names:
    print(f"  {name:<13} {mo.means[name]:<10.2f} {mo.stds[name]:<8.2f} "
          f"{mo.p1[name]:<10.2f} {mo.p5[name]:<10.2f} "
          f"{mo.p95[name]:<10.2f} {mo.p99[name]:.2f}", flush=True)
print(f"\n  PCE expansion: order=3, basis size={len(mo.pce_coefficients[mo.output_names[0]])}",
      flush=True)
print(f"  PCE-derived variance for J/S: {mo.pce_var['mean_js_db']:.3f} (sample var: "
      f"{mo.stds['mean_js_db']**2:.3f})", flush=True)
done(t0)


# ================================================================
t0 = step(22, "Active learning for ML propagation correction")
# ================================================================
print("  Active learning loop: iteratively add training points where GP uncertainty is high",
      flush=True)

# Build initial training data from existing PropagationCorrector reference set
from ai_propagation_correction import PropagationCorrector
pc = PropagationCorrector()
features, model_preds, ref_values = pc._generate_reference_data()
initial_data = (features, model_preds, ref_values)

al = ActiveLearningCorrector(initial_n_points=len(features), max_iterations=8)
al_result = al.run(initial_data)
print(f"  Initial training points: {al_result.n_initial_points}", flush=True)
print(f"  Final training points:   {al_result.n_final_points}", flush=True)
print(f"  Active learning iterations: {al_result.n_iterations}", flush=True)
print(f"  CV R^2 (initial):  {al_result.cv_r2_initial:+.3f}", flush=True)
print(f"  CV R^2 (final):    {al_result.cv_r2_final:+.3f}", flush=True)
delta_r2 = al_result.cv_r2_final - al_result.cv_r2_initial
print(f"  Improvement:        {delta_r2:+.3f}", flush=True)
done(t0)


# ================================================================
t0 = step(23, "Time-varying trajectory scenarios (dynamic J/S evolution)")
# ================================================================
print("  Simulating UAV trajectories with time-varying J/S:", flush=True)
traj_sim = TrajectorySimulator(n_iterations_per_point=300)
traj_results = traj_sim.run_standard_scenarios()
print_trajectory_summary(traj_results)
done(t0)


# ================================================================
t0 = step(24, "Multi-jammer coordination — coverage analysis of 4 networks")
# ================================================================
print("  Coverage analysis for predefined jammer networks:", flush=True)
mj_sim = MultiJammerSimulator(n_iterations=200)
networks = create_default_networks()
print(f"  {'Network':<18} {'#J':<4} {'Cost ($)':<12} {'Coverage':<11} {'Deadzone'}", flush=True)
print("  " + "-" * 60, flush=True)
for name, net in networks.items():
    cr = mj_sim.coverage_map(net, grid_size=18)
    print(f"  {name:<18} {cr.n_jammers:<4} ${net.total_cost_usd:<11,.0f} "
          f"{cr.coverage_pct:<10.1f}% {cr.deadzone_pct:.1f}%", flush=True)
done(t0)


# ================================================================
t0 = step(25, "UAV swarm attack scenarios — saturation analysis")
# ================================================================
print("  Swarm vs jammer network engagement:", flush=True)
swarm_sim = SwarmAttackSimulator(jammer_simulator=mj_sim)

# Compare swarm types vs 5_mesh
print(f"\n  Swarm type comparison (n=20 UAVs vs 5_mesh network):", flush=True)
type_results = []
for stype in SwarmType:
    config = SwarmConfig(
        swarm_type=stype, n_uavs=20,
        decoy_fraction=0.5 if stype == SwarmType.DECOY_STRIKE else 0.0
    )
    swarm = SwarmGenerator.generate(config, seed=SEED)
    r = swarm_sim.simulate_attack(swarm, networks['5_mesh'],
                                     duration_s=60.0, time_steps=12,
                                     swarm_type=stype)
    type_results.append(r)
print_swarm_results(type_results)

# Saturation curve (cooperative vs 3_triangle)
print(f"\n  Saturation curve (cooperative vs 3_triangle):", flush=True)
sat_results = swarm_sim.saturation_curve(
    networks['3_triangle'], swarm_sizes=[5, 10, 20, 30, 50],
    swarm_type=SwarmType.COOPERATIVE
)
print_swarm_results(sat_results)
done(t0)


# ================================================================
t0 = step(26, "AI-vs-AI adversarial co-evolution (Nash equilibrium)")
# ================================================================
print("  Multi-agent RL: jammer + FHSS defender learning simultaneously", flush=True)
trainer = CoevolutionTrainer(js_ratio_db=30.0)
result = trainer.train(n_episodes=120, steps_per_episode=40)

print(f"\n  Final jam rate:        {result.final_jam_rate*100:.1f}%", flush=True)
print(f"  Final survival rate:   {result.final_survival_rate*100:.1f}%", flush=True)
print(f"  Nash distance (std):   {result.nash_distance:.4f}", flush=True)
print(f"  Strategy oscillations: {result.cycles_detected}/100", flush=True)

print(f"\n  Jammer equilibrium strategy mix:", flush=True)
for s, p in sorted(result.jammer_strategy_mix.items(), key=lambda x: -x[1]):
    bar = "#" * int(p * 30)
    print(f"    {s:<16} {p*100:>5.1f}%  {bar}", flush=True)

print(f"\n  FHSS defender equilibrium strategy mix:", flush=True)
for s, p in sorted(result.fhss_strategy_mix.items(), key=lambda x: -x[1]):
    bar = "#" * int(p * 30)
    print(f"    {s:<22} {p*100:>5.1f}%  {bar}", flush=True)

# Convergence: compare early vs late episodes
early_jam = np.mean(result.jammer_history[:10])
late_jam = np.mean(result.jammer_history[-10:])
print(f"\n  Convergence: jam_rate {early_jam*100:.1f}% (early) -> "
      f"{late_jam*100:.1f}% (late)", flush=True)
done(t0)


# ================================================================
elapsed = time.time() - t_global
print(f"\n[####################] 100%  ALL COMPLETE in {elapsed:.1f}s", flush=True)
print("=" * 75, flush=True)
