#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: Visualization Module
Generates publication-quality figures from actual model/simulation outputs.

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from typing import Dict, List, Tuple, Optional
import os

# Try to import other modules
try:
    from monte_carlo_engine import MCResult, SimulationParams, MonteCarloEngine, ScenarioSimulator
    from fhss_emulator import JammingStrategy, JammingEffectivenessAnalyzer, OcuSyncProtocol
    from propagation_models import (
        AltitudeDependentModel, AlHouraniA2GModel, FriisModel, COST231HataModel, BERModel
    )
    from cps_analyzer import CUASArchitecture, ThreatType
except ImportError:
    pass

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (10, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


class UAVCPSVisualizer:
    """
    Visualization class for UAV-CPS-Analyzer results.
    All plots use actual model/simulation outputs -- no hardcoded data.
    """

    def __init__(self, output_dir: str = "output"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory for saving figures
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Color scheme
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'tertiary': '#F18F01',
            'quaternary': '#C73E1D',
            'success': '#3A7D44',
            'neutral': '#666666'
        }

    def _save_figure(self, fig: plt.Figure, save_name: str):
        """Save figure in PNG and PDF formats."""
        if save_name:
            fig.savefig(os.path.join(self.output_dir, f"{save_name}.png"))
            fig.savefig(os.path.join(self.output_dir, f"{save_name}.pdf"))

    def plot_monte_carlo_results(self, mc_result: MCResult,
                                  sensitivities: Dict[str, float] = None,
                                  distance_results: Dict[float, MCResult] = None,
                                  power_results: Dict[float, MCResult] = None,
                                  title: str = "Monte Carlo Simulation Results",
                                  save_name: str = None) -> plt.Figure:
        """
        Plot Monte Carlo simulation results (Figure 1 from paper).
        All subplots driven by actual simulation data.

        Args:
            mc_result: Primary MC result (for histogram)
            sensitivities: Dict of parameter -> sensitivity in dB (from engine)
            distance_results: Dict of distance -> MCResult (for distance sweep)
            power_results: Dict of power_W -> MCResult (for power sweep)
            title: Plot title
            save_name: Filename to save (without extension)

        Returns:
            matplotlib Figure object
        """
        js_values = mc_result.js_values
        mean_js = mc_result.mean_js_db
        ci_lower = mc_result.ci_95_lower
        ci_upper = mc_result.ci_95_upper

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # (a) J/S Distribution histogram
        ax1 = axes[0, 0]
        n, bins, patches = ax1.hist(js_values, bins=50, density=True,
                                     alpha=0.7, color=self.colors['primary'],
                                     edgecolor='white', linewidth=0.5)
        ax1.axvline(mean_js, color=self.colors['quaternary'], linestyle='-',
                    linewidth=2, label=f'Mean = {mean_js:.1f} dB')
        ax1.axvline(ci_lower, color=self.colors['tertiary'], linestyle='--',
                    linewidth=1.5, label=f'95% CI = [{ci_lower:.1f}, {ci_upper:.1f}]')
        ax1.axvline(ci_upper, color=self.colors['tertiary'], linestyle='--',
                    linewidth=1.5)
        ax1.fill_betweenx([0, max(n)*1.1], ci_lower, ci_upper,
                          alpha=0.2, color=self.colors['tertiary'])
        # Bootstrap CI annotation
        if mc_result.ci_lower_uncertainty != (0.0, 0.0):
            ci_l_unc = mc_result.ci_lower_uncertainty
            ci_u_unc = mc_result.ci_upper_uncertainty
            ax1.annotate(
                f'CI bounds uncertainty:\n'
                f'  lower: [{ci_l_unc[0]:.1f}, {ci_l_unc[1]:.1f}]\n'
                f'  upper: [{ci_u_unc[0]:.1f}, {ci_u_unc[1]:.1f}]',
                xy=(0.02, 0.72), xycoords='axes fraction', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5)
            )
        ax1.set_xlabel('J/S Ratio (dB)')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('(a) J/S Distribution')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # (b) Success probability vs distance -- from actual MC sweeps
        ax2 = axes[0, 1]
        if distance_results:
            distances = sorted(distance_results.keys())
            success_probs = [distance_results[d].success_probability for d in distances]
            ax2.plot(distances, success_probs, color=self.colors['primary'],
                     linewidth=2, marker='o', markersize=4, markevery=max(1, len(distances)//10))
        else:
            # Run actual simulations for distance sweep
            engine = MonteCarloEngine()
            distances = np.linspace(100, 3000, 15)
            success_probs = []
            base_params = SimulationParams(
                jammer_power_dbm=mc_result.js_values.mean() + 40 if len(mc_result.js_values) > 0 else 40,
                signal_distance_m=5000, altitude_m=100, fhss_enabled=False
            )
            for d in distances:
                p = SimulationParams(**{**vars(base_params), 'jammer_distance_m': d})
                r = engine.run_simulation(p, n_iterations=2000, parallel=False)
                success_probs.append(r.success_probability)
            ax2.plot(distances, success_probs, color=self.colors['primary'],
                     linewidth=2, marker='o', markersize=4, markevery=2)

        ax2.axhline(0.5, color=self.colors['neutral'], linestyle=':', alpha=0.7)
        ax2.set_xlabel('Jammer Distance (m)')
        ax2.set_ylabel('Jamming Success Probability')
        ax2.set_title('(b) Success Probability vs Distance')
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)

        # (c) Tornado diagram -- from actual sensitivity analysis
        ax3 = axes[1, 0]
        if sensitivities:
            sorted_sens = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
            params_names = [s[0].replace('_', ' ').title() for s in sorted_sens]
            sens_values = [s[1] for s in sorted_sens]
        else:
            # Run actual sensitivity analysis
            engine = MonteCarloEngine()
            base_params = SimulationParams(
                jammer_power_dbm=40, jammer_distance_m=500,
                signal_distance_m=5000, altitude_m=100, fhss_enabled=False
            )
            sensitivities = engine.sensitivity_analysis(base_params, n_iterations=1000)
            sorted_sens = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
            params_names = [s[0].replace('_', ' ').title() for s in sorted_sens]
            sens_values = [s[1] for s in sorted_sens]

        y_pos = np.arange(len(params_names))
        colors = [self.colors['quaternary'] if s > 2 else self.colors['primary']
                  for s in sens_values]

        bars = ax3.barh(y_pos, sens_values, color=colors, alpha=0.8,
                        edgecolor='white', linewidth=0.5)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(params_names)
        ax3.set_xlabel('Sensitivity (\u00b1dB per \u03c3)')
        ax3.set_title('(c) Parameter Sensitivity (Tornado Diagram)')
        ax3.axvline(0, color='black', linewidth=0.5)
        ax3.grid(True, alpha=0.3, axis='x')

        for bar, val in zip(bars, sens_values):
            ax3.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                    f'\u00b1{val:.1f}', va='center', fontsize=9)

        # (d) J/S vs Power with 95% CI -- from actual MC sweeps
        ax4 = axes[1, 1]
        if power_results:
            powers = sorted(power_results.keys())
            means = [power_results[p].mean_js_db for p in powers]
            ci_lows = [power_results[p].mean_js_db - power_results[p].ci_95_lower for p in powers]
            ci_highs = [power_results[p].ci_95_upper - power_results[p].mean_js_db for p in powers]
            powers_label = [f'{p}W' for p in powers]
        else:
            # Run actual simulations for power sweep
            engine = MonteCarloEngine()
            powers = [1, 5, 10, 50, 100, 500]
            means, ci_lows, ci_highs = [], [], []
            for pw in powers:
                p_dbm = 10 * np.log10(pw * 1000)
                p = SimulationParams(
                    jammer_power_dbm=p_dbm, jammer_distance_m=500,
                    signal_distance_m=5000, altitude_m=100, fhss_enabled=False
                )
                r = engine.run_simulation(p, n_iterations=2000, parallel=False)
                means.append(r.mean_js_db)
                ci_lows.append(r.mean_js_db - r.ci_95_lower)
                ci_highs.append(r.ci_95_upper - r.mean_js_db)
            powers_label = [f'{p}W' for p in powers]

        ax4.errorbar(range(len(powers)), means,
                     yerr=[ci_lows, ci_highs],
                     fmt='o-', color=self.colors['primary'],
                     capsize=5, capthick=2, markersize=8, linewidth=2)
        ax4.set_xticks(range(len(powers)))
        ax4.set_xticklabels(powers_label)
        ax4.set_xlabel('Jammer Power')
        ax4.set_ylabel('J/S Ratio (dB)')
        ax4.set_title('(d) J/S vs Power with 95% CI')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_fhss_analysis(self, save_name: str = None) -> plt.Figure:
        """
        Plot FHSS effectiveness analysis (Figure 2 from paper).
        Uses actual FHSS emulator outputs.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Run actual FHSS analysis
        protocol = OcuSyncProtocol(version="3.0")
        analyzer = JammingEffectivenessAnalyzer(protocol)
        results = analyzer.compare_strategies()

        strategies = list(results.keys())
        strategy_labels = [s.replace('_', '\n').title() for s in strategies]
        static_eff = [results[s]['effectiveness_static'] * 100 for s in strategies]
        fhss_eff = [results[s]['effectiveness_fhss'] * 100 for s in strategies]
        power_mult = [results[s]['power_multiplier'] for s in strategies]

        x = np.arange(len(strategies))
        width = 0.35

        # (a) Effectiveness comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(x - width/2, static_eff, width, label='Static Channel',
                        color=self.colors['primary'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, fhss_eff, width, label='FHSS (OcuSync)',
                        color=self.colors['secondary'], alpha=0.8)
        ax1.set_xlabel('Jamming Strategy')
        ax1.set_ylabel('Effectiveness (%)')
        ax1.set_title('(a) Effectiveness: Static vs FHSS')
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategy_labels)
        ax1.legend()
        ax1.set_ylim([0, 105])
        ax1.grid(True, alpha=0.3, axis='y')

        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=8)

        # (b) Power multiplier
        ax2 = axes[0, 1]
        colors = [self.colors['success'] if p <= 3 else
                  (self.colors['tertiary'] if p <= 10 else self.colors['quaternary'])
                  for p in power_mult]
        bars = ax2.bar(strategy_labels, power_mult, color=colors, alpha=0.8,
                       edgecolor='white', linewidth=0.5)
        ax2.set_xlabel('Jamming Strategy')
        ax2.set_ylabel('Power Multiplier (\u00d7)')
        ax2.set_title('(b) Required Power Multiplier for FHSS')
        ax2.set_yscale('log')
        ax2.set_ylim([0.5, 100])
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, power_mult):
            ax2.text(bar.get_x() + bar.get_width()/2., val * 1.2,
                    f'\u00d7{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # (c) Protocol resilience comparison (computed from FHSS channel counts)
        ax3 = axes[1, 0]
        channel_counts = [0, 20, 40, 80]
        protocol_labels = ['Static', 'FHSS-20', 'FHSS-40\n(OcuSync)', 'FHSS-80']
        resilience = []
        for n_ch in channel_counts:
            if n_ch == 0:
                resilience.append(0.0)
            else:
                # Resilience = 1 - P(narrowband hits active channel) = 1 - 1/n_ch
                resilience.append((1.0 - 1.0 / n_ch) * 100)

        bars = ax3.barh(protocol_labels, resilience, color=self.colors['primary'],
                        alpha=0.8, edgecolor='white')
        ax3.set_xlabel('Resilience to Narrowband Jamming (%)')
        ax3.set_title('(c) Protocol Jamming Resilience')
        ax3.set_xlim([0, 100])
        ax3.grid(True, alpha=0.3, axis='x')

        for bar, val in zip(bars, resilience):
            ax3.text(val + 2, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}%', va='center', fontsize=9)

        # (d) Required power vs distance for FHSS (from actual model)
        ax4 = axes[1, 1]
        distances = np.linspace(100, 2000, 20)
        model = AltitudeDependentModel()

        for strategy_name, style, label in [('broadband', '-', 'Broadband'),
                                             ('follower', '--', 'Follower'),
                                             ('narrowband', ':', 'Narrowband')]:
            strat = JammingStrategy(strategy_name)
            mult = analyzer.calculate_power_multiplier(strat)
            required_powers = []
            for d in distances:
                # Compute minimum power needed for J/S > 10 dB at distance d
                # J/S = P_j + G_j - PL_j - (P_s + G_s - PL_s) > threshold
                pl_j = model.path_loss(d, 100, 2437)
                pl_s = model.path_loss(5000, 100, 2437)
                # P_j (dBm) > threshold + (P_s + G_s - PL_s) + PL_j - G_j
                p_j_dbm = 10 + (20 + 2 - pl_s) + pl_j - 6
                p_j_w = 10 ** ((p_j_dbm - 30) / 10) * mult
                required_powers.append(max(0.1, p_j_w))
            ax4.plot(distances, required_powers, style, linewidth=2, label=label)

        ax4.set_xlabel('Distance (m)')
        ax4.set_ylabel('Required Power (W)')
        ax4.set_title('(d) Required Power vs Distance (FHSS)')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_altitude_analysis(self, save_name: str = None) -> plt.Figure:
        """
        Plot altitude-dependent propagation analysis (Figure 3 from paper).
        Uses actual propagation models.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        model = AltitudeDependentModel()
        friis = FriisModel()
        cost231 = COST231HataModel("medium")

        # (a) Effective range vs altitude -- from actual model binary search
        ax1 = axes[0]
        altitudes = np.linspace(50, 1000, 20)

        for power, color, label in [(10, self.colors['primary'], '10W'),
                                     (100, self.colors['secondary'], '100W'),
                                     (500, self.colors['tertiary'], '500W')]:
            tx_dbm = 10 * np.log10(power * 1000)
            ranges = []
            for h in altitudes:
                r = model.effective_range(tx_dbm, -90.0, h, 2437.0, margin_db=10.0)
                ranges.append(r)
            ax1.plot(altitudes, ranges, linewidth=2, color=color, label=label)

        ax1.set_xlabel('Altitude (m)')
        ax1.set_ylabel('Effective Range (m)')
        ax1.set_title('(a) Effective Jamming Range vs Altitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axvline(100, color='gray', linestyle=':', alpha=0.5)
        ax1.axvline(500, color='gray', linestyle=':', alpha=0.5)
        ax1.annotate('Urban', xy=(75, ax1.get_ylim()[0] + 100), fontsize=8, alpha=0.7)
        ax1.annotate('Transition', xy=(250, ax1.get_ylim()[0] + 100), fontsize=8, alpha=0.7)
        ax1.annotate('Free Space', xy=(700, ax1.get_ylim()[0] + 100), fontsize=8, alpha=0.7)

        # (b) Path loss vs altitude -- from actual models
        ax2 = axes[1]
        altitudes_fine = np.linspace(10, 800, 50)
        distance = 1000  # meters

        pl_cost231_vals = []
        pl_friis_vals = []
        pl_combined_vals = []

        for h in altitudes_fine:
            distance_3d = np.sqrt(distance**2 + h**2)
            pl_c = cost231.path_loss(distance, 2437.0, h_base=max(30, min(h, 200)))
            pl_f = friis.path_loss(distance_3d, 2437.0)
            pl_comb = model.path_loss(distance, h, 2437.0)

            pl_cost231_vals.append(pl_c)
            pl_friis_vals.append(pl_f)
            pl_combined_vals.append(pl_comb)

        ax2.plot(altitudes_fine, pl_cost231_vals, '--', color=self.colors['neutral'],
                 linewidth=1.5, label='COST 231-Hata', alpha=0.7)
        ax2.plot(altitudes_fine, pl_friis_vals, '--', color=self.colors['secondary'],
                 linewidth=1.5, label='Friis', alpha=0.7)
        ax2.plot(altitudes_fine, pl_combined_vals, '-', color=self.colors['primary'],
                 linewidth=2.5, label='Combined Model')

        ax2.set_xlabel('Altitude (m)')
        ax2.set_ylabel('Path Loss (dB)')
        ax2.set_title(f'(b) Path Loss vs Altitude (d = {distance}m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axvspan(0, 100, alpha=0.1, color='red')
        ax2.axvspan(100, 500, alpha=0.1, color='yellow')
        ax2.axvspan(500, 800, alpha=0.1, color='green')

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_temporal_analysis(self, save_name: str = None) -> plt.Figure:
        """
        Plot temporal characteristics / latency analysis (Figure 4 from paper).
        Uses actual C-UAS architecture latency data.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # (a) Latency breakdown -- from actual CUASArchitecture
        ax1 = axes[0]
        cuas = CUASArchitecture()
        latency_data = cuas.calculate_total_latency()

        stages = ['Detection', 'Classification']
        latencies_min = [latency_data['detection_s'], latency_data['classification_s']]

        # Add neutralization methods
        neut_latencies = latency_data['neutralization_s']
        for method_name, lat_s in sorted(neut_latencies.items(), key=lambda x: x[1]):
            stages.append(f'Neut: {method_name}')
            latencies_min.append(lat_s)

        y_pos = np.arange(len(stages))
        bar_colors = ([self.colors['primary']] * 2 +
                      [self.colors['secondary']] * len(neut_latencies))

        bars = ax1.barh(y_pos, latencies_min, color=bar_colors, alpha=0.8)

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(stages, fontsize=8)
        ax1.set_xlabel('Latency (seconds)')
        ax1.set_title('(a) C-UAS System Latency Breakdown')
        ax1.grid(True, alpha=0.3, axis='x')

        total_min = latency_data['total_min_s']
        total_max = latency_data['total_max_s']
        ax1.axvline(total_min, color=self.colors['tertiary'], linestyle='--', alpha=0.7)
        ax1.text(total_min + 0.1, len(stages) - 0.5,
                 f'Min total: {total_min:.1f}s', fontsize=8, color=self.colors['tertiary'])

        # (b) Intercept probability vs target speed -- from actual engagement simulation
        ax2 = axes[1]
        speeds = np.linspace(10, 80, 12)  # m/s

        for initial_range, style, label in [(500, '-', '500m'),
                                             (1000, '--', '1000m'),
                                             (2000, ':', '2000m')]:
            intercept_probs = []
            n_trials = 200
            for speed in speeds:
                successes = 0
                for _ in range(n_trials):
                    result = cuas.simulate_engagement(
                        ThreatType.RF_CONTROLLED, initial_range, speed
                    )
                    if result['neutralized']:
                        successes += 1
                intercept_probs.append(successes / n_trials)

            ax2.plot(speeds * 3.6, intercept_probs, style, linewidth=2,
                    label=f'Initial range: {label}')

        ax2.set_xlabel('Target Speed (km/h)')
        ax2.set_ylabel('Intercept Probability')
        ax2.set_title('(b) Intercept Probability vs Target Speed')
        ax2.legend()
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(144, color=self.colors['quaternary'], linestyle=':',
                    alpha=0.7)
        ax2.text(146, 0.1, 'FPV', fontsize=8, color=self.colors['quaternary'])

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_ber_analysis(self, save_name: str = None) -> plt.Figure:
        """
        Plot BER/PER analysis showing soft degradation vs hard threshold.
        New figure demonstrating the BER model improvement.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        js_range = np.linspace(-5, 30, 100)

        # (a) BER vs J/S for different modulations
        ax1 = axes[0]
        from propagation_models import ModulationType
        for mod, color, label in [
            (ModulationType.BPSK, self.colors['primary'], 'BPSK'),
            (ModulationType.QPSK, self.colors['secondary'], 'QPSK'),
            (ModulationType.QAM16, self.colors['tertiary'], '16-QAM'),
            (ModulationType.QAM64, self.colors['quaternary'], '64-QAM'),
        ]:
            ber_vals = [BERModel.ber(-js, mod) for js in js_range]  # SINR = -J/S approx
            ax1.semilogy(js_range, ber_vals, linewidth=2, color=color, label=label)

        ax1.axvline(10, color='black', linestyle='--', alpha=0.5, label='Old threshold (10 dB)')
        ax1.set_xlabel('J/S Ratio (dB)')
        ax1.set_ylabel('Bit Error Rate')
        ax1.set_title('(a) BER vs J/S Ratio by Modulation')
        ax1.legend(fontsize=8)
        ax1.set_ylim([1e-6, 0.5])
        ax1.grid(True, alpha=0.3)

        # (b) Jamming success probability (PER) vs J/S
        ax2 = axes[1]
        per_vals = [BERModel.jamming_success_probability(js) for js in js_range]
        hard_threshold = [1.0 if js > 10 else 0.0 for js in js_range]

        ax2.plot(js_range, per_vals, linewidth=2.5, color=self.colors['primary'],
                 label='BER/PER model (soft)')
        ax2.plot(js_range, hard_threshold, '--', linewidth=2, color=self.colors['quaternary'],
                 label='Hard threshold (10 dB)', alpha=0.7)
        ax2.fill_between(js_range, per_vals, hard_threshold, alpha=0.15,
                         color=self.colors['tertiary'])

        ax2.set_xlabel('J/S Ratio (dB)')
        ax2.set_ylabel('Jamming Success Probability')
        ax2.set_title('(b) Soft vs Hard Jamming Threshold')
        ax2.legend()
        ax2.set_ylim([-0.05, 1.05])
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def generate_all_figures(self, mc_result: MCResult = None,
                              sensitivities: Dict[str, float] = None) -> Dict[str, plt.Figure]:
        """
        Generate all figures for the paper.

        Args:
            mc_result: Monte Carlo result (runs simulation if None)
            sensitivities: Sensitivity analysis results (runs analysis if None)

        Returns:
            Dictionary of figure names to Figure objects
        """
        if mc_result is None:
            # Run actual simulation
            engine = MonteCarloEngine()
            params = SimulationParams(
                jammer_power_dbm=40, jammer_distance_m=500,
                signal_distance_m=5000, altitude_m=100, fhss_enabled=False
            )
            mc_result = engine.run_simulation(params, n_iterations=10000)

        figures = {}

        print("Generating figures...")

        print("  - Figure 1: Monte Carlo Results")
        figures['fig_monte_carlo'] = self.plot_monte_carlo_results(
            mc_result, sensitivities=sensitivities, save_name='fig_monte_carlo'
        )

        print("  - Figure 2: FHSS Analysis")
        figures['fig_fhss_analysis'] = self.plot_fhss_analysis(
            save_name='fig_fhss_analysis'
        )

        print("  - Figure 3: Altitude Analysis")
        figures['fig_altitude_analysis'] = self.plot_altitude_analysis(
            save_name='fig_altitude_analysis'
        )

        print("  - Figure 4: Temporal Analysis")
        figures['fig_temporal_analysis'] = self.plot_temporal_analysis(
            save_name='fig_temporal_analysis'
        )

        print("  - Figure 5: BER Analysis")
        figures['fig_ber_analysis'] = self.plot_ber_analysis(
            save_name='fig_ber_analysis'
        )

        print(f"All figures saved to: {self.output_dir}/")

        return figures


if __name__ == "__main__":
    print("=" * 60)
    print("UAV-CPS-Analyzer: Visualization Module Test")
    print("=" * 60)

    # Create visualizer
    viz = UAVCPSVisualizer(output_dir="output")

    # Generate all figures
    figures = viz.generate_all_figures()

    print(f"\nGenerated {len(figures)} figures:")
    for name in figures:
        print(f"  - {name}")

    print("\n" + "=" * 60)
    print("Visualization complete!")

    plt.show()
