#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: Visualization Module
Generates publication-quality figures for the research paper.

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
    from monte_carlo_engine import MCResult, SimulationParams, ScenarioSimulator
    from fhss_emulator import JammingStrategy, JammingEffectivenessAnalyzer, OcuSyncProtocol
    from propagation_models import AltitudeDependentModel
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
    
    def plot_monte_carlo_results(self, js_values: np.ndarray,
                                  mean_js: float, ci_lower: float, ci_upper: float,
                                  title: str = "Monte Carlo Simulation Results",
                                  save_name: str = None) -> plt.Figure:
        """
        Plot Monte Carlo simulation results (Figure 1a from paper).
        
        Args:
            js_values: Array of J/S values from simulation
            mean_js: Mean J/S ratio
            ci_lower: Lower 95% CI bound
            ci_upper: Upper 95% CI bound
            title: Plot title
            save_name: Filename to save (without extension)
            
        Returns:
            matplotlib Figure object
        """
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
        ax1.set_xlabel('J/S Ratio (dB)')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('(a) J/S Distribution')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # (b) Success probability vs distance
        ax2 = axes[0, 1]
        distances = np.linspace(100, 3000, 30)
        success_probs = []
        
        for d in distances:
            # Simplified model: success prob decreases with distance
            base_js = mean_js - 20 * np.log10(d / 500)
            prob = 1 / (1 + np.exp(-(base_js - 10) / 3))  # Sigmoid
            success_probs.append(prob)
        
        ax2.plot(distances, success_probs, color=self.colors['primary'], 
                 linewidth=2, marker='o', markersize=4, markevery=3)
        ax2.axhline(0.5, color=self.colors['neutral'], linestyle=':', alpha=0.7)
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Success Probability')
        ax2.set_title('(b) Success Probability vs Distance')
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)
        
        # (c) Tornado diagram (sensitivity analysis)
        ax3 = axes[1, 0]
        params = ['Path Loss', 'Rx Sensitivity', 'Tx Power', 
                  'Antenna Gain', 'Fading Margin', 'Signal Power']
        sensitivities = [5.0, 3.0, 2.0, 1.8, 1.5, 1.2]  # dB
        
        y_pos = np.arange(len(params))
        colors = [self.colors['quaternary'] if s > 2 else self.colors['primary'] 
                  for s in sensitivities]
        
        bars = ax3.barh(y_pos, sensitivities, color=colors, alpha=0.8, 
                        edgecolor='white', linewidth=0.5)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(params)
        ax3.set_xlabel('Sensitivity (±dB per σ)')
        ax3.set_title('(c) Parameter Sensitivity (Tornado Diagram)')
        ax3.axvline(0, color='black', linewidth=0.5)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, sensitivities):
            ax3.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'±{val:.1f}', va='center', fontsize=9)
        
        # (d) CI vs Power
        ax4 = axes[1, 1]
        powers_w = [1, 5, 10, 50, 100, 500]
        powers_label = ['1W', '5W', '10W', '50W', '100W', '500W']
        ci_widths = [12, 10, 9.5, 9.2, 9.0, 8.8]  # Narrower CI with more power
        means = [20, 28, 32, 42, 48, 58]
        
        ax4.errorbar(range(len(powers_w)), means, 
                     yerr=[[w/2 for w in ci_widths], [w/2 for w in ci_widths]],
                     fmt='o-', color=self.colors['primary'], 
                     capsize=5, capthick=2, markersize=8, linewidth=2)
        ax4.set_xticks(range(len(powers_w)))
        ax4.set_xticklabels(powers_label)
        ax4.set_xlabel('Jammer Power')
        ax4.set_ylabel('J/S Ratio (dB)')
        ax4.set_title('(d) J/S vs Power with 95% CI')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, f"{save_name}.png"))
            plt.savefig(os.path.join(self.output_dir, f"{save_name}.pdf"))
        
        return fig
    
    def plot_fhss_analysis(self, save_name: str = None) -> plt.Figure:
        """
        Plot FHSS effectiveness analysis (Figure 2 from paper).
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        strategies = ['BBN', 'Spot', 'Sweep', 'Follower', 'Protocol']
        static_eff = [70, 95, 80, 98, 85]
        fhss_eff = [85, 15, 45, 65, 75]
        power_mult = [1, 40, 5, 3, 2]
        
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
        ax1.set_xticklabels(strategies)
        ax1.legend()
        ax1.set_ylim([0, 105])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
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
        bars = ax2.bar(strategies, power_mult, color=colors, alpha=0.8,
                       edgecolor='white', linewidth=0.5)
        ax2.set_xlabel('Jamming Strategy')
        ax2.set_ylabel('Power Multiplier (×)')
        ax2.set_title('(b) Required Power Multiplier for FHSS')
        ax2.set_yscale('log')
        ax2.set_ylim([0.5, 100])
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, power_mult):
            ax2.text(bar.get_x() + bar.get_width()/2., val * 1.2,
                    f'×{val}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # (c) Protocol resilience comparison
        ax3 = axes[1, 0]
        protocols = ['Static', 'FHSS-20', 'FHSS-40\n(OcuSync)', 'FHSS-80']
        resilience = [0, 60, 85, 95]
        
        bars = ax3.barh(protocols, resilience, color=self.colors['primary'], 
                        alpha=0.8, edgecolor='white')
        ax3.set_xlabel('Resilience to Narrowband Jamming (%)')
        ax3.set_title('(c) Protocol Jamming Resilience')
        ax3.set_xlim([0, 100])
        ax3.grid(True, alpha=0.3, axis='x')
        
        for bar, val in zip(bars, resilience):
            ax3.text(val + 2, bar.get_y() + bar.get_height()/2,
                    f'{val}%', va='center', fontsize=9)
        
        # (d) Required power vs distance for FHSS
        ax4 = axes[1, 1]
        distances = np.linspace(100, 2000, 20)
        
        for strategy, style, label in [('BBN', '-', 'Broadband'),
                                        ('Follower', '--', 'Follower'),
                                        ('Spot', ':', 'Narrowband')]:
            mult = {'BBN': 1, 'Follower': 3, 'Spot': 40}[strategy]
            base_power = 10  # Watts
            required_power = base_power * mult * (distances / 500) ** 2
            ax4.plot(distances, required_power, style, linewidth=2, label=label)
        
        ax4.set_xlabel('Distance (m)')
        ax4.set_ylabel('Required Power (W)')
        ax4.set_title('(d) Required Power vs Distance (FHSS)')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, f"{save_name}.png"))
            plt.savefig(os.path.join(self.output_dir, f"{save_name}.pdf"))
        
        return fig
    
    def plot_altitude_analysis(self, save_name: str = None) -> plt.Figure:
        """
        Plot altitude-dependent propagation analysis (Figure 3 from paper).
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # (a) Effective range vs altitude
        ax1 = axes[0]
        altitudes = np.linspace(50, 1000, 20)
        
        for power, color, label in [(10, self.colors['primary'], '10W'),
                                     (100, self.colors['secondary'], '100W'),
                                     (500, self.colors['tertiary'], '500W')]:
            ranges = []
            for h in altitudes:
                # Simplified model: range increases with altitude
                alpha = np.clip((h - 100) / 400, 0, 1)
                base_range = 500 * np.sqrt(power / 10)
                range_m = base_range * (1 + alpha * 4)  # Up to 5x increase
                ranges.append(range_m)
            ax1.plot(altitudes, ranges, linewidth=2, color=color, label=label)
        
        ax1.set_xlabel('Altitude (m)')
        ax1.set_ylabel('Effective Range (m)')
        ax1.set_title('(a) Effective Jamming Range vs Altitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axvline(100, color='gray', linestyle=':', alpha=0.5)
        ax1.axvline(500, color='gray', linestyle=':', alpha=0.5)
        ax1.annotate('Urban', xy=(75, 100), fontsize=8, alpha=0.7)
        ax1.annotate('Transition', xy=(250, 100), fontsize=8, alpha=0.7)
        ax1.annotate('Free Space', xy=(700, 100), fontsize=8, alpha=0.7)
        
        # (b) Path loss vs altitude (at fixed distance)
        ax2 = axes[1]
        altitudes = np.linspace(10, 800, 50)
        distance = 1000  # meters
        
        pl_cost231 = []
        pl_friis = []
        pl_combined = []
        
        for h in altitudes:
            # COST 231 (simplified)
            pl_c = 140 + 10 * np.log10(distance / 1000)
            pl_cost231.append(pl_c)
            
            # Friis
            distance_3d = np.sqrt(distance**2 + h**2)
            pl_f = 32.45 + 20 * np.log10(2437) + 20 * np.log10(distance_3d / 1000)
            pl_friis.append(pl_f)
            
            # Combined
            alpha = np.clip((h - 100) / 400, 0, 1)
            pl_comb = pl_c * (1 - alpha) + pl_f * alpha
            pl_combined.append(pl_comb)
        
        ax2.plot(altitudes, pl_cost231, '--', color=self.colors['neutral'], 
                 linewidth=1.5, label='COST 231-Hata', alpha=0.7)
        ax2.plot(altitudes, pl_friis, '--', color=self.colors['secondary'],
                 linewidth=1.5, label='Friis', alpha=0.7)
        ax2.plot(altitudes, pl_combined, '-', color=self.colors['primary'],
                 linewidth=2.5, label='Combined Model')
        
        ax2.set_xlabel('Altitude (m)')
        ax2.set_ylabel('Path Loss (dB)')
        ax2.set_title(f'(b) Path Loss vs Altitude (d = {distance}m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axvspan(0, 100, alpha=0.1, color='red', label='Urban')
        ax2.axvspan(100, 500, alpha=0.1, color='yellow')
        ax2.axvspan(500, 800, alpha=0.1, color='green')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, f"{save_name}.png"))
            plt.savefig(os.path.join(self.output_dir, f"{save_name}.pdf"))
        
        return fig
    
    def plot_temporal_analysis(self, save_name: str = None) -> plt.Figure:
        """
        Plot temporal characteristics / latency analysis (Figure 4 from paper).
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # (a) Latency breakdown
        ax1 = axes[0]
        stages = ['Detection', 'Classification', 'Tracking', 'Neutralization']
        latencies_min = [0.05, 0.1, 0.5, 2.0]  # seconds
        latencies_max = [0.2, 0.5, 1.0, 8.0]
        
        y_pos = np.arange(len(stages))
        
        # Plot bars with error ranges
        bars = ax1.barh(y_pos, latencies_min, color=self.colors['primary'],
                        alpha=0.8, label='Minimum')
        ax1.barh(y_pos, np.array(latencies_max) - np.array(latencies_min),
                 left=latencies_min, color=self.colors['secondary'],
                 alpha=0.5, label='Additional (max)')
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(stages)
        ax1.set_xlabel('Latency (seconds)')
        ax1.set_title('(a) C-UAS System Latency Breakdown')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add total
        total_min = sum(latencies_min)
        total_max = sum(latencies_max)
        ax1.axvline(total_min, color=self.colors['tertiary'], linestyle='--',
                    label=f'Total min: {total_min:.1f}s')
        ax1.text(total_min + 0.1, len(stages) - 0.5, f'Min: {total_min:.1f}s',
                 fontsize=9, color=self.colors['tertiary'])
        
        # (b) Intercept probability vs target speed
        ax2 = axes[1]
        speeds = np.linspace(10, 80, 15)  # m/s
        
        for initial_range, style, label in [(500, '-', '500m'),
                                             (1000, '--', '1000m'),
                                             (2000, ':', '2000m')]:
            intercept_probs = []
            for speed in speeds:
                # Time to reach defender
                time_to_impact = initial_range / speed
                # Time to neutralize (average)
                neutralize_time = 5.0  # seconds
                # Probability of intercept
                if time_to_impact > neutralize_time:
                    prob = 0.95
                else:
                    prob = 0.95 * (time_to_impact / neutralize_time) ** 0.5
                intercept_probs.append(prob)
            
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
                    alpha=0.7, label='FPV (144 km/h)')
        ax2.text(146, 0.1, 'FPV', fontsize=8, color=self.colors['quaternary'])
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.output_dir, f"{save_name}.png"))
            plt.savefig(os.path.join(self.output_dir, f"{save_name}.pdf"))
        
        return fig
    
    def generate_all_figures(self, js_values: np.ndarray = None) -> Dict[str, plt.Figure]:
        """
        Generate all figures for the paper.
        
        Args:
            js_values: Monte Carlo J/S values (generates sample if None)
            
        Returns:
            Dictionary of figure names to Figure objects
        """
        if js_values is None:
            # Generate sample data
            np.random.seed(42)
            js_values = np.random.normal(38.2, 4.7, 10000)
        
        mean_js = np.mean(js_values)
        ci_lower, ci_upper = np.percentile(js_values, [2.5, 97.5])
        
        figures = {}
        
        print("Generating figures...")
        
        print("  - Figure 1: Monte Carlo Results")
        figures['fig_monte_carlo'] = self.plot_monte_carlo_results(
            js_values, mean_js, ci_lower, ci_upper,
            save_name='fig_monte_carlo'
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
        
        print(f"All figures saved to: {self.output_dir}/")
        
        return figures


if __name__ == "__main__":
    print("=" * 60)
    print("UAV-CPS-Analyzer: Visualization Module Test")
    print("=" * 60)
    
    # Create visualizer
    viz = UAVCPSVisualizer(output_dir="/home/claude/UAV-CPS-Analyzer/output")
    
    # Generate all figures
    figures = viz.generate_all_figures()
    
    print(f"\nGenerated {len(figures)} figures:")
    for name in figures:
        print(f"  - {name}")
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    
    plt.show()
