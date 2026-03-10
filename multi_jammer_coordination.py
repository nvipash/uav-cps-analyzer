#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: Multi-Jammer Coordination
Models distributed C-UAS jammer networks with coordinated coverage strategies.

Capabilities:
- Multiple jammers at different positions with power combining at target
- Mesh-network deployment optimization (genetic algorithm)
- Coverage analysis with overlapping zones
- Frequency band partitioning (FDMA-like)
- Coordinated strategy selection (each jammer different role)
- Defensive perimeter optimization

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from itertools import product

try:
    from monte_carlo_engine import MonteCarloEngine, SimulationParams
    from propagation_models import (
        AlHouraniA2GModel, AtmosphericAbsorption, UrbanMultiPathCorrection,
        BERModel, ModulationType
    )
except ImportError:
    import sys
    sys.path.insert(0, '.')
    from monte_carlo_engine import MonteCarloEngine, SimulationParams
    from propagation_models import (
        AlHouraniA2GModel, AtmosphericAbsorption, UrbanMultiPathCorrection,
        BERModel, ModulationType
    )


@dataclass
class JammerNode:
    """A single jammer in a mesh network."""
    id: str
    position: Tuple[float, float, float]   # (x, y, z) in meters
    power_dbm: float
    antenna_gain_dbi: float
    beam_width_deg: float = 60.0
    aim_angle_deg: float = 0.0              # azimuth direction (where antenna points)
    frequency_band: str = "2.4GHz"          # primary band
    bandwidth_mhz: float = 83.5
    role: str = "broadband"                 # broadband | narrowband | sweep | follower | reactive
    cost_usd: float = 5000.0


@dataclass
class JammerNetwork:
    """A coordinated network of jammers."""
    nodes: List[JammerNode]
    total_cost_usd: float = 0.0
    coordination_strategy: str = "all_active"   # all_active | sector_based | freq_partition

    def __post_init__(self):
        self.total_cost_usd = sum(n.cost_usd for n in self.nodes)


@dataclass
class CoverageResult:
    """Result of coverage analysis over an area."""
    grid_x: np.ndarray
    grid_y: np.ndarray
    jam_probability: np.ndarray   # 2D array of P(jam) over grid
    js_combined_db: np.ndarray    # 2D array of combined J/S over grid
    coverage_pct: float           # % of area with P(jam) > 0.5
    deadzone_pct: float            # % of area with P(jam) < 0.1 (gap)
    n_jammers: int


class MultiJammerSimulator:
    """
    Simulates coordinated multi-jammer networks vs UAV at various positions.
    """

    def __init__(self, n_iterations: int = 1000):
        self.n_iters = n_iterations
        self.engine = MonteCarloEngine(n_processes=1)

    def combine_jammer_powers(self, network: JammerNetwork,
                                target_pos: Tuple[float, float, float],
                                signal_distance_m: float = 5000.0,
                                operator_pos: Tuple[float, float] = None,
                                environment: str = "urban",
                                frequency_mhz: float = 2437.0
                                ) -> Tuple[float, float, float]:
        """
        Compute combined J/S at target from all jammers in network.

        Combines jammer powers in the linear domain (true incoherent sum
        of independent jammers), then computes effective J/S vs single signal.

        Args:
            network: Jammer network configuration
            target_pos: Target UAV position (x, y, z) in meters
            signal_distance_m: Distance from UAV operator to UAV
            operator_pos: Operator position (x, y) — if None, assume far from network
            environment: Propagation environment

        Returns:
            (combined_js_db, p_jam, dominant_jammer_contribution_pct)
        """
        tx, ty, tz = target_pos
        model = AlHouraniA2GModel(environment=environment)

        # Total jammer power received at target (linear scale)
        total_jammer_linear = 0.0
        per_jammer_powers = []

        for node in network.nodes:
            jx, jy, jz = node.position
            d_horiz = float(np.sqrt((tx - jx) ** 2 + (ty - jy) ** 2))
            altitude = max(1.0, abs(tz - jz))

            # Path loss
            pl = model.path_loss(max(10.0, d_horiz), altitude, frequency_mhz)
            # Atmospheric + multipath corrections
            pl += AtmosphericAbsorption.total_loss_db(d_horiz, frequency_mhz)
            pl += UrbanMultiPathCorrection.multi_path_loss_db(d_horiz, environment, altitude)

            # Antenna pattern: compute angle between aim direction and target
            target_angle = float(np.degrees(np.arctan2(ty - jy, tx - jx)))
            angular_offset = abs(((target_angle - node.aim_angle_deg + 180) % 360) - 180)

            # Cosine pattern with beam width
            half_bw = node.beam_width_deg / 2
            if angular_offset >= half_bw:
                # Outside main beam — strong attenuation
                effective_gain = node.antenna_gain_dbi - 20.0
            else:
                # Within beam, cosine taper
                cos_factor = max(0.001, np.cos(np.radians(angular_offset)) ** 2)
                effective_gain = node.antenna_gain_dbi + 10 * np.log10(cos_factor)

            # Sector/frequency partition: only some jammers contribute to relevant band
            band_factor = 1.0
            if network.coordination_strategy == "freq_partition":
                # Each jammer covers a fraction of the band
                n = len(network.nodes)
                band_factor = 1.0 / n if n > 1 else 1.0

            # Received jammer power at target (in dBm)
            rx_jammer_dbm = node.power_dbm + effective_gain - pl + 10 * np.log10(band_factor)
            rx_jammer_linear = 10 ** (rx_jammer_dbm / 10)
            total_jammer_linear += rx_jammer_linear
            per_jammer_powers.append(rx_jammer_linear)

        if total_jammer_linear < 1e-30:
            return -100.0, 0.0, 0.0

        rx_jammer_dbm_total = 10 * np.log10(total_jammer_linear)

        # Signal: assume operator at far edge, fixed distance
        # Signal path loss
        pl_signal = model.path_loss(signal_distance_m, abs(tz), frequency_mhz)
        pl_signal += AtmosphericAbsorption.total_loss_db(signal_distance_m, frequency_mhz)
        pl_signal += UrbanMultiPathCorrection.multi_path_loss_db(signal_distance_m, environment, abs(tz))
        rx_signal_dbm = 20.0 + 2.0 - pl_signal  # 100mW transmit, 2 dBi antenna gain

        # J/S
        js_db = rx_jammer_dbm_total - rx_signal_dbm

        # P(jam) using BER model
        p_jam = BERModel.jamming_success_probability(js_db, signal_power_db=rx_signal_dbm)

        # Dominant jammer contribution
        max_contrib = max(per_jammer_powers) / total_jammer_linear * 100 if per_jammer_powers else 0

        return float(js_db), float(p_jam), float(max_contrib)

    def coverage_map(self, network: JammerNetwork,
                      x_range: Tuple[float, float] = (-3000, 3000),
                      y_range: Tuple[float, float] = (-3000, 3000),
                      altitude_m: float = 100.0,
                      grid_size: int = 25,
                      environment: str = "urban") -> CoverageResult:
        """
        Generate 2D coverage heatmap of P(jam) over geographic area.

        Args:
            network: Jammer network
            x_range, y_range: Grid extents in meters
            altitude_m: UAV altitude
            grid_size: NxN grid resolution
            environment: Propagation environment

        Returns:
            CoverageResult with 2D grid and aggregate metrics
        """
        x_grid = np.linspace(x_range[0], x_range[1], grid_size)
        y_grid = np.linspace(y_range[0], y_range[1], grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)

        jam_prob = np.zeros_like(X)
        js_combined = np.zeros_like(X)

        for i in range(grid_size):
            for j in range(grid_size):
                target = (float(X[i, j]), float(Y[i, j]), altitude_m)
                js, p_j, _ = self.combine_jammer_powers(
                    network, target, environment=environment
                )
                jam_prob[i, j] = p_j
                js_combined[i, j] = js

        # Aggregate coverage metrics
        coverage_pct = float(np.mean(jam_prob > 0.5) * 100)
        deadzone_pct = float(np.mean(jam_prob < 0.1) * 100)

        return CoverageResult(
            grid_x=X, grid_y=Y, jam_probability=jam_prob,
            js_combined_db=js_combined,
            coverage_pct=coverage_pct, deadzone_pct=deadzone_pct,
            n_jammers=len(network.nodes)
        )


class JammerNetworkOptimizer:
    """
    Optimizes jammer placement using genetic algorithm-like approach.
    Maximizes coverage area subject to budget constraint.
    """

    def __init__(self, simulator: MultiJammerSimulator):
        self.sim = simulator

    def optimize_placement(self, n_jammers: int,
                            budget_usd: float,
                            area_x: Tuple[float, float] = (-2000, 2000),
                            area_y: Tuple[float, float] = (-2000, 2000),
                            altitude_m: float = 100.0,
                            n_generations: int = 30,
                            population_size: int = 20,
                            grid_size: int = 15) -> Tuple[JammerNetwork, List[float]]:
        """
        Find optimal placement of n_jammers within budget.

        Genetic algorithm: each individual = (positions of all jammers).
        Fitness = coverage_pct - deadzone_penalty - cost_penalty.

        Returns:
            (best_network, history_of_best_coverage)
        """
        # Default jammer template (mid-tier)
        cost_per = budget_usd / n_jammers

        if cost_per >= 50000:
            template = {'power_dbm': 50.0, 'antenna_gain_dbi': 10.0, 'beam_width': 30.0, 'cost': 50000}
        elif cost_per >= 15000:
            template = {'power_dbm': 45.0, 'antenna_gain_dbi': 8.0, 'beam_width': 45.0, 'cost': 15000}
        else:
            template = {'power_dbm': 40.0, 'antenna_gain_dbi': 6.0, 'beam_width': 60.0, 'cost': 5000}

        def random_individual():
            """Random network (positions and aim angles)."""
            nodes = []
            for i in range(n_jammers):
                x = np.random.uniform(area_x[0], area_x[1])
                y = np.random.uniform(area_y[0], area_y[1])
                aim = np.random.uniform(0, 360)
                nodes.append(JammerNode(
                    id=f"J{i}", position=(float(x), float(y), 30.0),
                    power_dbm=template['power_dbm'],
                    antenna_gain_dbi=template['antenna_gain_dbi'],
                    beam_width_deg=template['beam_width'],
                    aim_angle_deg=float(aim),
                    cost_usd=template['cost']
                ))
            return JammerNetwork(nodes=nodes)

        def fitness(network):
            """Higher = better (coverage maximized, deadzones minimized)."""
            cr = self.sim.coverage_map(network, area_x, area_y, altitude_m,
                                          grid_size=grid_size)
            return cr.coverage_pct - 0.5 * cr.deadzone_pct

        def mutate(network, rate=0.3):
            """Random small perturbation of positions/aims."""
            new_nodes = []
            for n in network.nodes:
                if np.random.random() < rate:
                    nx = n.position[0] + np.random.randn() * 300
                    ny = n.position[1] + np.random.randn() * 300
                    nx = float(np.clip(nx, area_x[0], area_x[1]))
                    ny = float(np.clip(ny, area_y[0], area_y[1]))
                    aim = (n.aim_angle_deg + np.random.randn() * 30) % 360
                    new_nodes.append(JammerNode(
                        id=n.id, position=(nx, ny, n.position[2]),
                        power_dbm=n.power_dbm, antenna_gain_dbi=n.antenna_gain_dbi,
                        beam_width_deg=n.beam_width_deg, aim_angle_deg=float(aim),
                        cost_usd=n.cost_usd
                    ))
                else:
                    new_nodes.append(n)
            return JammerNetwork(nodes=new_nodes)

        # Initial population
        population = [random_individual() for _ in range(population_size)]
        history = []

        for gen in range(n_generations):
            # Evaluate fitness
            scored = [(fitness(ind), ind) for ind in population]
            scored.sort(key=lambda x: -x[0])
            history.append(scored[0][0])

            # Keep top half, mutate them to fill rest
            survivors = [s[1] for s in scored[:population_size // 2]]
            new_pop = list(survivors)
            while len(new_pop) < population_size:
                parent = survivors[np.random.randint(len(survivors))]
                new_pop.append(mutate(parent))
            population = new_pop

        # Final best
        scored = [(fitness(ind), ind) for ind in population]
        scored.sort(key=lambda x: -x[0])
        best = scored[0][1]
        return best, history


def create_default_networks() -> Dict[str, JammerNetwork]:
    """Create reference jammer networks for benchmarking."""
    networks = {}

    # Single portable jammer (baseline)
    networks['1_portable'] = JammerNetwork(nodes=[
        JammerNode(id="J0", position=(0, 0, 30), power_dbm=40, antenna_gain_dbi=6,
                    beam_width_deg=60, aim_angle_deg=0, cost_usd=5000)
    ])

    # 3-jammer triangle (perimeter)
    r = 1500
    networks['3_triangle'] = JammerNetwork(nodes=[
        JammerNode(id="J0", position=(r * np.cos(0), r * np.sin(0), 30),
                    power_dbm=40, antenna_gain_dbi=6, beam_width_deg=60,
                    aim_angle_deg=180, cost_usd=5000),
        JammerNode(id="J1", position=(r * np.cos(2 * np.pi / 3), r * np.sin(2 * np.pi / 3), 30),
                    power_dbm=40, antenna_gain_dbi=6, beam_width_deg=60,
                    aim_angle_deg=300, cost_usd=5000),
        JammerNode(id="J2", position=(r * np.cos(4 * np.pi / 3), r * np.sin(4 * np.pi / 3), 30),
                    power_dbm=40, antenna_gain_dbi=6, beam_width_deg=60,
                    aim_angle_deg=60, cost_usd=5000),
    ])

    # 5-jammer mesh (mixed roles)
    networks['5_mesh'] = JammerNetwork(nodes=[
        JammerNode(id="C", position=(0, 0, 50), power_dbm=50, antenna_gain_dbi=10,
                    beam_width_deg=360, aim_angle_deg=0, cost_usd=20000),  # central omni
        JammerNode(id="N", position=(0, 1500, 30), power_dbm=43, antenna_gain_dbi=8,
                    beam_width_deg=90, aim_angle_deg=270, cost_usd=8000),
        JammerNode(id="S", position=(0, -1500, 30), power_dbm=43, antenna_gain_dbi=8,
                    beam_width_deg=90, aim_angle_deg=90, cost_usd=8000),
        JammerNode(id="E", position=(1500, 0, 30), power_dbm=43, antenna_gain_dbi=8,
                    beam_width_deg=90, aim_angle_deg=180, cost_usd=8000),
        JammerNode(id="W", position=(-1500, 0, 30), power_dbm=43, antenna_gain_dbi=8,
                    beam_width_deg=90, aim_angle_deg=0, cost_usd=8000),
    ])

    # High-power 2-stationary
    networks['2_stationary'] = JammerNetwork(nodes=[
        JammerNode(id="J0", position=(-1000, 0, 50), power_dbm=57, antenna_gain_dbi=15,
                    beam_width_deg=30, aim_angle_deg=0, cost_usd=100000),
        JammerNode(id="J1", position=(1000, 0, 50), power_dbm=57, antenna_gain_dbi=15,
                    beam_width_deg=30, aim_angle_deg=180, cost_usd=100000),
    ])

    return networks


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Jammer Coordination Network Analysis")
    print("=" * 60)

    sim = MultiJammerSimulator(n_iterations=200)
    networks = create_default_networks()

    print(f"\n{'Network':<18} {'#J':<4} {'Cost':<10} {'Coverage':<10} {'Deadzone':<10}")
    print("-" * 60)
    for name, net in networks.items():
        cr = sim.coverage_map(net, grid_size=20)
        print(f"{name:<18} {cr.n_jammers:<4} ${net.total_cost_usd:<9,.0f} "
              f"{cr.coverage_pct:<10.1f}% {cr.deadzone_pct:<10.1f}%")

    # Optional: run optimizer (slow)
    # opt = JammerNetworkOptimizer(sim)
    # best, history = opt.optimize_placement(n_jammers=3, budget_usd=15000,
    #                                          n_generations=10, population_size=10)
    # print(f"\nOptimized 3-jammer network: {history[-1]:.1f}% coverage")
