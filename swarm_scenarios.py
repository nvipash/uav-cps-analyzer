#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: UAV Swarm Scenarios
Models attacks by 10-100 coordinated UAVs against C-UAS systems.

Swarm types:
- Cooperative — coordinated approach with task allocation
- Decentralized — each UAV acts independently (consensus-free)
- Kamikaze — single-use attack, no return path
- Decoy + strike — some UAVs as decoys, others carry payload

Metrics:
- Saturation point — max swarm size handled by N jammers
- Survival rate — % UAVs reaching target
- Time-to-saturation
- Per-UAV vs aggregate jamming probability

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum

try:
    from monte_carlo_engine import MonteCarloEngine, SimulationParams
    from multi_jammer_coordination import (
        MultiJammerSimulator, JammerNetwork, JammerNode, create_default_networks
    )
    from propagation_models import BERModel, ModulationType
except ImportError:
    import sys
    sys.path.insert(0, '.')
    from monte_carlo_engine import MonteCarloEngine, SimulationParams
    from multi_jammer_coordination import (
        MultiJammerSimulator, JammerNetwork, JammerNode, create_default_networks
    )
    from propagation_models import BERModel, ModulationType


class SwarmType(Enum):
    COOPERATIVE = "cooperative"          # coordinated, task-allocated
    DECENTRALIZED = "decentralized"      # independent agents
    KAMIKAZE = "kamikaze"                  # one-way attack
    DECOY_STRIKE = "decoy_strike"          # decoys + real attackers


@dataclass
class SwarmUAV:
    """Single UAV in a swarm."""
    id: str
    position: Tuple[float, float, float]
    velocity_ms: float
    direction_deg: float
    is_decoy: bool = False
    target_pos: Tuple[float, float] = (0.0, 0.0)
    survives: bool = True


@dataclass
class SwarmConfig:
    """Configuration for a UAV swarm."""
    swarm_type: SwarmType
    n_uavs: int
    formation: str = "circular"   # circular | wedge | line | random
    initial_radius_m: float = 3000.0
    altitude_m: float = 100.0
    speed_ms: float = 20.0
    decoy_fraction: float = 0.0   # fraction that are decoys (for DECOY_STRIKE)


@dataclass
class SwarmAttackResult:
    """Result of a swarm attack scenario."""
    swarm_type: str
    n_uavs: int
    n_jammers: int
    survivors: int
    survival_rate: float
    mean_jam_prob_per_uav: float
    peak_simultaneous_active: int
    saturation_observed: bool
    time_to_first_breach_s: float
    coverage_efficiency: float    # jamming_capacity / swarm_size


class SwarmGenerator:
    """Generates UAV swarms in different formations."""

    @staticmethod
    def generate(config: SwarmConfig, target_pos: Tuple[float, float] = (0.0, 0.0),
                  seed: int = 42) -> List[SwarmUAV]:
        """Generate swarm of UAVs in initial positions."""
        np.random.seed(seed)
        uavs = []
        n = config.n_uavs

        if config.formation == "circular":
            # Surround target at initial radius
            for i in range(n):
                angle = 2 * np.pi * i / n
                x = config.initial_radius_m * np.cos(angle)
                y = config.initial_radius_m * np.sin(angle)
                # Direction toward target
                dir_deg = (np.degrees(np.arctan2(target_pos[1] - y, target_pos[0] - x))) % 360
                uavs.append(SwarmUAV(
                    id=f"U{i:03d}",
                    position=(float(x), float(y), config.altitude_m),
                    velocity_ms=config.speed_ms,
                    direction_deg=float(dir_deg),
                    target_pos=target_pos
                ))
        elif config.formation == "wedge":
            # V-shape approach
            for i in range(n):
                row = i // 2
                col = (i % 2) * 2 - 1  # -1 or +1
                offset_x = -config.initial_radius_m + row * 50
                offset_y = col * (row * 100 + 50)
                uavs.append(SwarmUAV(
                    id=f"U{i:03d}",
                    position=(float(offset_x), float(offset_y), config.altitude_m),
                    velocity_ms=config.speed_ms,
                    direction_deg=0.0,
                    target_pos=target_pos
                ))
        elif config.formation == "line":
            # Straight line at fixed distance
            for i in range(n):
                y = (i - n / 2) * 100
                uavs.append(SwarmUAV(
                    id=f"U{i:03d}",
                    position=(-config.initial_radius_m, float(y), config.altitude_m),
                    velocity_ms=config.speed_ms, direction_deg=0.0,
                    target_pos=target_pos
                ))
        else:  # random
            for i in range(n):
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(0.7 * config.initial_radius_m, config.initial_radius_m)
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                dir_deg = (np.degrees(np.arctan2(target_pos[1] - y, target_pos[0] - x))) % 360
                uavs.append(SwarmUAV(
                    id=f"U{i:03d}",
                    position=(float(x), float(y), config.altitude_m),
                    velocity_ms=config.speed_ms,
                    direction_deg=float(dir_deg),
                    target_pos=target_pos
                ))

        # Mark decoys for DECOY_STRIKE
        if config.swarm_type == SwarmType.DECOY_STRIKE and config.decoy_fraction > 0:
            n_decoys = int(n * config.decoy_fraction)
            decoy_indices = np.random.choice(n, size=n_decoys, replace=False)
            for idx in decoy_indices:
                uavs[idx].is_decoy = True

        return uavs


class SwarmAttackSimulator:
    """Simulates swarm vs C-UAS jammer network engagement."""

    def __init__(self, jammer_simulator: MultiJammerSimulator = None):
        self.jam_sim = jammer_simulator or MultiJammerSimulator(n_iterations=100)

    def simulate_attack(self, swarm: List[SwarmUAV],
                          jammer_network: JammerNetwork,
                          duration_s: float = 60.0,
                          time_steps: int = 20,
                          jam_threshold: float = 0.5,
                          environment: str = "urban",
                          swarm_type: SwarmType = SwarmType.COOPERATIVE
                          ) -> SwarmAttackResult:
        """
        Simulate swarm attack over time.

        Each time step: advance UAVs, compute J/S for each, mark jammed.
        Cooperative: jammed UAVs slow other UAVs (rely on shared comm).
        Decentralized: each UAV independent, no comm sharing penalty.
        Kamikaze: continue toward target even when jammed (autonomous).

        Returns:
            SwarmAttackResult with aggregate metrics
        """
        dt = duration_s / time_steps
        time_to_breach = -1.0
        peak_active = 0
        all_jam_probs = []

        # Track which UAVs are jammed at each time step
        for step in range(time_steps):
            t = step * dt
            active_count = 0

            for uav in swarm:
                if not uav.survives:
                    continue

                # Advance position
                rad = np.radians(uav.direction_deg)
                new_x = uav.position[0] + uav.velocity_ms * dt * np.cos(rad)
                new_y = uav.position[1] + uav.velocity_ms * dt * np.sin(rad)
                uav.position = (float(new_x), float(new_y), uav.position[2])

                # Compute combined J/S from network
                js, p_jam, _ = self.jam_sim.combine_jammer_powers(
                    jammer_network, uav.position, environment=environment
                )
                all_jam_probs.append(p_jam)

                # Determine outcome based on swarm type
                if swarm_type == SwarmType.KAMIKAZE:
                    # Kamikaze: continue even if jammed (autonomous), but jamming
                    # disrupts target acquisition — survives RF but maybe loses guidance
                    # Use 0.8 threshold instead of 0.5
                    if p_jam > 0.8:
                        uav.survives = False
                elif swarm_type == SwarmType.COOPERATIVE:
                    # Cooperative: needs comm; jammed = lost
                    if p_jam > jam_threshold:
                        uav.survives = False
                elif swarm_type == SwarmType.DECENTRALIZED:
                    # Decentralized: more resilient, individual link
                    if p_jam > jam_threshold + 0.2:
                        uav.survives = False
                else:  # DECOY_STRIKE
                    if uav.is_decoy:
                        # Decoys absorb jamming, never marked dead
                        pass
                    else:
                        # Real attackers: standard behavior
                        if p_jam > jam_threshold:
                            uav.survives = False

                if uav.survives:
                    active_count += 1
                    # Check if reached target
                    d_to_target = np.sqrt(
                        (uav.position[0] - uav.target_pos[0]) ** 2
                        + (uav.position[1] - uav.target_pos[1]) ** 2
                    )
                    if d_to_target < 100 and time_to_breach < 0:
                        time_to_breach = t

            peak_active = max(peak_active, active_count)

        # Final stats
        survivors = sum(1 for u in swarm if u.survives)
        # Real survivors exclude decoys (which always survive but don't count)
        if swarm_type == SwarmType.DECOY_STRIKE:
            real_survivors = sum(1 for u in swarm if u.survives and not u.is_decoy)
        else:
            real_survivors = survivors

        n_real = sum(1 for u in swarm if not u.is_decoy)
        survival_rate = real_survivors / max(1, n_real)

        # Coverage efficiency: jammers per UAV ratio
        coverage_eff = len(jammer_network.nodes) / max(1, len(swarm))
        saturation = peak_active > 5 * len(jammer_network.nodes)

        return SwarmAttackResult(
            swarm_type=swarm_type.value,
            n_uavs=len(swarm),
            n_jammers=len(jammer_network.nodes),
            survivors=real_survivors,
            survival_rate=survival_rate,
            mean_jam_prob_per_uav=float(np.mean(all_jam_probs)) if all_jam_probs else 0.0,
            peak_simultaneous_active=peak_active,
            saturation_observed=saturation,
            time_to_first_breach_s=time_to_breach,
            coverage_efficiency=coverage_eff
        )

    def saturation_curve(self, jammer_network: JammerNetwork,
                          swarm_sizes: List[int] = None,
                          swarm_type: SwarmType = SwarmType.COOPERATIVE
                          ) -> List[SwarmAttackResult]:
        """
        Build saturation curve: survival rate vs swarm size.
        Identifies the saturation point of the jammer network.
        """
        if swarm_sizes is None:
            swarm_sizes = [5, 10, 20, 30, 50, 80]

        results = []
        for n in swarm_sizes:
            config = SwarmConfig(
                swarm_type=swarm_type, n_uavs=n,
                formation="circular", initial_radius_m=3000.0,
                altitude_m=100.0, speed_ms=20.0
            )
            swarm = SwarmGenerator.generate(config)
            result = self.simulate_attack(swarm, jammer_network,
                                            duration_s=60.0, time_steps=15,
                                            swarm_type=swarm_type)
            results.append(result)
        return results


def print_swarm_results(results: List[SwarmAttackResult]):
    """Print formatted swarm attack results."""
    print(f"  {'Type':<14} {'#UAVs':<6} {'#Jam':<5} {'Survive':<10} {'Rate':<8} "
          f"{'Mean P(jam)':<12} {'Saturated'}", flush=True)
    print("  " + "-" * 75, flush=True)
    for r in results:
        sat = "Yes" if r.saturation_observed else "No"
        print(f"  {r.swarm_type:<14} {r.n_uavs:<6} {r.n_jammers:<5} "
              f"{r.survivors:<10} {r.survival_rate*100:<7.1f}% "
              f"{r.mean_jam_prob_per_uav:<12.3f} {sat}", flush=True)


if __name__ == "__main__":
    print("=" * 60)
    print("UAV Swarm Attack Scenarios")
    print("=" * 60)

    networks = create_default_networks()
    sim = SwarmAttackSimulator()

    # Test different swarm types vs 5-mesh network
    print("\n1. Swarm types vs 5-mesh jammer network (n=20 UAVs):")
    for stype in SwarmType:
        config = SwarmConfig(swarm_type=stype, n_uavs=20,
                              decoy_fraction=0.5 if stype == SwarmType.DECOY_STRIKE else 0)
        swarm = SwarmGenerator.generate(config)
        result = sim.simulate_attack(swarm, networks['5_mesh'], swarm_type=stype)
        print(f"  {stype.value:<14}: {result.survivors}/{result.n_uavs} survive "
              f"({result.survival_rate*100:.1f}%)")

    # Saturation curve for cooperative swarm
    print("\n2. Saturation curve (cooperative vs 3_triangle):")
    sat_results = sim.saturation_curve(networks['3_triangle'],
                                          swarm_sizes=[5, 10, 20, 30, 50])
    print_swarm_results(sat_results)
