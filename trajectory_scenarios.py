#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: Time-Varying Trajectory Scenarios
Models UAV trajectories (Bezier/linear/circular) and computes time-series J/S evolution.

Scenarios:
- Approach attack: UAV flies toward jammer along straight line
- Fly-by: UAV passes near jammer at constant altitude
- Circular orbit: UAV circles around target area
- Bezier evasive: UAV uses curved evasive trajectory

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable

try:
    from monte_carlo_engine import MonteCarloEngine, SimulationParams
except ImportError:
    import sys
    sys.path.insert(0, '.')
    from monte_carlo_engine import MonteCarloEngine, SimulationParams


@dataclass
class TrajectoryPoint:
    """Single point in UAV trajectory."""
    t: float           # time (seconds)
    x: float           # horizontal position (m)
    y: float           # horizontal position (m)
    z: float           # altitude (m)
    velocity_ms: float  # speed at this point


@dataclass
class TrajectoryResult:
    """Result of trajectory simulation."""
    name: str
    times: np.ndarray
    distances: np.ndarray         # horizontal distance to jammer over time
    altitudes: np.ndarray
    js_values: np.ndarray         # J/S over time (mean)
    js_lower: np.ndarray          # 95% CI lower
    js_upper: np.ndarray          # 95% CI upper
    success_probs: np.ndarray     # P(jam) over time
    time_jammed_s: float          # cumulative time with P(jam) > 0.5
    min_distance_m: float          # closest approach
    max_js_db: float
    n_segments: int


class TrajectoryGenerator:
    """Generates UAV trajectories of different types."""

    @staticmethod
    def linear_approach(start: Tuple[float, float, float],
                         end: Tuple[float, float, float],
                         duration_s: float, n_points: int = 20) -> List[TrajectoryPoint]:
        """Straight-line trajectory from start to end."""
        ts = np.linspace(0, duration_s, n_points)
        sx, sy, sz = start
        ex, ey, ez = end
        velocity = np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2 + (ez - sz) ** 2) / duration_s
        points = []
        for i, t in enumerate(ts):
            frac = t / duration_s if duration_s > 0 else 0
            points.append(TrajectoryPoint(
                t=t,
                x=sx + frac * (ex - sx),
                y=sy + frac * (ey - sy),
                z=sz + frac * (ez - sz),
                velocity_ms=velocity
            ))
        return points

    @staticmethod
    def fly_by(jammer_pos: Tuple[float, float, float],
                offset_distance_m: float = 500.0,
                altitude_m: float = 100.0,
                travel_distance_m: float = 4000.0,
                duration_s: float = 60.0,
                n_points: int = 25) -> List[TrajectoryPoint]:
        """UAV passes near jammer with offset distance."""
        ts = np.linspace(0, duration_s, n_points)
        velocity = travel_distance_m / duration_s
        jx, jy, jz = jammer_pos
        # Path: y = jy + offset_distance_m, x ranges from jx-half to jx+half
        half = travel_distance_m / 2
        points = []
        for i, t in enumerate(ts):
            frac = t / duration_s if duration_s > 0 else 0
            x = jx - half + frac * travel_distance_m
            y = jy + offset_distance_m
            z = altitude_m
            points.append(TrajectoryPoint(t=t, x=x, y=y, z=z, velocity_ms=velocity))
        return points

    @staticmethod
    def circular_orbit(center: Tuple[float, float, float],
                        radius_m: float = 1000.0,
                        altitude_m: float = 200.0,
                        n_revolutions: float = 1.0,
                        duration_s: float = 120.0,
                        n_points: int = 30) -> List[TrajectoryPoint]:
        """UAV circles around center point at fixed altitude."""
        ts = np.linspace(0, duration_s, n_points)
        cx, cy, cz = center
        velocity = 2 * np.pi * radius_m * n_revolutions / duration_s
        points = []
        for t in ts:
            angle = 2 * np.pi * n_revolutions * (t / duration_s)
            x = cx + radius_m * np.cos(angle)
            y = cy + radius_m * np.sin(angle)
            z = altitude_m
            points.append(TrajectoryPoint(t=t, x=x, y=y, z=z, velocity_ms=velocity))
        return points

    @staticmethod
    def bezier_evasive(start: Tuple[float, float, float],
                        end: Tuple[float, float, float],
                        control1: Tuple[float, float, float],
                        control2: Tuple[float, float, float],
                        duration_s: float = 90.0,
                        n_points: int = 30) -> List[TrajectoryPoint]:
        """
        Cubic Bezier evasive trajectory.
        B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
        """
        ts = np.linspace(0, duration_s, n_points)
        P0 = np.array(start)
        P1 = np.array(control1)
        P2 = np.array(control2)
        P3 = np.array(end)
        points = []
        prev_pos = P0
        for i, t in enumerate(ts):
            u = t / duration_s if duration_s > 0 else 0
            pos = ((1 - u) ** 3 * P0 + 3 * (1 - u) ** 2 * u * P1
                    + 3 * (1 - u) * u ** 2 * P2 + u ** 3 * P3)
            if i > 0:
                dt = ts[i] - ts[i - 1]
                velocity = float(np.linalg.norm(pos - prev_pos) / max(dt, 0.001))
            else:
                velocity = 0.0
            points.append(TrajectoryPoint(
                t=t, x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
                velocity_ms=velocity
            ))
            prev_pos = pos
        return points


class TrajectorySimulator:
    """Simulates J/S evolution along a UAV trajectory."""

    def __init__(self, n_iterations_per_point: int = 500):
        self.n_iters = n_iterations_per_point
        self.engine = MonteCarloEngine(n_processes=1)

    def simulate(self, name: str,
                  trajectory: List[TrajectoryPoint],
                  jammer_pos: Tuple[float, float, float],
                  base_params: SimulationParams = None) -> TrajectoryResult:
        """
        Simulate J/S evolution along trajectory.

        Args:
            name: Scenario name
            trajectory: List of trajectory points
            jammer_pos: Jammer position (x, y, z)
            base_params: Base simulation parameters

        Returns:
            TrajectoryResult with time-series data
        """
        if base_params is None:
            base_params = SimulationParams(
                jammer_power_dbm=40.0, fhss_enabled=False,
                propagation_model='al_hourani', environment='urban'
            )

        n = len(trajectory)
        times = np.zeros(n)
        distances = np.zeros(n)
        altitudes = np.zeros(n)
        js_values = np.zeros(n)
        js_lower = np.zeros(n)
        js_upper = np.zeros(n)
        success_probs = np.zeros(n)

        jx, jy, jz = jammer_pos

        for i, pt in enumerate(trajectory):
            # Compute geometry
            d_horizontal = float(np.sqrt((pt.x - jx) ** 2 + (pt.y - jy) ** 2))
            altitude = max(1.0, abs(pt.z - jz))

            params = SimulationParams(**vars(base_params))
            params.jammer_distance_m = max(10.0, d_horizontal)
            params.altitude_m = altitude
            params.target_velocity_ms = pt.velocity_ms

            r = self.engine.run_simulation(params, self.n_iters,
                                             parallel=False, random_seed=42 + i)

            times[i] = pt.t
            distances[i] = d_horizontal
            altitudes[i] = altitude
            js_values[i] = r.mean_js_db
            js_lower[i] = r.ci_95_lower
            js_upper[i] = r.ci_95_upper
            success_probs[i] = r.success_probability

        # Compute summary metrics
        # Time jammed = sum of dt where success_prob > 0.5
        time_jammed = 0.0
        for i in range(1, n):
            if success_probs[i] > 0.5:
                dt = times[i] - times[i - 1]
                time_jammed += dt

        return TrajectoryResult(
            name=name,
            times=times,
            distances=distances,
            altitudes=altitudes,
            js_values=js_values,
            js_lower=js_lower,
            js_upper=js_upper,
            success_probs=success_probs,
            time_jammed_s=time_jammed,
            min_distance_m=float(np.min(distances)),
            max_js_db=float(np.max(js_values)),
            n_segments=n,
        )

    def run_standard_scenarios(self) -> Dict[str, TrajectoryResult]:
        """Run a suite of standard trajectory scenarios."""
        results = {}
        jammer_pos = (0.0, 0.0, 50.0)  # jammer at origin

        # Approach attack: UAV from 3km away approaches jammer
        traj_approach = TrajectoryGenerator.linear_approach(
            start=(3000.0, 0.0, 100.0),
            end=(50.0, 0.0, 100.0),
            duration_s=60.0, n_points=15
        )
        results['approach_attack'] = self.simulate(
            "Approach attack (3km->50m)", traj_approach, jammer_pos
        )

        # Fly-by: UAV passes 500m offset
        traj_flyby = TrajectoryGenerator.fly_by(
            jammer_pos=jammer_pos, offset_distance_m=500.0,
            altitude_m=100.0, travel_distance_m=4000.0,
            duration_s=60.0, n_points=15
        )
        results['fly_by'] = self.simulate(
            "Fly-by (500m offset)", traj_flyby, jammer_pos
        )

        # Circular orbit at 1km radius
        traj_orbit = TrajectoryGenerator.circular_orbit(
            center=jammer_pos, radius_m=1000.0, altitude_m=200.0,
            n_revolutions=1.0, duration_s=90.0, n_points=15
        )
        results['circular_orbit'] = self.simulate(
            "Circular orbit (r=1km)", traj_orbit, jammer_pos
        )

        # Bezier evasive
        traj_bezier = TrajectoryGenerator.bezier_evasive(
            start=(2000.0, -500.0, 80.0),
            end=(2000.0, 500.0, 200.0),
            control1=(500.0, -800.0, 100.0),  # dip toward jammer first
            control2=(500.0, 800.0, 250.0),    # then away
            duration_s=70.0, n_points=15
        )
        results['bezier_evasive'] = self.simulate(
            "Bezier evasive maneuver", traj_bezier, jammer_pos
        )

        return results


def print_trajectory_summary(results: Dict[str, TrajectoryResult]):
    """Print summary of trajectory scenarios."""
    print(f"\n  {'Scenario':<30} {'Min dist':<10} {'Max J/S':<10} "
          f"{'Time jammed':<14} {'P(jam)>0.5'}")
    print("  " + "-" * 75)
    for name, r in results.items():
        time_jammed_pct = (r.time_jammed_s / r.times[-1] * 100) if r.times[-1] > 0 else 0
        n_jammed = int(np.sum(r.success_probs > 0.5))
        print(f"  {r.name:<30} {r.min_distance_m:<10.0f} {r.max_js_db:<10.1f} "
              f"{r.time_jammed_s:<6.1f}s ({time_jammed_pct:.0f}%) {n_jammed}/{r.n_segments} pts")


if __name__ == "__main__":
    print("=" * 60)
    print("Trajectory Scenarios — Time-Varying J/S Analysis")
    print("=" * 60)
    sim = TrajectorySimulator(n_iterations_per_point=300)
    results = sim.run_standard_scenarios()
    print_trajectory_summary(results)
