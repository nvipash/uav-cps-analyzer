#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: AI-vs-AI Adversarial Co-evolution
Both jammer and FHSS protocol learn simultaneously through multi-agent RL.

Game-theoretic framework:
- Jammer agent chooses strategy (broadband, narrowband, sweep, follower)
- FHSS agent chooses defense (hop pattern, channel selection, modulation)
- Both adapt simultaneously — Nash equilibrium emerges
- Tracks convergence, equilibrium strategies, exploitation gaps

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

try:
    from fhss_emulator import (
        OcuSyncProtocol, JammingStrategy, JammingEffectivenessAnalyzer
    )
except ImportError:
    import sys
    sys.path.insert(0, '.')
    from fhss_emulator import (
        OcuSyncProtocol, JammingStrategy, JammingEffectivenessAnalyzer
    )


# Jammer actions: strategy selection
JAMMER_ACTIONS = [
    JammingStrategy.BROADBAND,
    JammingStrategy.NARROWBAND,
    JammingStrategy.SWEEP,
    JammingStrategy.FOLLOWER,
]
N_JAMMER_ACTIONS = len(JAMMER_ACTIONS)

# FHSS defender actions: defensive posture
FHSS_DEFENSES = [
    "uniform_hopping",       # baseline LFSR
    "avoid_jammed_channels",  # AMC reactive
    "high_modulation",        # 64-QAM (more data per hop)
    "spread_pattern",         # randomized timing
    "frequency_diversity",    # spread across more bands
]
N_FHSS_ACTIONS = len(FHSS_DEFENSES)

# Joint state space (jammer_action_history_bin, fhss_defense_history_bin)
N_STATE_BINS = 4
N_JOINT_STATES = N_STATE_BINS * N_STATE_BINS


@dataclass
class CoevolutionResult:
    """Result of AI-vs-AI co-evolution training."""
    n_episodes: int
    jammer_history: List[float]      # mean jam_rate per episode
    fhss_history: List[float]         # mean survival_rate per episode
    final_jam_rate: float
    final_survival_rate: float
    jammer_strategy_mix: Dict[str, float]  # frequency of each jammer action
    fhss_strategy_mix: Dict[str, float]    # frequency of each defense
    nash_distance: float              # how close to Nash equilibrium
    cycles_detected: int              # number of strategy oscillations


def _discretize(jam_rate: float) -> int:
    """Bin jam_rate into N_STATE_BINS levels."""
    return min(N_STATE_BINS - 1, int(jam_rate * N_STATE_BINS))


class JammerAgent:
    """Q-learning jammer agent."""

    def __init__(self, lr: float = 0.1, gamma: float = 0.9,
                 epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((N_JOINT_STATES, N_JAMMER_ACTIONS))

    def act(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(N_JAMMER_ACTIONS)
        return int(np.argmax(self.q_table[state]))

    def learn(self, s: int, a: int, r: float, s_next: int):
        td_target = r + self.gamma * np.max(self.q_table[s_next])
        self.q_table[s, a] += self.lr * (td_target - self.q_table[s, a])

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class FHSSAgent:
    """Q-learning FHSS defender agent (mirror image of jammer)."""

    def __init__(self, lr: float = 0.1, gamma: float = 0.9,
                 epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((N_JOINT_STATES, N_FHSS_ACTIONS))

    def act(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(N_FHSS_ACTIONS)
        return int(np.argmax(self.q_table[state]))

    def learn(self, s: int, a: int, r: float, s_next: int):
        td_target = r + self.gamma * np.max(self.q_table[s_next])
        self.q_table[s, a] += self.lr * (td_target - self.q_table[s, a])

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class CoevolutionEnvironment:
    """
    Multi-agent environment.
    Reward structure (zero-sum game):
    - r_jammer = +jam_rate
    - r_fhss = -jam_rate (or +survival_rate)
    """

    BASE_EFFECTIVENESS = {
        JammingStrategy.BROADBAND: 0.50,
        JammingStrategy.NARROWBAND: 0.025,
        JammingStrategy.SWEEP: 0.45,
        JammingStrategy.FOLLOWER: 0.49,
    }

    # Defense modifiers: how each defense reduces jamming effectiveness
    DEFENSE_MULTIPLIERS = {
        "uniform_hopping":         {JammingStrategy.BROADBAND: 1.0, JammingStrategy.NARROWBAND: 1.0,
                                     JammingStrategy.SWEEP: 1.0, JammingStrategy.FOLLOWER: 1.0},
        "avoid_jammed_channels":   {JammingStrategy.BROADBAND: 0.85, JammingStrategy.NARROWBAND: 0.10,
                                     JammingStrategy.SWEEP: 0.50, JammingStrategy.FOLLOWER: 0.40},
        "high_modulation":          {JammingStrategy.BROADBAND: 1.10, JammingStrategy.NARROWBAND: 1.10,
                                     JammingStrategy.SWEEP: 1.10, JammingStrategy.FOLLOWER: 1.10},
        "spread_pattern":          {JammingStrategy.BROADBAND: 0.95, JammingStrategy.NARROWBAND: 0.50,
                                     JammingStrategy.SWEEP: 0.60, JammingStrategy.FOLLOWER: 0.30},
        "frequency_diversity":     {JammingStrategy.BROADBAND: 0.40, JammingStrategy.NARROWBAND: 0.20,
                                     JammingStrategy.SWEEP: 0.70, JammingStrategy.FOLLOWER: 0.50},
    }

    def __init__(self, window_hops: int = 50, js_ratio_db: float = 30.0):
        self.window_hops = window_hops
        self.js_ratio_db = js_ratio_db
        self.jam_history = []
        self.def_history = []

    def reset(self) -> int:
        self.jam_history = []
        self.def_history = []
        return _discretize(0.0) * N_STATE_BINS + _discretize(0.0)

    def step(self, jam_action: int, def_action: int) -> Tuple[int, float, float]:
        """
        Both agents act simultaneously.

        Returns:
            (next_state, r_jammer, r_fhss)
        """
        strategy = JAMMER_ACTIONS[jam_action]
        defense = FHSS_DEFENSES[def_action]

        # Base effectiveness modulated by defense
        base_eff = self.BASE_EFFECTIVENESS[strategy]
        modifier = self.DEFENSE_MULTIPLIERS[defense][strategy]
        effectiveness = max(0.001, min(1.0, base_eff * modifier))

        # Simulate hops
        jammed = 0
        for _ in range(self.window_hops):
            if np.random.random() < effectiveness and self.js_ratio_db > 0:
                # Soft jam probability
                jam_prob = min(1.0, self.js_ratio_db / 20.0)
                if np.random.random() < jam_prob:
                    jammed += 1
        jam_rate = jammed / self.window_hops

        # Track history for state
        self.jam_history.append(jam_rate)
        self.def_history.append(jam_rate)
        if len(self.jam_history) > 5:
            self.jam_history.pop(0)
        if len(self.def_history) > 5:
            self.def_history.pop(0)

        # Joint state: (jam_rate_bin, defense_bin) — recent average
        recent_jam = np.mean(self.jam_history) if self.jam_history else 0
        next_state = _discretize(recent_jam) * N_STATE_BINS + _discretize(recent_jam)

        # Zero-sum reward
        r_jammer = jam_rate
        r_fhss = 1.0 - jam_rate

        return next_state, r_jammer, r_fhss


class CoevolutionTrainer:
    """Trains jammer and FHSS agents simultaneously."""

    def __init__(self, js_ratio_db: float = 30.0):
        self.js_ratio_db = js_ratio_db

    def train(self, n_episodes: int = 200, steps_per_episode: int = 100
              ) -> CoevolutionResult:
        """Train both agents simultaneously."""
        env = CoevolutionEnvironment(js_ratio_db=self.js_ratio_db)
        jammer = JammerAgent()
        fhss = FHSSAgent()

        jam_history = []
        fhss_history = []
        jam_action_counts = np.zeros(N_JAMMER_ACTIONS)
        def_action_counts = np.zeros(N_FHSS_ACTIONS)

        # Track strategy oscillations (cycle detection)
        recent_jam_actions = []
        cycles = 0

        for ep in range(n_episodes):
            state = env.reset()
            ep_jam = []
            ep_fhss = []

            for _ in range(steps_per_episode):
                jam_act = jammer.act(state)
                def_act = fhss.act(state)

                next_state, r_j, r_f = env.step(jam_act, def_act)

                jammer.learn(state, jam_act, r_j, next_state)
                fhss.learn(state, def_act, r_f, next_state)

                state = next_state
                ep_jam.append(r_j)
                ep_fhss.append(r_f)

                # Track action mix in latter half of training
                if ep > n_episodes // 2:
                    jam_action_counts[jam_act] += 1
                    def_action_counts[def_act] += 1
                    recent_jam_actions.append(jam_act)
                    if len(recent_jam_actions) > 100:
                        recent_jam_actions.pop(0)

            jammer.decay()
            fhss.decay()
            jam_history.append(float(np.mean(ep_jam)))
            fhss_history.append(float(np.mean(ep_fhss)))

        # Cycle detection: count switches between strategies in recent history
        if len(recent_jam_actions) > 10:
            cycles = sum(1 for i in range(1, len(recent_jam_actions))
                          if recent_jam_actions[i] != recent_jam_actions[i - 1])

        # Strategy mix
        if jam_action_counts.sum() > 0:
            jammer_mix = {JAMMER_ACTIONS[i].value: jam_action_counts[i] / jam_action_counts.sum()
                           for i in range(N_JAMMER_ACTIONS)}
        else:
            jammer_mix = {a.value: 1 / N_JAMMER_ACTIONS for a in JAMMER_ACTIONS}

        if def_action_counts.sum() > 0:
            fhss_mix = {FHSS_DEFENSES[i]: def_action_counts[i] / def_action_counts.sum()
                         for i in range(N_FHSS_ACTIONS)}
        else:
            fhss_mix = {d: 1 / N_FHSS_ACTIONS for d in FHSS_DEFENSES}

        # Nash distance: variance of jam_rate in last 50 episodes
        # Low variance = converged to equilibrium; high variance = still cycling
        last_50 = jam_history[-50:]
        nash_distance = float(np.std(last_50)) if len(last_50) > 0 else 1.0

        return CoevolutionResult(
            n_episodes=n_episodes,
            jammer_history=jam_history,
            fhss_history=fhss_history,
            final_jam_rate=float(np.mean(jam_history[-20:])) if len(jam_history) >= 20 else 0.0,
            final_survival_rate=float(np.mean(fhss_history[-20:])) if len(fhss_history) >= 20 else 0.0,
            jammer_strategy_mix=jammer_mix,
            fhss_strategy_mix=fhss_mix,
            nash_distance=nash_distance,
            cycles_detected=cycles,
        )


def run_coevolution_demo(n_episodes: int = 100) -> CoevolutionResult:
    """Run a default co-evolution training and return results."""
    trainer = CoevolutionTrainer(js_ratio_db=30.0)
    return trainer.train(n_episodes=n_episodes, steps_per_episode=50)


if __name__ == "__main__":
    print("=" * 60)
    print("AI-vs-AI Adversarial Co-evolution")
    print("=" * 60)
    result = run_coevolution_demo(n_episodes=150)

    print(f"\nFinal jam rate:        {result.final_jam_rate*100:.1f}%")
    print(f"Final survival rate:   {result.final_survival_rate*100:.1f}%")
    print(f"Nash distance (std):   {result.nash_distance:.4f}")
    print(f"Cycles detected:       {result.cycles_detected}/100")

    print(f"\nJammer strategy equilibrium mix:")
    for s, p in sorted(result.jammer_strategy_mix.items(), key=lambda x: -x[1]):
        bar = "#" * int(p * 30)
        print(f"  {s:<16} {p*100:.1f}%  {bar}")

    print(f"\nFHSS defender equilibrium mix:")
    for s, p in sorted(result.fhss_strategy_mix.items(), key=lambda x: -x[1]):
        bar = "#" * int(p * 30)
        print(f"  {s:<22} {p*100:.1f}%  {bar}")

    print(f"\nLearning curves (first 5, last 5 episodes):")
    print(f"  Jammer (early):  {[f'{x:.2f}' for x in result.jammer_history[:5]]}")
    print(f"  Jammer (late):   {[f'{x:.2f}' for x in result.jammer_history[-5:]]}")
    print(f"  FHSS (early):    {[f'{x:.2f}' for x in result.fhss_history[:5]]}")
    print(f"  FHSS (late):     {[f'{x:.2f}' for x in result.fhss_history[-5:]]}")
