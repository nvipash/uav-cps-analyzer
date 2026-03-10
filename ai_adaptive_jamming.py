#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: RL Adaptive Jamming Agent
Q-learning agent that dynamically selects jamming strategy based on FHSS behavior.

The agent observes the communication channel and switches between jamming strategies
(broadband, narrowband, sweep, follower) to maximize disruption against adaptive FHSS.

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

try:
    from fhss_emulator import (
        OcuSyncProtocol, JammingStrategy, JammingEffectivenessAnalyzer, ChannelSimulator
    )
except ImportError:
    import sys
    sys.path.insert(0, '.')
    from fhss_emulator import (
        OcuSyncProtocol, JammingStrategy, JammingEffectivenessAnalyzer, ChannelSimulator
    )

# Available actions (strategies the agent can switch between)
ACTIONS = [
    JammingStrategy.BROADBAND,
    JammingStrategy.NARROWBAND,
    JammingStrategy.SWEEP,
    JammingStrategy.FOLLOWER,
]
N_ACTIONS = len(ACTIONS)

# State discretization: (recent_jam_rate_bin, current_strategy_index)
# jam_rate quantized into 5 bins: [0-20%, 20-40%, 40-60%, 60-80%, 80-100%]
N_JAM_BINS = 5
N_STATES = N_JAM_BINS * N_ACTIONS


def _discretize_state(jam_rate: float, current_action_idx: int) -> int:
    """Convert continuous state to discrete state index."""
    jam_bin = min(N_JAM_BINS - 1, int(jam_rate * N_JAM_BINS))
    return jam_bin * N_ACTIONS + current_action_idx


@dataclass
class EpisodeResult:
    """Result of one RL episode."""
    total_reward: float
    mean_jam_rate: float
    strategy_switches: int
    strategy_history: List[str]
    jam_rate_history: List[float]
    n_steps: int


class QLearningAgent:
    """
    Tabular Q-learning agent for jamming strategy selection.

    State: (recent_jam_rate_bin, current_strategy)
    Action: switch to strategy {broadband, narrowband, sweep, follower}
    Reward: jam_rate in current time window
    """

    def __init__(self, learning_rate: float = 0.1, discount: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.05):
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((N_STATES, N_ACTIONS))

    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int):
        """Q-learning update rule."""
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class JammingEnvironment:
    """
    Gym-like environment for adaptive jamming against FHSS.
    Each step = one time window (e.g., 100ms of transmission).

    Adversarial mode: FHSS protocol adapts to jamming pattern by:
    - Adjusting hop sequence to avoid jammed channels
    - Increasing modulation order when channel quality is good
    - Adding randomization when broadband detected

    Multi-objective reward: jam_rate + power_efficiency - stealth_penalty
    """

    def __init__(self, js_ratio_db: float = 30.0, window_hops: int = 50,
                 adversarial: bool = False, multi_objective: bool = False):
        """
        Args:
            js_ratio_db: Fixed J/S ratio
            window_hops: Number of hops per evaluation window
            adversarial: FHSS adapts to jamming
            multi_objective: Reward balances jam_rate, efficiency, stealth
        """
        self.js_ratio_db = js_ratio_db
        self.window_hops = window_hops
        self.adversarial = adversarial
        self.multi_objective = multi_objective
        self.protocol = OcuSyncProtocol(version="3.0")
        self.analyzer = JammingEffectivenessAnalyzer(self.protocol)
        self.current_action_idx = 0
        self.recent_jam_rate = 0.0
        self.adaptive_factor = 1.0  # FHSS protocol's adaptation level
        self.jamming_history = []  # tracked for adversarial response

    def reset(self) -> int:
        """Reset environment and return initial state."""
        self.current_action_idx = 0
        self.recent_jam_rate = 0.0
        self.adaptive_factor = 1.0
        self.jamming_history = []
        return _discretize_state(0.0, 0)

    def _update_adversary(self, action_idx: int):
        """
        FHSS adversary adapts to jamming pattern.
        - Counts how often each strategy is used recently
        - Reduces effectiveness multiplicatively when same strategy repeats
        - Recovers when jammer switches strategies
        """
        self.jamming_history.append(action_idx)
        if len(self.jamming_history) > 10:
            self.jamming_history.pop(0)

        # Count recent uses of current strategy
        recent_count = self.jamming_history.count(action_idx)
        # Penalty: if same strategy used >5 times in last 10, FHSS adapts
        if recent_count >= 5:
            self.adaptive_factor = max(0.4, self.adaptive_factor - 0.1)
        else:
            # Recovery when jammer diversifies
            self.adaptive_factor = min(1.0, self.adaptive_factor + 0.05)

    def step(self, action_idx: int) -> Tuple[int, float, bool]:
        """Execute one step: apply strategy, observe result, return next state."""
        strategy = ACTIONS[action_idx]
        effectiveness = self.analyzer.calculate_effectiveness(
            strategy, jammer_bandwidth_mhz=83.5,
            tracking_delay_ms=0.5 if strategy == JammingStrategy.FOLLOWER else 0
        )

        # Adversarial adaptation
        if self.adversarial:
            self._update_adversary(action_idx)
            effectiveness *= self.adaptive_factor

        # Simulate hops in this window
        jammed = 0
        for _ in range(self.window_hops):
            if strategy == JammingStrategy.NARROWBAND:
                hit = np.random.random() < (1.0 / self.protocol.params.n_channels)
            else:
                hit = np.random.random() < effectiveness

            if hit and self.js_ratio_db > 0:
                jam_prob = min(1.0, self.js_ratio_db / 20.0)
                if np.random.random() < jam_prob:
                    jammed += 1

        jam_rate = jammed / self.window_hops
        self.recent_jam_rate = jam_rate
        prev_action = self.current_action_idx
        self.current_action_idx = action_idx

        # Reward calculation
        if self.multi_objective:
            # jam_rate + power_efficiency - stealth_penalty
            # Power efficiency: narrowband = high, broadband = low (uses more power per channel)
            power_efficiency = {
                0: 0.3,   # broadband - inefficient
                1: 0.95,  # narrowband - efficient
                2: 0.5,   # sweep
                3: 0.7,   # follower
            }.get(action_idx, 0.5)

            # Stealth: switching often = less detectable
            stealth_bonus = 0.05 if action_idx != prev_action else 0.0

            reward = (0.6 * jam_rate +
                       0.3 * power_efficiency * jam_rate +
                       0.1 * stealth_bonus)
        else:
            switch_penalty = 0.02 if action_idx != prev_action else 0
            reward = jam_rate - switch_penalty

        next_state = _discretize_state(jam_rate, action_idx)
        return next_state, reward, False


class AdaptiveJammingSimulator:
    """Trains RL agent and compares against fixed strategies."""

    def __init__(self, js_ratio_db: float = 30.0,
                 adversarial: bool = False, multi_objective: bool = False):
        self.js_ratio_db = js_ratio_db
        self.adversarial = adversarial
        self.multi_objective = multi_objective
        self.env = JammingEnvironment(js_ratio_db, adversarial=adversarial,
                                        multi_objective=multi_objective)

    def train_agent(self, n_episodes: int = 200, steps_per_episode: int = 100
                    ) -> Tuple[QLearningAgent, List[float]]:
        """
        Train RL agent over multiple episodes.

        Returns:
            (trained_agent, episode_rewards)
        """
        agent = QLearningAgent()
        episode_rewards = []

        for ep in range(n_episodes):
            state = self.env.reset()
            total_reward = 0

            for _ in range(steps_per_episode):
                action = agent.select_action(state)
                next_state, reward, _ = self.env.step(action)
                agent.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            agent.decay_epsilon()
            episode_rewards.append(total_reward)

        return agent, episode_rewards

    def evaluate_agent(self, agent: QLearningAgent,
                       n_episodes: int = 50, steps: int = 100) -> EpisodeResult:
        """Evaluate trained agent (no exploration)."""
        old_eps = agent.epsilon
        agent.epsilon = 0  # greedy

        total_rewards = []
        total_jam_rates = []

        for _ in range(n_episodes):
            state = self.env.reset()
            ep_reward = 0
            ep_jam_rates = []

            for _ in range(steps):
                action = agent.select_action(state)
                next_state, reward, _ = self.env.step(action)
                state = next_state
                ep_reward += reward
                ep_jam_rates.append(self.env.recent_jam_rate)

            total_rewards.append(ep_reward)
            total_jam_rates.append(np.mean(ep_jam_rates))

        agent.epsilon = old_eps

        return EpisodeResult(
            total_reward=float(np.mean(total_rewards)),
            mean_jam_rate=float(np.mean(total_jam_rates)),
            strategy_switches=0,
            strategy_history=[],
            jam_rate_history=[],
            n_steps=steps
        )

    def evaluate_fixed_strategy(self, strategy_idx: int,
                                 n_episodes: int = 50, steps: int = 100) -> EpisodeResult:
        """Evaluate a fixed (non-adaptive) strategy."""
        total_jam_rates = []

        for _ in range(n_episodes):
            self.env.reset()
            ep_jam_rates = []
            for _ in range(steps):
                _, _, _ = self.env.step(strategy_idx)
                ep_jam_rates.append(self.env.recent_jam_rate)
            total_jam_rates.append(np.mean(ep_jam_rates))

        return EpisodeResult(
            total_reward=0,
            mean_jam_rate=float(np.mean(total_jam_rates)),
            strategy_switches=0,
            strategy_history=[ACTIONS[strategy_idx].value],
            jam_rate_history=[],
            n_steps=steps
        )

    def run_comparison(self, n_train_episodes: int = 200) -> Dict:
        """
        Train RL agent and compare against all fixed strategies.

        Returns:
            Comparison dict with jam rates for each approach
        """
        # Train
        agent, learning_curve = self.train_agent(n_train_episodes)

        # Evaluate RL
        rl_result = self.evaluate_agent(agent)

        # Evaluate fixed strategies
        fixed_results = {}
        for i, strategy in enumerate(ACTIONS):
            fixed_results[strategy.value] = self.evaluate_fixed_strategy(i)

        # Best fixed strategy
        best_fixed_name = max(fixed_results, key=lambda k: fixed_results[k].mean_jam_rate)
        best_fixed_rate = fixed_results[best_fixed_name].mean_jam_rate

        improvement = ((rl_result.mean_jam_rate - best_fixed_rate) / best_fixed_rate * 100
                       if best_fixed_rate > 0 else 0)

        return {
            'rl_jam_rate': rl_result.mean_jam_rate,
            'fixed_results': {k: v.mean_jam_rate for k, v in fixed_results.items()},
            'best_fixed': best_fixed_name,
            'best_fixed_rate': best_fixed_rate,
            'rl_improvement': improvement,
            'learning_curve': learning_curve,
            'q_table': agent.q_table,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("RL Adaptive Jamming — Training & Comparison")
    print("=" * 60)

    sim = AdaptiveJammingSimulator(js_ratio_db=30.0)
    results = sim.run_comparison(n_train_episodes=200)

    print(f"\nFixed strategies:")
    for name, rate in results['fixed_results'].items():
        print(f"  {name:<16}: {rate*100:.1f}%")

    print(f"\nRL adaptive:       {results['rl_jam_rate']*100:.1f}%")
    print(f"Best fixed:        {results['best_fixed']} ({results['best_fixed_rate']*100:.1f}%)")
    print(f"RL improvement:    {results['rl_improvement']:+.1f}%")
