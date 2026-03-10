#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: FHSS Protocol Emulation Module
Emulates Frequency Hopping Spread Spectrum protocols including DJI OcuSync.

Based on:
- D. Torrieri, "Principles of Spread-Spectrum Communication Systems", 4th ed., 2018
- DJI OcuSync transmission system specifications

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

try:
    from propagation_models import BERModel, ModulationType
except ImportError:
    BERModel = None
    ModulationType = None


class JammingStrategy(Enum):
    """Types of jamming strategies against FHSS systems."""
    BROADBAND = "broadband"      # BBN - covers entire band
    NARROWBAND = "narrowband"    # Spot - single frequency
    SWEEP = "sweep"              # Sweeps across band
    FOLLOWER = "follower"        # Adaptive, tracks hops
    PROTOCOL_AWARE = "protocol"  # Exploits protocol knowledge


@dataclass
class FHSSParams:
    """Parameters for FHSS protocol."""
    n_channels: int = 40           # Number of frequency channels
    hop_rate_hz: float = 500.0     # Hopping rate in Hz
    channel_bandwidth_mhz: float = 2.0  # Bandwidth per channel
    band_start_mhz: float = 2400.0      # Band start frequency
    band_end_mhz: float = 2483.5        # Band end frequency
    dwell_time_ms: float = 2.0          # Time on each channel


class LFSR:
    """
    Linear Feedback Shift Register for pseudo-random sequence generation.
    Used for generating hop sequences in FHSS systems.
    """
    
    def __init__(self, seed: int = 0x1234, n_bits: int = 16, 
                 taps: List[int] = None):
        """
        Initialize LFSR.
        
        Args:
            seed: Initial state (non-zero)
            n_bits: Number of bits in register
            taps: Feedback tap positions (default: maximal length for n_bits)
        """
        self.n_bits = n_bits
        self.state = seed if seed != 0 else 1
        self.mask = (1 << n_bits) - 1
        
        # Default taps for maximal length sequences
        default_taps = {
            8: [8, 6, 5, 4],
            16: [16, 15, 13, 4],
            32: [32, 31, 29, 1]
        }
        self.taps = taps if taps else default_taps.get(n_bits, [n_bits, n_bits-1])
    
    def next(self) -> int:
        """Generate next value in sequence."""
        # Calculate feedback bit
        feedback = 0
        for tap in self.taps:
            feedback ^= (self.state >> (tap - 1)) & 1
        
        # Shift and insert feedback
        self.state = ((self.state << 1) | feedback) & self.mask
        
        return self.state
    
    def reset(self, seed: int = None):
        """Reset LFSR to initial or new seed."""
        if seed is not None:
            self.state = seed if seed != 0 else 1
        else:
            self.state = 0x1234


class FHSSProtocol:
    """
    Base class for FHSS protocol emulation.
    Implements generic frequency hopping behavior.
    """
    
    def __init__(self, params: FHSSParams = None):
        """
        Initialize FHSS protocol.
        
        Args:
            params: FHSS parameters (uses defaults if None)
        """
        self.params = params or FHSSParams()
        
        # Generate channel frequencies
        self.channels = np.linspace(
            self.params.band_start_mhz,
            self.params.band_end_mhz,
            self.params.n_channels
        )
        
        # Initialize LFSR for hop sequence
        self.lfsr = LFSR(seed=0x1234, n_bits=16)
        
        # Hop history for analysis
        self.hop_history: List[Tuple[float, float]] = []
    
    def get_channel_at_time(self, time_s: float) -> Tuple[int, float]:
        """
        Get channel index and frequency at given time.
        
        Args:
            time_s: Time in seconds
            
        Returns:
            Tuple of (channel_index, frequency_mhz)
        """
        # Calculate hop number
        hop_number = int(time_s * self.params.hop_rate_hz)
        
        # Reset LFSR and advance to current hop
        self.lfsr.reset()
        for _ in range(hop_number + 1):
            lfsr_value = self.lfsr.next()
        
        # Map LFSR output to channel
        channel_idx = lfsr_value % self.params.n_channels
        freq = self.channels[channel_idx]
        
        return channel_idx, freq
    
    def generate_hop_sequence(self, duration_s: float) -> List[Tuple[float, int, float]]:
        """
        Generate complete hop sequence for given duration.
        
        Args:
            duration_s: Duration in seconds
            
        Returns:
            List of (time, channel_index, frequency) tuples
        """
        sequence = []
        n_hops = int(duration_s * self.params.hop_rate_hz) + 1
        
        self.lfsr.reset()
        for i in range(n_hops):
            time = i / self.params.hop_rate_hz
            lfsr_value = self.lfsr.next()
            channel_idx = lfsr_value % self.params.n_channels
            freq = self.channels[channel_idx]
            sequence.append((time, channel_idx, freq))
        
        self.hop_history = sequence
        return sequence
    
    def get_channel_occupancy(self, duration_s: float = 1.0) -> np.ndarray:
        """
        Calculate channel occupancy statistics.
        
        Args:
            duration_s: Analysis duration in seconds
            
        Returns:
            Array of occupancy counts per channel
        """
        sequence = self.generate_hop_sequence(duration_s)
        occupancy = np.zeros(self.params.n_channels)
        
        for _, channel_idx, _ in sequence:
            occupancy[channel_idx] += 1
        
        return occupancy


class OcuSyncProtocol(FHSSProtocol):
    """
    Emulation of DJI OcuSync transmission protocol.
    
    OcuSync features:
    - 40 channels in 2.4 GHz band
    - ~500 Hz hopping rate
    - Adaptive modulation and coding (AMC)
    - Automatic frequency selection
    
    Note: This is a simplified emulation based on public specifications.
    Actual OcuSync protocol is proprietary.
    """
    
    def __init__(self, version: str = "3.0"):
        """
        Initialize OcuSync protocol emulation.
        
        Args:
            version: OcuSync version ('2.0', '3.0', '3+')
        """
        # OcuSync parameters based on version
        params_by_version = {
            "2.0": FHSSParams(n_channels=40, hop_rate_hz=400, 
                             channel_bandwidth_mhz=2.0),
            "3.0": FHSSParams(n_channels=40, hop_rate_hz=500,
                             channel_bandwidth_mhz=2.0),
            "3+": FHSSParams(n_channels=40, hop_rate_hz=500,
                            channel_bandwidth_mhz=2.5)
        }
        
        params = params_by_version.get(version, params_by_version["3.0"])
        super().__init__(params)
        
        self.version = version
        self.amc_enabled = True
        
        # Channel quality tracking (for adaptive features)
        self.channel_quality = np.ones(self.params.n_channels)
    
    def update_channel_quality(self, channel_idx: int, quality: float):
        """
        Update quality metric for a channel (simulates AMC feedback).
        
        Args:
            channel_idx: Channel index
            quality: Quality metric (0-1, where 1 is best)
        """
        self.channel_quality[channel_idx] = np.clip(quality, 0, 1)
    
    def get_channel_at_time(self, time_s: float) -> Tuple[int, float]:
        """
        Get channel considering AMC (avoids low-quality channels).
        """
        base_idx, base_freq = super().get_channel_at_time(time_s)
        
        # If AMC enabled and channel quality is low, try next channel
        if self.amc_enabled and self.channel_quality[base_idx] < 0.3:
            alt_idx = (base_idx + 1) % self.params.n_channels
            if self.channel_quality[alt_idx] > self.channel_quality[base_idx]:
                return alt_idx, self.channels[alt_idx]
        
        return base_idx, base_freq


class JammingEffectivenessAnalyzer:
    """
    Analyzes effectiveness of different jamming strategies against FHSS.
    """
    
    def __init__(self, protocol: FHSSProtocol):
        """
        Initialize analyzer with target protocol.
        
        Args:
            protocol: FHSS protocol to analyze
        """
        self.protocol = protocol
    
    def calculate_power_multiplier(self, strategy: JammingStrategy) -> float:
        """
        Calculate power multiplier needed for effective jamming.
        
        Args:
            strategy: Jamming strategy type
            
        Returns:
            Power multiplier relative to single-channel jamming
        """
        n_channels = self.protocol.params.n_channels
        
        multipliers = {
            JammingStrategy.BROADBAND: 1.0,       # Covers all, but spread thin
            JammingStrategy.NARROWBAND: float(n_channels),  # Must cover each channel
            JammingStrategy.SWEEP: n_channels / 8,  # Partial coverage
            JammingStrategy.FOLLOWER: 3.0,         # Tracking overhead
            JammingStrategy.PROTOCOL_AWARE: 2.0    # Protocol knowledge helps
        }
        
        return multipliers.get(strategy, 1.0)
    
    def calculate_effectiveness(self, strategy: JammingStrategy,
                                jammer_bandwidth_mhz: float,
                                tracking_delay_ms: float = 0.0) -> float:
        """
        Calculate jamming effectiveness (probability of successful jam).
        
        Args:
            strategy: Jamming strategy
            jammer_bandwidth_mhz: Jammer bandwidth in MHz
            tracking_delay_ms: Delay in tracking hops (for follower)
            
        Returns:
            Effectiveness as probability (0-1)
        """
        total_bandwidth = (self.protocol.params.band_end_mhz - 
                         self.protocol.params.band_start_mhz)
        channel_bw = self.protocol.params.channel_bandwidth_mhz
        n_channels = self.protocol.params.n_channels
        dwell_time = self.protocol.params.dwell_time_ms
        
        if strategy == JammingStrategy.BROADBAND:
            # Broadband jammer spreads power across entire band.
            # Against FHSS: power is diluted across all channels, so per-channel
            # power density = total_power / n_channels. Effectiveness is reduced
            # compared to static because the jammer cannot concentrate energy.
            if jammer_bandwidth_mhz >= total_bandwidth:
                # Full band coverage but power diluted across n_channels
                # Effective J/S per channel drops by 10*log10(n_channels) ~ 16 dB for 40ch
                # This makes broadband less effective than against static channel
                return 0.50  # 50% — covers all hops but each at reduced power
            else:
                coverage = jammer_bandwidth_mhz / total_bandwidth
                return coverage * 0.50
        
        elif strategy == JammingStrategy.NARROWBAND:
            # Only effective if happens to hit current channel
            return 1.0 / n_channels  # ~2.5% for 40 channels
        
        elif strategy == JammingStrategy.SWEEP:
            # Depends on sweep rate vs hop rate
            channels_covered = jammer_bandwidth_mhz / channel_bw
            sweep_effectiveness = channels_covered / n_channels
            return min(0.45, sweep_effectiveness)
        
        elif strategy == JammingStrategy.FOLLOWER:
            # Depends on tracking delay vs dwell time
            if tracking_delay_ms >= dwell_time:
                return 0.1  # Too slow to track
            delay_factor = 1 - (tracking_delay_ms / dwell_time)
            return 0.65 * delay_factor
        
        elif strategy == JammingStrategy.PROTOCOL_AWARE:
            # Can predict some hops but protocol is proprietary
            return 0.75  # Limited by closed protocol
        
        return 0.0
    
    def compare_strategies(self, jammer_bandwidth_mhz: float = 83.5) -> dict:
        """
        Compare all jamming strategies.
        
        Args:
            jammer_bandwidth_mhz: Available jammer bandwidth
            
        Returns:
            Dictionary with strategy comparisons
        """
        results = {}
        
        for strategy in JammingStrategy:
            multiplier = self.calculate_power_multiplier(strategy)
            effectiveness = self.calculate_effectiveness(
                strategy, jammer_bandwidth_mhz,
                tracking_delay_ms=0.5 if strategy == JammingStrategy.FOLLOWER else 0
            )
            
            results[strategy.value] = {
                'power_multiplier': multiplier,
                'effectiveness_fhss': effectiveness,
                'effectiveness_static': self._static_effectiveness(strategy)
            }
        
        return results
    
    def _static_effectiveness(self, strategy: JammingStrategy) -> float:
        """Effectiveness against static (non-hopping) channel."""
        static_eff = {
            JammingStrategy.BROADBAND: 0.65,   # Covers target freq but power diluted across band
            JammingStrategy.NARROWBAND: 0.95,   # Concentrated on known frequency
            JammingStrategy.SWEEP: 0.80,
            JammingStrategy.FOLLOWER: 0.98,
            JammingStrategy.PROTOCOL_AWARE: 0.85
        }
        return static_eff.get(strategy, 0.5)


class ChannelSimulator:
    """
    Simulates communication channel with FHSS under jamming.
    """
    
    def __init__(self, protocol: FHSSProtocol):
        """
        Initialize channel simulator.
        
        Args:
            protocol: FHSS protocol to simulate
        """
        self.protocol = protocol
        self.analyzer = JammingEffectivenessAnalyzer(protocol)
    
    def simulate_transmission(self, duration_s: float,
                             jamming_strategy: JammingStrategy,
                             js_ratio_db: float) -> dict:
        """
        Simulate transmission under jamming.
        
        Args:
            duration_s: Simulation duration
            jamming_strategy: Type of jamming
            js_ratio_db: Jamming to signal ratio in dB
            
        Returns:
            Simulation results dictionary
        """
        # Generate hop sequence
        sequence = self.protocol.generate_hop_sequence(duration_s)
        
        # Get jamming effectiveness
        effectiveness = self.analyzer.calculate_effectiveness(
            jamming_strategy,
            jammer_bandwidth_mhz=83.5,
            tracking_delay_ms=0.5
        )
        
        # Simulate each hop using BER-based soft degradation
        successful_hops = 0
        jammed_hops = 0

        for time, channel_idx, freq in sequence:
            # Determine if this hop is jammed
            if jamming_strategy == JammingStrategy.NARROWBAND:
                # Random single channel jammed
                jammed_channel = np.random.randint(0, self.protocol.params.n_channels)
                channel_hit = (channel_idx == jammed_channel)
            else:
                # Probabilistic jamming based on effectiveness
                channel_hit = np.random.random() < effectiveness

            if channel_hit and BERModel is not None:
                # Use BER/PER model for soft degradation
                jam_prob = BERModel.jamming_success_probability(js_ratio_db)
                is_jammed = np.random.random() < jam_prob
            elif channel_hit:
                # Fallback to hard threshold if BERModel not available
                is_jammed = js_ratio_db > 10.0
            else:
                is_jammed = False

            if is_jammed:
                jammed_hops += 1
            else:
                successful_hops += 1
        
        total_hops = len(sequence)
        
        return {
            'total_hops': total_hops,
            'successful_hops': successful_hops,
            'jammed_hops': jammed_hops,
            'success_rate': successful_hops / total_hops if total_hops > 0 else 0,
            'jam_rate': jammed_hops / total_hops if total_hops > 0 else 0,
            'jamming_strategy': jamming_strategy.value,
            'js_ratio_db': js_ratio_db
        }


if __name__ == "__main__":
    print("=" * 60)
    print("UAV-CPS-Analyzer: FHSS Emulator Test")
    print("=" * 60)
    
    # Create OcuSync protocol
    ocusync = OcuSyncProtocol(version="3.0")
    print(f"\n1. OcuSync 3.0 Parameters:")
    print(f"   Channels: {ocusync.params.n_channels}")
    print(f"   Hop rate: {ocusync.params.hop_rate_hz} Hz")
    print(f"   Band: {ocusync.params.band_start_mhz}-{ocusync.params.band_end_mhz} MHz")
    
    # Generate hop sequence
    print(f"\n2. Sample Hop Sequence (first 10 hops):")
    sequence = ocusync.generate_hop_sequence(0.02)  # 20ms = 10 hops
    for time, ch_idx, freq in sequence[:10]:
        print(f"   t={time*1000:.1f}ms: Channel {ch_idx:2d} ({freq:.1f} MHz)")
    
    # Channel occupancy
    print(f"\n3. Channel Occupancy (1 second):")
    occupancy = ocusync.get_channel_occupancy(1.0)
    print(f"   Min: {occupancy.min():.0f}, Max: {occupancy.max():.0f}, "
          f"Mean: {occupancy.mean():.1f}")
    
    # Jamming analysis
    print(f"\n4. Jamming Strategy Comparison:")
    analyzer = JammingEffectivenessAnalyzer(ocusync)
    results = analyzer.compare_strategies()
    
    print(f"   {'Strategy':<15} {'Static':<10} {'FHSS':<10} {'Power×':<10}")
    print(f"   {'-'*45}")
    for strategy, data in results.items():
        print(f"   {strategy:<15} {data['effectiveness_static']*100:>6.0f}%    "
              f"{data['effectiveness_fhss']*100:>6.0f}%    "
              f"×{data['power_multiplier']:.0f}")
    
    # Transmission simulation
    print(f"\n5. Transmission Simulation (J/S = 30 dB):")
    simulator = ChannelSimulator(ocusync)
    for strategy in JammingStrategy:
        result = simulator.simulate_transmission(1.0, strategy, 30.0)
        print(f"   {strategy.value:<15}: Success={result['success_rate']*100:.1f}%, "
              f"Jammed={result['jam_rate']*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("Tests completed successfully!")
