"""
Drum sequencer as a reusable class
Class-based version of bodysynth_drum.py for modular integration
"""

import numpy as np
import time
import threading
from pyo import *

from senseSpaceLib.senseSpace.protocol import Frame
from senseSpaceLib.senseSpace.enums import UniversalJoint

# --- Configuration ---
SAMPLE_RATE = 44100
CHANNELS = 2
LOOP_LENGTH = 2.4
NUM_LANES = 1  # Set to 1 for kick only, 2 for kick+snare, 3 for kick+snare+cymbal

TOP_LEFT = [-3000, 0]   # x,z in mm
BOTTOM_RIGHT = [3000, -6000]  # x,z in mm
Y_MAX = 2500
Y_MIN = 500


def generate_click():
    """Generate a sharp click sound for metronome"""
    duration = 0.05
    sample_length = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, sample_length)
    
    freq = 1200  # Hz
    envelope = np.exp(-t / 0.01)
    
    click = np.sin(2 * np.pi * freq * t) * envelope * 1.0
    return click.astype(np.float32)


def generate_drum_variant(base_freq: float, duration: float, drum_type: str = 'kick', 
                         distortion: float = 0.0, tone: float = 0.5, decay_rate: float = 1.0):
    """Generate a drum sample with specified parameters
    
    Args:
        base_freq: Base frequency in Hz (kick: 40-80, snare: 150-250, cymbal: 3000-8000)
        duration: Duration in seconds (0.1-2.0)
        drum_type: 'kick', 'snare', or 'cymbal'
        distortion: Amount of distortion/saturation (0.0-1.0)
        tone: Tonal balance - lower = darker, higher = brighter (0.0-1.0)
        decay_rate: How fast the sound decays (0.5=slow, 1.0=normal, 2.0=fast)
    """
    sample_length = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, sample_length)
    
    if drum_type == 'kick':
        # Kick drum: pitch sweep + short body
        # Start at higher freq and sweep down
        freq_start = base_freq * 2.5
        freq_end = base_freq
        freq_sweep = np.linspace(freq_start, freq_end, sample_length)
        
        # Very short attack, fast decay
        attack_time = 0.005
        decay_time = 0.05
        
        envelope = np.zeros(sample_length)
        attack_samples = int(attack_time * SAMPLE_RATE)
        decay_samples = int(decay_time * SAMPLE_RATE)
        
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        if decay_samples > 0 and attack_samples + decay_samples < sample_length:
            envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, 0.3, decay_samples)
        
        # Exponential tail
        tail_start = attack_samples + decay_samples
        if tail_start < sample_length:
            tail_samples = sample_length - tail_start
            envelope[tail_start:] = 0.3 * np.exp(-np.linspace(0, 8 * decay_rate, tail_samples))
        
        # Generate swept sine wave
        phase = np.cumsum(2 * np.pi * freq_sweep / SAMPLE_RATE)
        fundamental = np.sin(phase)
        
        # Add some harmonics for punch
        harmonic2 = np.sin(phase * 2) * 0.2 * tone
        
        # Add click for attack
        click = np.exp(-t * 200) * np.random.randn(sample_length) * 0.3
        
        synth = (fundamental + harmonic2) * envelope + click * envelope
        
    elif drum_type == 'snare':
        # Snare: pitched tone + noise
        # Short attack, medium decay
        attack_time = 0.005
        decay_time = 0.08
        
        envelope = np.zeros(sample_length)
        attack_samples = int(attack_time * SAMPLE_RATE)
        decay_samples = int(decay_time * SAMPLE_RATE)
        
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        if decay_samples > 0 and attack_samples + decay_samples < sample_length:
            envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, 0.4, decay_samples)
        
        # Exponential tail
        tail_start = attack_samples + decay_samples
        if tail_start < sample_length:
            tail_samples = sample_length - tail_start
            envelope[tail_start:] = 0.4 * np.exp(-np.linspace(0, 6 * decay_rate, tail_samples))
        
        # Tonal component (fundamental + harmonics)
        fundamental = np.sin(2 * np.pi * base_freq * t)
        harmonic2 = np.sin(2 * np.pi * base_freq * 2 * t) * 0.3
        harmonic3 = np.sin(2 * np.pi * base_freq * 3.5 * t) * 0.2 * tone
        tonal = (fundamental + harmonic2 + harmonic3) * (1.0 - tone * 0.3)
        
        # Noise component (white noise for snare buzz)
        noise = np.random.randn(sample_length) * (0.6 + tone * 0.4)
        
        # Apply bandpass-like filtering to noise (rough approximation)
        # Emphasize mid-high frequencies
        noise_filtered = noise * (1 + np.sin(2 * np.pi * 5000 * t)) * 0.5
        
        synth = (tonal * 0.4 + noise_filtered * 0.6) * envelope
        
    elif drum_type == 'cymbal':
        # Cymbal: mostly noise with metallic character
        # Very short attack, long decay
        attack_time = 0.002
        decay_time = 0.05
        
        envelope = np.zeros(sample_length)
        attack_samples = int(attack_time * SAMPLE_RATE)
        decay_samples = int(decay_time * SAMPLE_RATE)
        
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        if decay_samples > 0 and attack_samples + decay_samples < sample_length:
            envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, 0.5, decay_samples)
        
        # Long exponential tail
        tail_start = attack_samples + decay_samples
        if tail_start < sample_length:
            tail_samples = sample_length - tail_start
            envelope[tail_start:] = 0.5 * np.exp(-np.linspace(0, 3 * decay_rate, tail_samples))
        
        # Multiple high-frequency sine waves (metallic partials)
        partials = np.zeros(sample_length)
        num_partials = 8
        for i in range(num_partials):
            freq_ratio = 1.0 + i * 1.5 + np.random.rand() * 0.3  # Inharmonic
            partial_freq = base_freq * freq_ratio
            amplitude = 1.0 / (i + 1)
            partials += np.sin(2 * np.pi * partial_freq * t) * amplitude
        
        # High-frequency noise
        noise = np.random.randn(sample_length)
        
        # Mix partials and noise based on tone parameter
        synth = (partials * tone * 0.3 + noise * (1.0 - tone * 0.5)) * envelope
        
    else:
        raise ValueError(f"Unknown drum_type: {drum_type}")
    
    # Normalize
    if np.max(np.abs(synth)) > 0:
        synth = synth / np.max(np.abs(synth))
    
    # Apply distortion/saturation if requested
    if distortion > 0:
        # Soft clipping with tanh
        drive = 1.0 + distortion * 4.0  # distortion 0-1 maps to drive 1-5
        synth = np.tanh(synth * drive) / np.tanh(drive)
    
    # RMS normalization - equalize energy per unit time
    # Calculate RMS only on the first 0.5s to avoid long tail affecting normalization
    rms_window = min(int(0.5 * SAMPLE_RATE), len(synth))
    rms = np.sqrt(np.mean(synth[:rms_window]**2))
    target_rms = 0.15  # Target RMS level
    if rms > 0:
        synth = synth * (target_rms / rms)
    
    # Final soft limiting to prevent clipping
    synth = np.tanh(synth * 1.2) * 0.9
    
    return synth.astype(np.float32)


class PyoSamplerVoice:
    """Pyo-based one-shot sample player using TrigEnv for reliable triggering"""
    def __init__(self, voice_id, buffer: np.ndarray, duration: float, gain: float = 1.0, server=None):
        self.id = voice_id
        self.lock = threading.Lock()
        self.buffer = buffer
        self.duration = duration
        
        self.table = DataTable(size=len(buffer), init=buffer.tolist())
        
        # Use Trig for manual triggering
        self.trig = Trig()
        
        # TrigEnv plays the sample once when triggered
        self.player = TrigEnv(self.trig, table=self.table, dur=duration, mul=gain)
        self.player.out()
    
    def trigger_sample(self, gain: float):
        """Trigger the sample to play once with the given gain"""
        with self.lock:
            self.player.mul = gain
            self.trig.play()
    
    def stop(self):
        self.player.stop()


class DrumSequencer:
    """Spatial drum sequencer that triggers samples based on head position"""
    
    def __init__(self, pyo_server, num_lanes=NUM_LANES, loop_length=LOOP_LENGTH):
        self.pyo_server = pyo_server
        self.num_lanes = num_lanes
        self.loop_length = loop_length
        
        
        self.loop_start_time = None
        self.last_loop_position = 0.0
        self.triggered_this_loop = set()

        self.debug_max_volume = True
        
        # Define drum parameters for all possible drum types
        all_drum_params = [
            {'base_freq': 60, 'duration': 0.5, 'drum_type': 'kick', 'distortion': 0.2, 'tone': 0.3, 'decay_rate': 1.5},
            {'base_freq': 200, 'duration': 0.3, 'drum_type': 'snare', 'distortion': 0.1, 'tone': 0.6, 'decay_rate': 1.2},
            {'base_freq': 5000, 'duration': 0.8, 'drum_type': 'cymbal', 'distortion': 0.0, 'tone': 0.7, 'decay_rate': 0.7}
        ]
        
        # Generate drum parameters for all lanes
        # Cycle through available drum types if we have more lanes than drum types
        self.drum_params = []
        for lane_idx in range(num_lanes):
            base_params = all_drum_params[lane_idx % len(all_drum_params)].copy()
            # If we're cycling beyond the base types, modify frequency slightly for variety
            if lane_idx >= len(all_drum_params):
                cycle = lane_idx // len(all_drum_params)
                base_params['base_freq'] *= (1.0 + cycle * 0.1)  # Increase pitch slightly
            self.drum_params.append(base_params)
        
        # Generate samples and create voices
        self.click_sample = generate_click()
        self.click_duration = len(self.click_sample) / SAMPLE_RATE
        self.click_voice = PyoSamplerVoice("click", self.click_sample, duration=self.click_duration, gain=1.0, server=pyo_server)
        
        self.sample_banks = {}
        self.sample_voices = {}
        
        for lane_idx in range(num_lanes):
            params = self.drum_params[lane_idx]
            self.sample_banks[lane_idx] = generate_drum_variant(**params)
            self.sample_voices[lane_idx] = PyoSamplerVoice(
                f"lane_{lane_idx}", 
                self.sample_banks[lane_idx],
                duration=params['duration'],
                gain=1.0, 
                server=pyo_server
            )
            print(f"[DrumSequencer] Lane {lane_idx}: {params['drum_type']}, freq={params['base_freq']}Hz, duration={params['duration']}s")
        
        if self.debug_max_volume:
            print("[DrumSequencer] DEBUG MODE: All samples will play at maximum volume")
        print(f"[DrumSequencer] Initialized with {num_lanes} lanes")
    
    def quantize_z(self, z, zmin, zmax):
        """Map Z position to a lane index"""
        width = (zmax - zmin) / self.num_lanes
        idx = np.floor((z - zmin) / width)
        return int(np.clip(idx, 0, self.num_lanes - 1))
    
    def place_sample(self, head_data):
        """Determine sample lane, timing, and volume from head position"""
        zmin = BOTTOM_RIGHT[1]
        zmax = TOP_LEFT[1]
        xmin = BOTTOM_RIGHT[0]
        xmax = TOP_LEFT[0]
        
        x = head_data['headpos'][0].x
        y = head_data['headpos'][0].y
        z = head_data['headpos'][0].z
        
        sample_idx = self.quantize_z(z, zmin, zmax)
        t_play = float(np.clip((x - xmin) / (xmax - xmin) * self.loop_length, 0.0, self.loop_length))
        
        # Map Y position (Y_MIN to Y_MAX) to volume (0.0 to 1.0)
        y_normalized = (y - Y_MIN) / (Y_MAX - Y_MIN)
        volume = float(np.clip(y_normalized, 0.0, 1.0))
        
        # Debug output
        lane_width = (zmax - zmin) / self.num_lanes
        
        return sample_idx, t_play, volume
    
    def update_num_lanes(self, new_num_lanes):
        """Dynamically update the number of lanes and regenerate samples"""
        if new_num_lanes == self.num_lanes:
            return  # No change needed
        
        print(f"[DrumSequencer] Updating lanes from {self.num_lanes} to {new_num_lanes}")
        
        # Stop old voices
        for voice in self.sample_voices.values():
            voice.stop()
        
        self.num_lanes = new_num_lanes
        self.sample_banks = {}
        self.sample_voices = {}
        
        # Regenerate drum parameters for new number of lanes
        all_drum_params = [
            {'base_freq': 60, 'duration': 0.5, 'drum_type': 'kick', 'distortion': 0.2, 'tone': 0.3, 'decay_rate': 1.5},
            {'base_freq': 200, 'duration': 0.3, 'drum_type': 'snare', 'distortion': 0.1, 'tone': 0.6, 'decay_rate': 1.2},
            {'base_freq': 5000, 'duration': 0.8, 'drum_type': 'cymbal', 'distortion': 0.0, 'tone': 0.7, 'decay_rate': 0.7}
        ]
        
        self.drum_params = []
        for lane_idx in range(new_num_lanes):
            base_params = all_drum_params[lane_idx % len(all_drum_params)].copy()
            if lane_idx >= len(all_drum_params):
                cycle = lane_idx // len(all_drum_params)
                base_params['base_freq'] *= (1.0 + cycle * 0.1)
            self.drum_params.append(base_params)
        
        # Regenerate samples and voices
        for lane_idx in range(new_num_lanes):
            params = self.drum_params[lane_idx]
            self.sample_banks[lane_idx] = generate_drum_variant(**params)
            self.sample_voices[lane_idx] = PyoSamplerVoice(
                f"lane_{lane_idx}", 
                self.sample_banks[lane_idx],
                duration=params['duration'],
                gain=1.0, 
                server=self.pyo_server
            )
            print(f"[DrumSequencer] Lane {lane_idx}: {params['drum_type']}, freq={params['base_freq']}Hz")
    
    def process_frame(self, head_positions):
        """Process frame data and trigger samples as needed
        
        Args:
            head_positions: List of dicts with 'p' (person_id) and 'headpos' (joint data)
        
        Returns:
            dict: Current loop position and trigger info
        """
        # Dynamically adjust number of lanes based on number of people
        num_people = len(head_positions)
        if num_people > 0 and num_people != self.num_lanes:
            self.update_num_lanes(num_people)
        
        if self.loop_start_time is None:
            self.loop_start_time = time.time()
        
        elapsed = time.time() - self.loop_start_time
        current_loop_position = (elapsed % self.loop_length) / self.loop_length
        current_loop_count = int(elapsed // self.loop_length)
        
        # Detect loop wrap
        new_loop = False
        if current_loop_position < self.last_loop_position:
            self.triggered_this_loop.clear()
            new_loop = True
            print(f"[DrumSequencer] NEW LOOP: {current_loop_count}")
            
            # Trigger click
            if self.click_voice:
                self.click_voice.trigger_sample(0.8)
        
        # Process each person
        for head in head_positions:
            sample_idx, t_play, volume = self.place_sample(head)
            person_id = head['p']
            person_loop_position = t_play / self.loop_length
            
            # Override volume if debug mode is enabled
            if self.debug_max_volume:
                volume = 1.0
            
            if person_id not in self.triggered_this_loop:
                if current_loop_position > person_loop_position:
                    drum_name = self.drum_params[sample_idx]['drum_type']
                    print(f"[DrumSequencer] TRIGGER: person {person_id}, lane {sample_idx} ({drum_name}), vol {volume:.2f}")
                    if sample_idx in self.sample_voices:
                        self.sample_voices[sample_idx].trigger_sample(volume)
                        self.triggered_this_loop.add(person_id)
        
        self.last_loop_position = current_loop_position
        
        return {
            'loop_position': current_loop_position,
            'new_loop': new_loop
        }
    
    def stop(self):
        """Stop all voices"""
        if self.click_voice:
            self.click_voice.stop()
        for voice in self.sample_voices.values():
            voice.stop()
        print("[DrumSequencer] Stopped")
