"""
Wavetable synthesizer as a reusable class
Class-based version of bodysynth_wave.py for modular integration
"""

import numpy as np
import math
import threading
import time
from pyo import *

from senseSpaceLib.senseSpace.protocol import Frame
from senseSpaceLib.senseSpace.enums import UniversalJoint

# --- Configuration ---
SAMPLE_RATE = 44100
CHANNELS = 2

MAX_ARMLEN = 2400  # in mm
MIN_ARMLEN = 450  # in mm
MAX_FREQ = 440.0  # in Hz
MIN_FREQ = 22.5   # in Hz

AMPLITUDE_SCALE_FACTOR = 1.0

MIN_REVERB_WET = 0.2  # Minimum reverb wetness (0.0-1.0)
REVERB_SIZE = 0.9
REVERB_DAMP = 0.9

DISTORTION_AMOUNT = 0.0  # Soft saturation amount (0.0-1.0)

LFO_AMOUNT = 0.5  # LFO amplitude modulation amount (0.0=off, 1.0=max influence)
LFO_SLOWDOWN = 200.0

# Musical scales (semitones from root note)
SCALES = {
    "Chromatic (None)": list(range(12)),  # All semitones
    "Major (Ionian)": [0, 2, 4, 5, 7, 9, 11],
    "Minor (Aeolian)": [0, 2, 3, 5, 7, 8, 10],
    "Dorian": [0, 2, 3, 5, 7, 9, 10],
    "Phrygian": [0, 1, 3, 5, 7, 8, 10],
    "Lydian": [0, 2, 4, 6, 7, 9, 11],
    "Mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "Pentatonic Major": [0, 2, 4, 7, 9],
    "Pentatonic Minor": [0, 3, 5, 7, 10],
    "Blues": [0, 3, 5, 6, 7, 10],
    "Whole Tone": [0, 2, 4, 6, 8, 10]
}


def quantize_frequency_to_scale(freq, scale_name="Chromatic (None)", root_freq=55.0):
    """Quantize frequency to nearest note in the given scale
    
    Args:
        freq: Input frequency in Hz
        scale_name: Name of the scale to quantize to
        root_freq: Root note frequency (default A1 = 55Hz)
    
    Returns:
        Quantized frequency in Hz
    """
    if scale_name not in SCALES or scale_name == "Chromatic (None)":
        return freq  # No quantization
    
    # Convert frequency to semitones from root
    semitones_from_root = 12 * math.log2(freq / root_freq)
    
    # Get scale intervals
    scale_intervals = SCALES[scale_name]
    
    # Find the octave and note within octave
    octave = int(semitones_from_root // 12)
    note_in_octave = semitones_from_root % 12
    
    # Find closest note in scale
    min_distance = float('inf')
    closest_interval = 0
    for interval in scale_intervals:
        distance = abs(note_in_octave - interval)
        if distance < min_distance:
            min_distance = distance
            closest_interval = interval
    
    # Reconstruct quantized frequency
    quantized_semitones = octave * 12 + closest_interval
    quantized_freq = root_freq * (2 ** (quantized_semitones / 12))
    
    return quantized_freq


class PyoWavetableVoice:
    """Pyo-based wavetable oscillator with reverb and LFO amplitude modulation"""
    def __init__(self, voice_id, table_data: np.ndarray, gain: float = 1.0, reverb_mix: float = 0.0, channels: int = 2, lfo_slowdown: float = LFO_SLOWDOWN, lfo_amount: float = LFO_AMOUNT):
        self.id = voice_id
        self.lock = threading.Lock()
        self.channels = channels
        self.lfo_slowdown = lfo_slowdown
        self.lfo_amount = lfo_amount
        
        # Audio wavetable
        self.table = DataTable(size=len(table_data), init=table_data.tolist())
        self.freq_ctrl = Sig(SAMPLE_RATE / len(table_data))
        
        # LFO wavetable (same shape as audio, but much slower frequency)
        self.lfo_table = DataTable(size=len(table_data), init=table_data.tolist())
        audio_freq = SAMPLE_RATE / len(table_data)
        lfo_freq = audio_freq / self.lfo_slowdown
        self.lfo_freq_ctrl = Sig(lfo_freq)
        
        # LFO oscillator - output range needs to be 0 to 1 for amplitude modulation
        # We'll scale and offset the LFO to be positive
        self.lfo_osc = Osc(table=self.lfo_table, freq=self.lfo_freq_ctrl, mul=0.5, add=0.5)
        
        # LFO amount control (0.0 = no LFO, 1.0 = full LFO modulation)
        self.lfo_amount_ctrl = Sig(self.lfo_amount)
        
        # Audio oscillator - modulated by LFO based on lfo_amount (0.0 to 1.0)
        # Interpolate between constant gain (lfo_amount=0) and full LFO modulation (lfo_amount=1)
        # Formula: gain * [(1 - lfo_amount) + (lfo_osc * lfo_amount)]
        # When lfo_amount=0: gain * 1.0 (constant)
        # When lfo_amount=1: gain * lfo_osc (full modulation from 0 to 1)
        lfo_modulation = (1.0 - self.lfo_amount_ctrl) + (self.lfo_osc * self.lfo_amount_ctrl)
        self.osc = Osc(table=self.table, freq=self.freq_ctrl, mul=lfo_modulation * gain)
        
        # Create reverb effect
        self.reverb_mix = Sig(reverb_mix)
        self.reverb = Freeverb(self.osc, size=REVERB_SIZE, damp=REVERB_DAMP, bal=self.reverb_mix)
        
        # Output the reverb to all channels
        # Use Pan to duplicate mono signal to multi-channel (pan=0.5 means center)
        self.stereo = Pan(self.reverb, outs=self.channels, pan=0.5)
        self.stereo.out()
        
        self.base_gain = gain
    
    def update_table(self, new_table: np.ndarray):
        with self.lock:
            # Create a new DataTable with the correct size for audio
            old_table = self.table
            self.table = DataTable(size=len(new_table), init=new_table.tolist())
            
            # Update the audio oscillator to use the new table
            self.osc.setTable(self.table)
            
            # Update frequency based on new table length
            new_freq = SAMPLE_RATE / len(new_table)
            self.freq_ctrl.value = new_freq
            
            # Also update LFO table with same waveform shape
            old_lfo_table = self.lfo_table
            self.lfo_table = DataTable(size=len(new_table), init=new_table.tolist())
            self.lfo_osc.setTable(self.lfo_table)
            
            # Update LFO frequency based on current slowdown setting
            lfo_freq = new_freq / self.lfo_slowdown
            self.lfo_freq_ctrl.value = lfo_freq
    
    def update_gain(self, gain):
        self.gain = gain
        # Update the oscillator multiplier - with LFO modulation based on current lfo_amount
        # Interpolate between constant gain (lfo_amount=0) and full LFO modulation (lfo_amount=1)
        lfo_modulation = (1.0 - self.lfo_amount_ctrl) + (self.lfo_osc * self.lfo_amount_ctrl)
        self.osc.mul = lfo_modulation * self.gain
    
    def update_reverb(self, reverb_mix: float):
        with self.lock:
            # reverb_mix should be 0.0 (dry) to 1.0 (wet)
            self.reverb_mix.value = reverb_mix
    
    def update_lfo_slowdown(self, lfo_slowdown: float):
        """Update LFO slowdown factor and recalculate LFO frequency"""
        with self.lock:
            self.lfo_slowdown = lfo_slowdown
            # Recalculate LFO frequency based on current audio frequency
            audio_freq = self.freq_ctrl.value
            lfo_freq = audio_freq / self.lfo_slowdown
            self.lfo_freq_ctrl.value = lfo_freq
    
    def update_lfo_amount(self, lfo_amount: float):
        """Update LFO modulation amount (0.0 = no LFO, 1.0 = full modulation)"""
        with self.lock:
            self.lfo_amount = lfo_amount
            self.lfo_amount_ctrl.value = lfo_amount
    
    def stop(self):
        self.stereo.stop()
        self.reverb.stop()
        self.osc.stop()
        self.lfo_osc.stop()


class WavetableSynth:
    """Wavetable synthesizer that generates tones from arm geometry"""
    
    def __init__(self, pyo_server, gain=0.3, max_voices=10, sample_rate=SAMPLE_RATE, channels=2, root_change_callback=None, shared_state=None):
        self.pyo_server = pyo_server
        self.gain = gain
        self.sample_rate = sample_rate
        self.channels = channels
        self.voices = {}
        self.voices_lock = threading.Lock()
        self.shared_state = shared_state  # Optional shared state from modular system
        self.current_scale = "Dorian"  # Default scale (can be overridden by shared state)
        
        # Callback for root frequency changes
        self.root_change_callback = root_change_callback
        
        # Adjustable parameters
        self.smoothing_amount = 0.0  # 0.0 = no smoothing (order 12), 1.0 = max smoothing (order 4)
        self.distortion_amount = DISTORTION_AMOUNT
        self.lfo_amount = LFO_AMOUNT
        self.lfo_slowdown = LFO_SLOWDOWN
        self.min_reverb_wet = MIN_REVERB_WET
        self.max_armlen = MAX_ARMLEN
        self.min_armlen = MIN_ARMLEN
        # Use octave offsets from root instead of fixed frequencies
        self.octave_range_low = -2.0   # octaves below root
        self.octave_range_high = 2.0   # octaves above root
        
        # Fallback root frequency (220 Hz = A) if no shared state
        # When shared_state is provided, it will be used instead
        self.root_freq = 220.0
        
        # Pre-create a pool of voices to avoid threading issues
        # Create dummy wavetable for initialization
        dummy_table = np.sin(np.linspace(0, 2*np.pi, 100)).astype(np.float32)
        self.voice_pool = {}
        for i in range(max_voices):
            voice_id = f"pool_{i}"
            self.voice_pool[voice_id] = PyoWavetableVoice(voice_id, dummy_table, gain=0.0, reverb_mix=0.0, channels=self.channels, lfo_slowdown=self.lfo_slowdown, lfo_amount=self.lfo_amount)
        
        self.available_voices = list(self.voice_pool.keys())
        self.person_to_voice = {}  # Map person_id to pool voice_id
        
        print(f"[WavetableSynth] Initialized with gain={gain}, pre-created {max_voices} voices")
    
    def best_fit_line_xz(self, points):
        """Fit a line to points in the XZ plane (ignoring Y coordinates)
        
        Returns:
            tuple: (centroid, angle, direction_xz)
                - centroid: (x, y, z) 3D centroid of all points
                - angle: 0-90 degrees, angle of fitted line from X-axis
                         0° = line along X (min reverb), 90° = line along Z (max reverb)
                - direction_xz: (x, z) normalized direction of the fitted line in XZ plane
        """
        # Project points onto XZ plane
        points_xz = points[:, [0, 2]]  # Take only X and Z coordinates
        centroid = points.mean(axis=0)  # Full 3D centroid
        centroid_xz = points_xz.mean(axis=0)
        
        # Fit line in XZ plane using SVD
        uu, dd, vv = np.linalg.svd(points_xz - centroid_xz)
        direction = vv[0]  # First component is the line direction (x, z)
        
        # Calculate angle of the line from X-axis in XZ plane
        # atan2(z_component, x_component) gives angle from positive X-axis
        angle_rad = math.atan2(abs(direction[1]), abs(direction[0]))  # Use abs to get 0-90° range
        angle = math.degrees(angle_rad)
        
        # Now angle is 0° when line is along X, 90° when line is along Z
        
        return centroid, angle, direction
    
    def make_wavetable_mirror(self, arm_joints):
        """Generate wavetable from arm geometry
        
        Args:
            arm_joints: List of (x, y, z) tuples representing arm joint positions
        
        Returns:
            tuple: (wavetable array, line_info dict, frequency, angle, horizontal_extent)
        """
        points = np.array(arm_joints, dtype=float)
        
        raw_coords = points[:, 1]  # Y coordinates
        line_length = float(raw_coords.max() - raw_coords.min())
        
        x_extent = float(points[:, 0].max() - points[:, 0].min())
        z_extent = float(points[:, 2].max() - points[:, 2].min())
        horizontal_extent = np.sqrt(x_extent**2 + z_extent**2)
        
        # Use horizontal_extent (combined X-Z spread) for frequency calculation
        clipped_extent = np.clip(horizontal_extent, self.min_armlen, self.max_armlen)
        norm_extent = float(np.clip((clipped_extent - self.min_armlen) / (self.max_armlen - self.min_armlen + 1e-12), 0.0, 1.0))
        
        # Map normalized extent to octave offset from root
        # norm_extent=0 (min arm length) -> octave_range_high octaves above root
        # norm_extent=1 (max arm length) -> octave_range_low octaves below root
        octave_offset = self.octave_range_high + (self.octave_range_low - self.octave_range_high) * norm_extent
        
        # Get current root frequency and scale from shared state if available
        current_root_freq = self.shared_state.get_root_freq() if self.shared_state else self.root_freq
        current_scale = self.shared_state.get_scale() if self.shared_state else self.current_scale
        
        # Calculate frequency as root * 2^(octave_offset)
        freq = current_root_freq * (2.0 ** octave_offset)

        # Quantize frequency to selected scale using current root frequency
        quantized_freq = quantize_frequency_to_scale(freq, current_scale, current_root_freq)

        # Use quantized frequency for table length calculation
        table_len = max(3, int(round(self.sample_rate / quantized_freq)))
        
        coords = raw_coords - raw_coords[0]
        mirrored_coords = np.concatenate([coords, -coords[:0:-1]])
        
        # Fit line in XZ plane only (top-down view)
        centroid, angle, direction_xz = self.best_fit_line_xz(points)
        
        if np.ptp(mirrored_coords) > 1e-8 and horizontal_extent > 1e-8:
            scale_factor = (line_length / (horizontal_extent + 1e-8)) * AMPLITUDE_SCALE_FACTOR
            scale_factor = np.clip(scale_factor, 0.0, 1.0)
            mirrored_coords = mirrored_coords - (mirrored_coords.max() + mirrored_coords.min()) / 2
            if np.max(np.abs(mirrored_coords)) > 0:
                mirrored_coords = mirrored_coords / np.max(np.abs(mirrored_coords)) * scale_factor
        else:
            mirrored_coords = np.zeros_like(mirrored_coords)
        
        idx = np.linspace(0, len(mirrored_coords) - 1, table_len)
        wavetable = np.interp(idx, np.arange(len(mirrored_coords)), mirrored_coords).astype(np.float32)
        
        # Apply polynomial smoothing if enabled (before distortion for cleaner smoothing)
        # Higher smoothing_amount = lower polynomial order = smoother curve
        # At 0.0: no smoothing applied (uses raw interpolated wavetable)
        if self.smoothing_amount > 0.0 and len(wavetable) > 2:
            # Map smoothing 0.0-1.0 to polynomial order (not applied at 0.0)
            # At 0.0: no smoothing (skipped)
            # At 0.5: order 8 (moderate smoothing)
            # At 1.0: order 4 (maximum smoothing)
            poly_order = int(12 - self.smoothing_amount * 8)
            poly_order = min(poly_order, len(wavetable) - 1)  # Can't exceed data points
            poly_order = max(poly_order, 4)  # Minimum order 4
            
            if poly_order < len(wavetable):
                x = np.arange(len(wavetable))
                coeffs = np.polyfit(x, wavetable, poly_order)
                wavetable = np.polyval(coeffs, x).astype(np.float32)
        
        # Apply soft distortion/saturation using tanh
        if self.distortion_amount > 0:
            drive = 1.0 + self.distortion_amount * 3.0  # Map 0-1 to 1-4
            wavetable = np.tanh(wavetable * drive) / np.tanh(drive)
        
        # Angle is calculated from best_fit_line_xz (0-90° range)
        # 0° = line along X-axis (horizontal) → min reverb
        # 90° = line along Z-axis (perpendicular) → max reverb
        
        line_info = {
            'centroid': centroid,
            'direction': direction_xz,
            'angle': angle
        }
        
        return wavetable, line_info, freq, angle, horizontal_extent
    
    def process_frame(self, armline_data):
        """Process frame data and update wavetable voices
        
        Args:
            armline_data: List of dicts with 'p' (person_id) and 'armlines' (list of joint tuples)
        
        Returns:
            dict: Per-person wavetable info {person_id: {'wavetable': array, 'freq': float, 'angle': float, 'reverb': float}}
        """
        active_person_ids = set()
        result = {}
        
        for arm_data in armline_data:
            person_id = arm_data['p']
            active_person_ids.add(person_id)
            
            # Convert armline joints to list of (x,y,z) tuples
            arm_joints = arm_data['armlines']
            
            table, line_info, freq, angle, horizontal_extent = self.make_wavetable_mirror(arm_joints)
            
            # freq is already quantized in make_wavetable_mirror
            
                        # Map angle (0-90°) to reverb range: min_reverb_wet to 1.0
            # 0° (along X) = minimum reverb
            # 90° (along Z) = maximum reverb
            angle_normalized = angle / 90.0
            reverb_amount = self.min_reverb_wet + angle_normalized * (1.0 - self.min_reverb_wet)
            
            # Debug output
            #print(f"[WavetableSynth] Person {person_id}: horizontal_extent={horizontal_extent:.0f}mm, freq={freq:.1f}Hz, angle={angle:.1f}°, reverb={reverb_amount:.2f}")
            
            # Manage voices using pre-created pool
            with self.voices_lock:
                if person_id not in self.person_to_voice:
                    # Assign a voice from the pool
                    if self.available_voices:
                        voice_id = self.available_voices.pop(0)
                        self.person_to_voice[person_id] = voice_id
                        # Activate the voice
                        self.voice_pool[voice_id].update_table(table)
                        self.voice_pool[voice_id].update_gain(self.gain)
                        self.voice_pool[voice_id].update_reverb(reverb_amount)
                else:
                    # Update existing voice
                    voice_id = self.person_to_voice[person_id]
                    self.voice_pool[voice_id].update_table(table)
                    self.voice_pool[voice_id].update_reverb(reverb_amount)
            
            result[person_id] = {
                'wavetable': table,
                'freq': freq,  # Already quantized in make_wavetable_mirror
                'angle': angle,
                'reverb': reverb_amount,
                'line_info': line_info
            }
        
        # Remove inactive voices - return them to pool
        with self.voices_lock:
            inactive_ids = set(self.person_to_voice.keys()) - active_person_ids
            for p_id in inactive_ids:
                if p_id in self.person_to_voice:
                    voice_id = self.person_to_voice[p_id]
                    # Mute the voice
                    self.voice_pool[voice_id].update_gain(0.0)
                    # Return to available pool
                    self.available_voices.append(voice_id)
                    del self.person_to_voice[p_id]
        
        return result
    
    def set_scale(self, scale_name):
        """Set the musical scale for frequency quantization"""
        if scale_name in SCALES:
            self.current_scale = scale_name
            print(f"[WavetableSynth] Scale changed to: {scale_name}")
        else:
            print(f"[WavetableSynth] Unknown scale: {scale_name}")
    
    def set_gain(self, gain: float):
        """Set overall gain/volume for the wavetable synth"""
        self.gain = gain
        # Update all active voices
        with self.voices_lock:
            for person_id, voice_id in self.person_to_voice.items():
                if voice_id in self.voice_pool:
                    self.voice_pool[voice_id].update_gain(gain)
        print(f"[WavetableSynth] Master gain set to {gain:.2f}")
    
    def set_smoothing(self, smoothing: float):
        """Set wavetable smoothing amount (0.0 = no smoothing/order 12, 1.0 = max smoothing/order 4)"""
        self.smoothing_amount = smoothing
        poly_order = int(12 - smoothing * 8) if smoothing > 0 else 12
        print(f"[WavetableSynth] Smoothing set to {smoothing:.2f} (poly order {poly_order})")
    
    def set_distortion(self, distortion: float):
        """Set distortion amount (0.0-1.0)"""
        self.distortion_amount = distortion
        print(f"[WavetableSynth] Distortion set to {distortion:.2f}")
    
    def set_lfo_amount(self, lfo_amount: float):
        """Set LFO modulation amount (0.0-1.0)"""
        self.lfo_amount = lfo_amount
        # Update all active voices
        with self.voices_lock:
            for person_id, voice_id in self.person_to_voice.items():
                if voice_id in self.voice_pool:
                    self.voice_pool[voice_id].update_lfo_amount(lfo_amount)
        print(f"[WavetableSynth] LFO amount set to {lfo_amount:.2f}")
    
    def set_lfo_slowdown(self, lfo_slowdown: float):
        """Set LFO slowdown factor (higher = slower LFO)"""
        self.lfo_slowdown = lfo_slowdown
        # Update all active voices
        with self.voices_lock:
            for person_id, voice_id in self.person_to_voice.items():
                if voice_id in self.voice_pool:
                    self.voice_pool[voice_id].update_lfo_slowdown(lfo_slowdown)
        print(f"[WavetableSynth] LFO slowdown set to {lfo_slowdown:.1f}")
    
    def set_min_reverb(self, min_reverb: float):
        """Set minimum reverb wetness (0.0-1.0)"""
        self.min_reverb_wet = min_reverb
        print(f"[WavetableSynth] Min reverb set to {min_reverb:.2f}")
    
    def set_max_armlen(self, max_armlen: float):
        """Set maximum arm length in mm"""
        self.max_armlen = max_armlen
        print(f"[WavetableSynth] Max arm length set to {max_armlen:.0f} mm")
    
    def set_min_armlen(self, min_armlen: float):
        """Set minimum arm length in mm"""
        self.min_armlen = min_armlen
        print(f"[WavetableSynth] Min arm length set to {min_armlen:.0f} mm")
    
    def set_octave_range_high(self, octaves: float):
        """Set octave range above root (for short arms)"""
        self.octave_range_high = octaves
        print(f"[WavetableSynth] High octave range set to {octaves:+.1f} octaves from root")
    
    def set_octave_range_low(self, octaves: float):
        """Set octave range below root (for long arms)"""
        self.octave_range_low = octaves
        print(f"[WavetableSynth] Low octave range set to {octaves:+.1f} octaves from root")
    
    def stop(self):
        """Stop all voices"""
        with self.voices_lock:
            for voice in self.voice_pool.values():
                voice.stop()
            self.voice_pool.clear()
            self.person_to_voice.clear()
            self.available_voices.clear()
        print("[WavetableSynth] Stopped")
