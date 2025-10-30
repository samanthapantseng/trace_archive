"""
Wavetable synthesizer as a reusable class
Class-based version of bodysynth_wave.py for modular integration
"""

import numpy as np
import math
import threading
from pyo import *

from senseSpaceLib.senseSpace.protocol import Frame
from senseSpaceLib.senseSpace.enums import UniversalJoint

# --- Configuration ---
SAMPLE_RATE = 44100
CHANNELS = 2

MAX_ARMLEN = 1200  # in mm
MIN_ARMLEN = 450  # in mm
MAX_FREQ = 320.0  # in Hz
MIN_FREQ = 80.0   # in Hz

AMPLITUDE_SCALE_FACTOR = 1.0

MIN_REVERB_WET = 0.5  # Minimum reverb wetness (0.0-1.0)
DISTORTION_AMOUNT = 0.3  # Soft saturation amount (0.0-1.0)
LFO_ENABLED = False  # Enable/disable LFO amplitude modulation

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
    def __init__(self, voice_id, table_data: np.ndarray, gain: float = 1.0, reverb_mix: float = 0.0, channels: int = 2):
        self.id = voice_id
        self.lock = threading.Lock()
        self.channels = channels
        
        # Audio wavetable
        self.table = DataTable(size=len(table_data), init=table_data.tolist())
        self.freq_ctrl = Sig(SAMPLE_RATE / len(table_data))
        
        # LFO wavetable (same shape as audio, but much slower frequency)
        self.lfo_table = DataTable(size=len(table_data), init=table_data.tolist())
        audio_freq = SAMPLE_RATE / len(table_data)
        lfo_freq = audio_freq / 100.0  # 100 times slower
        
        # LFO oscillator - output range needs to be 0 to 1 for amplitude modulation
        # We'll scale and offset the LFO to be positive
        self.lfo_osc = Osc(table=self.lfo_table, freq=lfo_freq, mul=0.5, add=0.5)
        
        # Audio oscillator - modulated by LFO if enabled, otherwise constant gain
        if LFO_ENABLED:
            self.osc = Osc(table=self.table, freq=self.freq_ctrl, mul=self.lfo_osc * gain)
        else:
            self.osc = Osc(table=self.table, freq=self.freq_ctrl, mul=gain)
        
        # Create reverb effect
        self.reverb_mix = Sig(reverb_mix)
        self.reverb = Freeverb(self.osc, size=0.99, damp=0.99, bal=self.reverb_mix)
        
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
            
            # Update LFO frequency (1000x slower than audio)
            lfo_freq = new_freq / 1000.0
            self.lfo_osc.setFreq(lfo_freq)
    
    def update_gain(self, gain):
        self.gain = gain
        # Update the oscillator multiplier - with LFO modulation if enabled
        if LFO_ENABLED:
            self.osc.mul = self.lfo_osc * self.gain
        else:
            self.osc.mul = self.gain
    
    def update_reverb(self, reverb_mix: float):
        with self.lock:
            # reverb_mix should be 0.0 (dry) to 1.0 (wet)
            self.reverb_mix.value = reverb_mix
    
    def stop(self):
        self.stereo.stop()
        self.reverb.stop()
        self.osc.stop()
        self.lfo_osc.stop()


class WavetableSynth:
    """Wavetable synthesizer that generates tones from arm geometry"""
    
    def __init__(self, pyo_server, gain=0.3, max_voices=10, sample_rate=SAMPLE_RATE, channels=2):
        self.pyo_server = pyo_server
        self.gain = gain
        self.sample_rate = sample_rate
        self.channels = channels
        self.voices = {}
        self.voices_lock = threading.Lock()
        self.current_scale = "Chromatic (None)"  # Default scale
        
        # Pre-create a pool of voices to avoid threading issues
        # Create dummy wavetable for initialization
        dummy_table = np.sin(np.linspace(0, 2*np.pi, 100)).astype(np.float32)
        self.voice_pool = {}
        for i in range(max_voices):
            voice_id = f"pool_{i}"
            self.voice_pool[voice_id] = PyoWavetableVoice(voice_id, dummy_table, gain=0.0, reverb_mix=0.0, channels=self.channels)
        
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
        clipped_extent = np.clip(horizontal_extent, MIN_ARMLEN, MAX_ARMLEN)
        norm_extent = float(np.clip((clipped_extent - MIN_ARMLEN) / (MAX_ARMLEN - MIN_ARMLEN + 1e-12), 0.0, 1.0))
        freq = MIN_FREQ + (1.0 - norm_extent) * (MAX_FREQ - MIN_FREQ)
        
        # Quantize frequency to selected scale
        quantized_freq = quantize_frequency_to_scale(freq, self.current_scale)
        
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
        
        # Apply soft distortion/saturation using tanh
        if DISTORTION_AMOUNT > 0:
            drive = 1.0 + DISTORTION_AMOUNT * 3.0  # Map 0-1 to 1-4
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
            
            # Map angle (0-90°) to reverb range: MIN_REVERB_WET to 1.0
            # 0° = line along X-axis (horizontal) → min reverb
            # 90° = line along Z-axis (perpendicular to X) → max reverb
            angle_normalized = float(np.clip(angle / 90.0, 0.0, 1.0))
            reverb_amount = MIN_REVERB_WET + angle_normalized * (1.0 - MIN_REVERB_WET)
            
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
    
    def stop(self):
        """Stop all voices"""
        with self.voices_lock:
            for voice in self.voice_pool.values():
                voice.stop()
            self.voice_pool.clear()
            self.person_to_voice.clear()
            self.available_voices.clear()
        print("[WavetableSynth] Stopped")
