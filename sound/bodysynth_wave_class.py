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

MAX_ARMLEN = 2000  # in mm
MIN_ARMLEN = 1000  # in mm
MAX_FREQ = 320.0  # in Hz
MIN_FREQ = 80.0   # in Hz
AMPLITUDE_SCALE_FACTOR = 1.0


class PyoWavetableVoice:
    """Pyo-based wavetable oscillator with reverb"""
    def __init__(self, voice_id, table_data: np.ndarray, gain: float = 1.0, reverb_mix: float = 0.0):
        self.id = voice_id
        self.lock = threading.Lock()
        
        self.table = DataTable(size=len(table_data), init=table_data.tolist())
        self.freq_ctrl = Sig(SAMPLE_RATE / len(table_data))
        self.osc = Osc(table=self.table, freq=self.freq_ctrl, mul=gain)
        
        # Create reverb effect - use simpler approach to avoid segfault
        self.reverb_mix = Sig(reverb_mix)
        self.reverb = Freeverb(self.osc, size=0.8, damp=0.7, bal=self.reverb_mix)
        
        # Output the reverb
        self.reverb.out()
    
    def update_table(self, new_table: np.ndarray):
        with self.lock:
            # Create a new DataTable with the correct size
            old_table = self.table
            self.table = DataTable(size=len(new_table), init=new_table.tolist())
            
            # Update the oscillator to use the new table
            self.osc.setTable(self.table)
            
            # Update frequency based on new table length
            new_freq = SAMPLE_RATE / len(new_table)
            self.freq_ctrl.value = new_freq
    
    def update_gain(self, new_gain: float):
        with self.lock:
            self.osc.mul = new_gain
    
    def update_reverb(self, reverb_mix: float):
        with self.lock:
            # reverb_mix should be 0.0 (dry) to 1.0 (wet)
            self.reverb_mix.value = reverb_mix
    
    def stop(self):
        self.reverb.stop()
        self.osc.stop()


class WavetableSynth:
    """Wavetable synthesizer that generates tones from arm geometry"""
    
    def __init__(self, pyo_server, gain=0.3, max_voices=10):
        self.pyo_server = pyo_server
        self.gain = gain
        self.voices = {}
        self.voices_lock = threading.Lock()
        
        # Pre-create a pool of voices to avoid threading issues
        # Create dummy wavetable for initialization
        dummy_table = np.sin(np.linspace(0, 2*np.pi, 100)).astype(np.float32)
        self.voice_pool = {}
        for i in range(max_voices):
            voice_id = f"pool_{i}"
            self.voice_pool[voice_id] = PyoWavetableVoice(voice_id, dummy_table, gain=0.0, reverb_mix=0.0)
        
        self.available_voices = list(self.voice_pool.keys())
        self.person_to_voice = {}  # Map person_id to pool voice_id
        
        print(f"[WavetableSynth] Initialized with gain={gain}, pre-created {max_voices} voices")
    
    def best_fit_plane(self, points):
        """Fit a plane to 3D points using SVD"""
        centroid = points.mean(axis=0)
        uu, dd, vv = np.linalg.svd(points - centroid)
        normal = vv[2]
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        return centroid, normal
    
    def make_wavetable_mirror(self, arm_joints, sample_rate=SAMPLE_RATE):
        """Generate wavetable from arm geometry
        
        Args:
            arm_joints: List of (x, y, z) tuples representing arm joint positions
        
        Returns:
            tuple: (wavetable array, normal vector, frequency, angle)
        """
        points = np.array(arm_joints, dtype=float)
        
        raw_coords = points[:, 1]  # Y coordinates
        line_length = float(raw_coords.max() - raw_coords.min())
        
        x_extent = float(points[:, 0].max() - points[:, 0].min())
        z_extent = float(points[:, 2].max() - points[:, 2].min())
        horizontal_extent = np.sqrt(x_extent**2 + z_extent**2)
        
        clipped_x = np.clip(x_extent, MIN_ARMLEN, MAX_ARMLEN)
        norm_x = float(np.clip((clipped_x - MIN_ARMLEN) / (MAX_ARMLEN - MIN_ARMLEN + 1e-12), 0.0, 1.0))
        freq = MIN_FREQ + (1.0 - norm_x) * (MAX_FREQ - MIN_FREQ)
        
        table_len = max(3, int(round(sample_rate / freq)))
        
        coords = raw_coords - raw_coords[0]
        mirrored_coords = np.concatenate([coords, -coords[:0:-1]])
        
        centroid = points.mean(axis=0)
        uu, dd, vv = np.linalg.svd(points - centroid)
        normal = vv[2]
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        
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
        
        # Calculate angle for reverb
        first_joint = points[0]
        last_joint = points[-1]
        arm_direction_xz = np.array([last_joint[0] - first_joint[0], 0, last_joint[2] - first_joint[2]])
        
        if np.linalg.norm(arm_direction_xz) > 1e-6:
            arm_direction_xz = arm_direction_xz / np.linalg.norm(arm_direction_xz)
            angle = math.degrees(math.atan2(arm_direction_xz[2], arm_direction_xz[0]))
            angle = abs(angle)
        else:
            angle = 0.0
        
        return wavetable, normal, freq, angle
    
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
            
            table, normal, freq, angle = self.make_wavetable_mirror(arm_joints)
            reverb_amount = float(np.clip(angle / 180.0, 0.0, 1.0))
            
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
                'freq': freq,
                'angle': angle,
                'reverb': reverb_amount
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
    
    def stop(self):
        """Stop all voices"""
        with self.voices_lock:
            for voice in self.voice_pool.values():
                voice.stop()
            self.voice_pool.clear()
            self.person_to_voice.clear()
            self.available_voices.clear()
        print("[WavetableSynth] Stopped")
