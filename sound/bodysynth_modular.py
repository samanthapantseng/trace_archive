"""
Modular bodysynth system - combines drum sequencer, wavetable synth, and visualization
Components can be independently enabled/disabled via command-line arguments
"""

import numpy as np
import time
import threading
import signal
import random
from pyo import *
import argparse
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

# Setup paths
from senseSpaceLib.senseSpace import setup_paths
setup_paths()

from senseSpaceLib.senseSpace import MinimalClient, Frame
from senseSpaceLib.senseSpace.protocol import Frame
from senseSpaceLib.senseSpace.enums import UniversalJoint

# Import bodysynth modules
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bodysynth_sequencer import DrumSequencer, LOOP_LENGTH, NUM_LANES, TOP_LEFT, BOTTOM_RIGHT
from bodysynth_wave import WavetableSynth
from bodysynth_osc_sampler import OSCSampler
from bodysynth_gui import SynthMonitor

# --- Choose ONE LLM Singer implementation (uncomment the one you want) ---
# All use the same class name "LLMSinger" and same interface, so you can switch by uncommenting the desired line:
from bodysynth_llm_singer import LLMSinger              # Local: Ollama + Piper (pre-generated, male voice, FREE)
#from bodysynth_llm_singer_bark import LLMSinger         # Local: Ollama + Bark (pre-generated, expressive AI voices, FREE, SLOW first run)



# --- Configuration ---
AUDIO_DEVICE = 10
CHANNELS = 2

MIN_CONFIDENCE = 70  # Minimum confidence to consider a person detected
CACHE_TIME = 1.0   # Time in seconds to keep cached person data

# GUI data
synth_voice_data = {}
data_lock = threading.Lock()


def extract_head_and_arm(frame):
    """Extract head positions and armline joints from frame"""
    body_model = frame.body_model if frame.body_model else "BODY_34"
    
    curr_armlines = []
    curr_headpos = []
    for person in frame.people:
        if person.confidence > MIN_CONFIDENCE:
            headpos = {'p': person.id, 'headpos': person.get_joint(UniversalJoint.NOSE, body_model)}
            curr_headpos.append(headpos)
            
            # Extract armline joints
            armline_joints_raw = [
                person.get_joint(UniversalJoint.LEFT_WRIST, body_model),
                person.get_joint(UniversalJoint.LEFT_ELBOW, body_model),
                person.get_joint(UniversalJoint.LEFT_SHOULDER, body_model),
                person.get_joint(UniversalJoint.LEFT_CLAVICLE, body_model),
                person.get_joint(UniversalJoint.NECK, body_model),
                person.get_joint(UniversalJoint.RIGHT_CLAVICLE, body_model),
                person.get_joint(UniversalJoint.RIGHT_SHOULDER, body_model),
                person.get_joint(UniversalJoint.RIGHT_ELBOW, body_model),
                person.get_joint(UniversalJoint.RIGHT_WRIST, body_model)
            ]
            
            # Convert to list of (x,y,z) tuples for easier processing
            armline_tuples = []
            for joint_list in armline_joints_raw:
                if joint_list and len(joint_list) > 0:
                    armline_tuples.append((joint_list[0].x, joint_list[0].y, joint_list[0].z))
            
            armline = {'p': person.id, 'armlines': armline_tuples, 'armlines_raw': armline_joints_raw}
            curr_armlines.append(armline)
    
    snap = {'headpos': curr_headpos, 'armlines': curr_armlines}
    return snap


class SharedMusicalState:
    """Shared musical state across all components (root frequency, scale, loop count)"""
    
    # Scale definitions (semitones from root note)
    SCALES = {
        "Chromatic (None)": list(range(12)),
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
    
    # Note names for chromatic scale
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    def __init__(self, root_change_callback=None):
        # Loop tracking
        self.loop_count = 0
        self.root_change_interval = 4  # Change root every 4 loops
        
        # Base frequency for C3
        self.base_c_freq = 130.81  # C3
        
        # Musical scale
        self.current_scale = "Dorian"
        
        # Generate initial root notes based on scale
        self._update_available_roots()
        
        # Start with a random root from the scale
        self.current_root_index = random.randint(0, len(self.root_frequencies) - 1)
        self.root_freq = self.root_frequencies[self.current_root_index]
        
        # Time-based fallback for root changes (if no drum sequencer)
        self.last_root_change_time = time.time()
        self.time_based_interval = 15.0  # 15 seconds between root changes
        
        # Callback for GUI updates
        self.root_change_callback = root_change_callback
        
        print(f"[SharedState] Initialized with root {self.root_note_names[self.current_root_index]} ({self.root_freq:.1f} Hz)")
    
    def update_loop_count(self, new_loop_count):
        """Update loop count and check if root should change"""
        if new_loop_count > 0 and new_loop_count != self.loop_count:
            self.loop_count = new_loop_count
            
            # Check if we've crossed a root change boundary
            if self.loop_count % self.root_change_interval == 0:
                self._change_root()
    
    def check_time_based_root_change(self):
        """Check if enough time has passed for a root change (fallback if no drums)"""
        current_time = time.time()
        time_since_last = current_time - self.last_root_change_time
        
        if time_since_last >= self.time_based_interval:
            self._change_root(time_based=True)
    
    def _update_available_roots(self):
        """Generate available root frequencies based on current scale"""
        scale_intervals = self.SCALES.get(self.current_scale, self.SCALES["Dorian"])
        
        # Generate root frequencies for each note in the scale
        self.root_frequencies = []
        self.root_note_names = []
        
        for interval in scale_intervals:
            # Calculate frequency: base_freq * 2^(semitones/12)
            freq = self.base_c_freq * (2 ** (interval / 12))
            self.root_frequencies.append(freq)
            self.root_note_names.append(self.NOTE_NAMES[interval])
        
        print(f"[SharedState] Available roots for {self.current_scale}: {', '.join(self.root_note_names)}")
    
    def _change_root(self, time_based=False):
        """Change to a random root frequency (different from current)"""
        if len(self.root_frequencies) <= 1:
            return  # Can't change if only one root available
        
        # Choose a random index different from current
        available_indices = [i for i in range(len(self.root_frequencies)) if i != self.current_root_index]
        self.current_root_index = random.choice(available_indices)
        
        self.root_freq = self.root_frequencies[self.current_root_index]
        note_name = self.root_note_names[self.current_root_index]
        
        change_type = "TIME-BASED" if time_based else f"LOOP {self.loop_count}"
        print(f"[SharedState] *** ROOT CHANGE ({change_type}): {note_name} ({self.root_freq:.1f} Hz) ***")
        
        # Update timestamp for time-based tracking
        self.last_root_change_time = time.time()
        
        # Notify GUI
        if self.root_change_callback:
            self.root_change_callback(note_name, self.root_freq)
    
    def set_scale(self, scale_name):
        """Update musical scale and regenerate available roots"""
        self.current_scale = scale_name
        old_root_freq = self.root_freq
        
        # Regenerate available roots based on new scale
        self._update_available_roots()
        
        # Try to keep a similar root frequency, or pick the closest one
        closest_index = 0
        min_diff = float('inf')
        for i, freq in enumerate(self.root_frequencies):
            diff = abs(freq - old_root_freq)
            if diff < min_diff:
                min_diff = diff
                closest_index = i
        
        self.current_root_index = closest_index
        self.root_freq = self.root_frequencies[self.current_root_index]
        
        print(f"[SharedState] Scale changed to: {scale_name}, root: {self.root_note_names[self.current_root_index]}")
    
    def get_root_freq(self):
        """Get current root frequency"""
        return self.root_freq
    
    def get_scale(self):
        """Get current scale"""
        return self.current_scale


class BodysynthClient:
    """Client that processes frames and dispatches to enabled components"""
    
    def __init__(self, drum_enabled=True, wave_enabled=True, gui_enabled=True, llm_enabled=True, osc_enabled=True, shared_state=None):
        self.drum_enabled = drum_enabled
        self.wave_enabled = wave_enabled
        self.gui_enabled = gui_enabled
        self.llm_enabled = llm_enabled
        self.osc_enabled = osc_enabled
        
        self.drum_sequencer = None
        self.wavetable_synth = None
        self.llm_singer = None
        self.osc_sampler = None
        self.shared_state = shared_state
        
        # Flag to control frame processing
        self.ready_to_process = False
        
        # Person caching: track last seen time and data for each person
        self.person_last_seen = {}  # {person_id: timestamp}
        self.person_cache = {}  # {person_id: {'headpos': ..., 'armline': ...}}
        self.cache_time = CACHE_TIME  # Cache for 5 seconds
        
        # Track known people to detect new entries
        self.known_people = set()  # Set of person IDs we've seen before
        
        print(f"[BodysynthClient] Initialized - Drum: {drum_enabled}, Wave: {wave_enabled}, GUI: {gui_enabled}, LLM: {llm_enabled}, OSC: {osc_enabled}")
    
    def set_drum_sequencer(self, drum_sequencer):
        """Set the drum sequencer instance"""
        self.drum_sequencer = drum_sequencer
    
    def set_wavetable_synth(self, wavetable_synth):
        """Set the wavetable synth instance"""
        self.wavetable_synth = wavetable_synth
    
    def set_llm_singer(self, llm_singer):
        """Set the LLM singer instance"""
        self.llm_singer = llm_singer
    
    def set_osc_sampler(self, osc_sampler):
        """Set the OSC sampler instance"""
        self.osc_sampler = osc_sampler
    
    def on_init(self):
        print(f"[INIT] Connected to server")
    
    def enable_processing(self):
        """Enable frame processing - called after LLM pre-generation is complete"""
        self.ready_to_process = True
        print("[BodysynthClient] Frame processing ENABLED")
    
    def on_frame(self, frame: Frame):
        """Process incoming frame and dispatch to enabled components"""
        # Don't process frames until LLM pre-generation is complete
        if self.llm_enabled and not self.ready_to_process:
            return
        
        snap = extract_head_and_arm(frame)
        
        current_time = time.time()
        
        # Update last seen time and cache data for all currently detected people
        active_ids = set([arm['p'] for arm in snap['armlines']])
        
        # Check for new people entering the frame
        new_people = active_ids - self.known_people
        if new_people and self.shared_state:
            for person_id in new_people:
                print(f"[BodysynthClient] New person detected: {person_id}")
                self.shared_state._change_root()  # Trigger root change for new person
        
        # Update known people set
        self.known_people.update(active_ids)
        
        for person_id in active_ids:
            self.person_last_seen[person_id] = current_time
            
            # Cache the person's data
            headpos_data = next((h for h in snap['headpos'] if h['p'] == person_id), None)
            armline_data = next((a for a in snap['armlines'] if a['p'] == person_id), None)
            
            self.person_cache[person_id] = {
                'headpos': headpos_data,
                'armline': armline_data
            }
        
        # Determine which people to keep (currently active OR within cache window)
        kept_person_ids = set()
        cached_headpos = []
        cached_armlines = []
        
        for person_id, last_seen in list(self.person_last_seen.items()):
            time_since_seen = current_time - last_seen
            if time_since_seen <= self.cache_time:
                kept_person_ids.add(person_id)
                
                # Use cached data if person is not currently active
                if person_id not in active_ids and person_id in self.person_cache:
                    #print(f"[BodysynthClient] Using CACHED data for person {person_id} (not seen for {time_since_seen:.2f}s)")
                    cache = self.person_cache[person_id]
                    if cache['headpos']:
                        cached_headpos.append(cache['headpos'])
                    if cache['armline']:
                        cached_armlines.append(cache['armline'])
            else:
                # Remove from tracking and cache after cache period
                print(f"[BodysynthClient] REMOVING person {person_id} from cache (not seen for {time_since_seen:.2f}s)")
                del self.person_last_seen[person_id]
                if person_id in self.person_cache:
                    del self.person_cache[person_id]
                # Also remove from known_people so they can trigger root change if they come back
                self.known_people.discard(person_id)
        
        # Combine live and cached data
        snap['headpos'] = [h for h in snap['headpos'] if h['p'] in kept_person_ids] + cached_headpos
        snap['armlines'] = [a for a in snap['armlines'] if a['p'] in kept_person_ids] + cached_armlines
        
        drum_info = None
        wave_info = {}
        
        # Process drum sequencer if enabled
        if self.drum_enabled and self.drum_sequencer:
            drum_info = self.drum_sequencer.process_frame(snap['headpos'])
            
            # Update shared state with loop count
            if drum_info and 'loop_count' in drum_info and self.shared_state:
                self.shared_state.update_loop_count(drum_info['loop_count'])
        
        # Check time-based root change if no drums (shared state fallback)
        if not self.drum_enabled and self.shared_state:
            self.shared_state.check_time_based_root_change()
        
        # Process wavetable synth if enabled
        if self.wave_enabled and self.wavetable_synth:
            wave_info = self.wavetable_synth.process_frame(snap['armlines'])
        
        # Process LLM singer (pass frame data for analysis)
        if self.llm_singer:
            frame_data = {
                'people': snap['headpos'],
                'num_people': len(snap['headpos'])
            }
            self.llm_singer.process_frame(frame_data)
        
        # Update GUI data if enabled
        if self.gui_enabled:
            with data_lock:
                for arm in snap['armlines']:
                    person_id = arm['p']
                    
                    # Extract full skeleton
                    skeleton_joints = []
                    for person in frame.people:
                        if person.id == person_id and person.skeleton:
                            skeleton_joints = [(joint.pos.x, joint.pos.y, joint.pos.z) for joint in person.skeleton]
                            break
                    
                    # Extract head position
                    head_pos = None
                    for head in snap['headpos']:
                        if head['p'] == person_id:
                            nose_joint = head['headpos']
                            if nose_joint and len(nose_joint) > 0:
                                head_pos = (nose_joint[0].x, nose_joint[0].y, nose_joint[0].z)
                            break
                    
                    # Build GUI data
                    gui_data = {
                        'wavetable': np.array([]),
                        'freq': 0,
                        'angle': 0,
                        'reverb': 0,
                        'skeleton': skeleton_joints,
                        'head_pos': head_pos,
                        'armline': arm['armlines']
                    }
                    
                    # Add wavetable info if available
                    if person_id in wave_info:
                        gui_data['wavetable'] = wave_info[person_id]['wavetable']
                        gui_data['freq'] = wave_info[person_id]['freq']
                        gui_data['angle'] = wave_info[person_id]['angle']
                        gui_data['reverb'] = wave_info[person_id]['reverb']
                        gui_data['line_info'] = wave_info[person_id]['line_info']
                    
                    synth_voice_data[person_id] = gui_data
                
                # Clean up inactive persons (those not in kept_person_ids)
                inactive_ids = set(synth_voice_data.keys()) - kept_person_ids
                for p_id in inactive_ids:
                    del synth_voice_data[p_id]
    
    def on_connection_changed(self, connected: bool):
        status = "Connected" if connected else "Disconnected"
        print(f"[CONNECTION] {status}")


def main():
    parser = argparse.ArgumentParser(description="Modular Bodysynth System")
    parser.add_argument("--server", "-s", default="192.168.1.4", help="Server IP")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    parser.add_argument("--rec", type=str, default=None, help="Playback mode: path to .ssrec recording file")
    
    # Component enable/disable flags
    parser.add_argument("--drum", action="store_true", help="Enable drum sequencer")
    parser.add_argument("--wave", action="store_true", help="Enable wavetable synthesis")
    parser.add_argument("--llm", action="store_true", help="Enable LLM Singer")
    parser.add_argument("--osc", action="store_true", help="Enable OSC Sampler")
    parser.add_argument("--gui", action="store_true", help="Enable GUI visualization")
    parser.add_argument("--all", action="store_true", help="Enable all components (drum + wave + llm + osc + gui)")
    
    args = parser.parse_args()
    
    # Determine which components to enable
    drum_enabled = args.drum or args.all
    wave_enabled = args.wave or args.all
    llm_enabled = args.llm or args.all
    osc_enabled = args.osc or args.all
    gui_enabled = args.gui or args.all

    
    # If nothing specified, enable all by default
    if not (args.drum or args.wave or args.llm or args.osc or args.gui or args.all):
        print("[INFO] No components specified, enabling all by default")
        drum_enabled = wave_enabled = llm_enabled = osc_enabled = gui_enabled = True
    
    print(f"[INFO] Enabled components: Drum={drum_enabled}, Wave={wave_enabled}, LLM={llm_enabled}, OSC={osc_enabled}, GUI={gui_enabled}")
    
    # Setup GUI/QApplication FIRST before any pyo initialization
    app = None
    if gui_enabled:
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        print("[INFO] QApplication created")
    
    # Initialize pyo audio server if any audio component is enabled
    pyo_server = None
    actual_sample_rate = 44100
    if drum_enabled or wave_enabled or llm_enabled or osc_enabled:
        # Query the audio device to get its sample rate using pyo
        device_index = AUDIO_DEVICE  # Your audio device
        devices_info = pa_get_devices_infos()
        for key, value in devices_info[1].items():
            print(f"{key}: {value}")
        device_info = devices_info[1][device_index]
        actual_sample_rate = int(device_info['default sr'])
        print(f"[INFO] Audio device {device_index} sample rate: {actual_sample_rate} Hz")
            
        s = Server(sr=actual_sample_rate, nchnls=CHANNELS, duplex=0)
        s.setOutputDevice(AUDIO_DEVICE)
        s.boot().start()
        pyo_server = s
        print(f"[INFO] Pyo audio server started on output device {AUDIO_DEVICE} at {actual_sample_rate} Hz")
    
    # Initialize components in main thread (critical for pyo)
    monitor = None
    drum_sequencer = None
    wavetable_synth = None
    llm_singer = None
    osc_sampler = None
    
    # Define root change callback before creating shared state
    def on_root_changed(note_name, frequency):
        """Callback for when root frequency changes"""
        if monitor:
            monitor.update_root_note_display(note_name, frequency)
    
    # Create shared musical state (root, scale, loop count)
    shared_state = SharedMusicalState(root_change_callback=on_root_changed)
    
    if drum_enabled:
        drum_sequencer = DrumSequencer(pyo_server, num_lanes=NUM_LANES, loop_length=LOOP_LENGTH, channels=CHANNELS)
    
    if wave_enabled:
        wavetable_synth = WavetableSynth(pyo_server, gain=0.3, sample_rate=actual_sample_rate, channels=CHANNELS, shared_state=shared_state)
    
    # Initialize OSC Sampler if enabled
    if osc_enabled:
            osc_sampler = OSCSampler(
            osc_ip="0.0.0.0",
            osc_port=8000,
            pyo_server=pyo_server,
            gain=0.8
            )
            print(f"[INFO] OSC Sampler initialized on port 8000")
    
    # LLM Singer will be initialized after GUI
    llm_singer = None
    
    # Create client with shared state
    bodysynth_client = BodysynthClient(drum_enabled=drum_enabled, 
                                       wave_enabled=wave_enabled, 
                                       gui_enabled=gui_enabled,
                                       llm_enabled=llm_enabled,
                                       osc_enabled=osc_enabled,
                                       shared_state=shared_state)
    
    if drum_sequencer:
        bodysynth_client.set_drum_sequencer(drum_sequencer)
    if wavetable_synth:
        bodysynth_client.set_wavetable_synth(wavetable_synth)
    if osc_sampler:
        bodysynth_client.set_osc_sampler(osc_sampler)
    
    # Create and run senseSpace client in background thread
    client = MinimalClient(
        server_ip=args.server,
        server_port=args.port,
        playback_file=args.rec,
        on_init=bodysynth_client.on_init,
        on_frame=bodysynth_client.on_frame,
        on_connection_changed=bodysynth_client.on_connection_changed
    )
    
    def run_client():
        client.run()
    
    client_thread = threading.Thread(target=run_client, daemon=True)
    client_thread.start()
    print("[INFO] Client thread started")
    
    if gui_enabled:
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Create monitor with drum sequencer parameters if drum is enabled
        if drum_enabled:
            monitor = SynthMonitor(num_voices=5, num_lanes=NUM_LANES, 
                                  top_left=TOP_LEFT, bottom_right=BOTTOM_RIGHT)
        else:
            monitor = SynthMonitor(num_voices=5)
        
        # Set initial root note display from shared state
        if wave_enabled and wavetable_synth:
            root_freq = shared_state.get_root_freq()
            # Get note name from shared state's current index
            note_name = shared_state.root_note_names[shared_state.current_root_index]
            monitor.update_root_note_display(note_name, root_freq)
        
        monitor.show()
        
        # Initialize LLM Singer with lyrics callback to GUI (if enabled)
        if llm_enabled:
            def on_lyrics_generated(lyrics):
                """Callback when lyrics are generated"""
                monitor.llm_lyrics_update.emit(lyrics)
            
            print("[INFO] Initializing LLM Singer (this may take a moment)...")
            llm_singer = LLMSinger(pyo_server, channels=CHANNELS, on_lyrics_callback=on_lyrics_generated)
            llm_singer.start()  # This will block until pre-generation is complete
            bodysynth_client.set_llm_singer(llm_singer)
            
            if llm_singer.is_ready():
                print(f"[INFO] LLM Singer ready with {len(llm_singer.pregenerated_phrases)} phrases!")
                # Enable frame processing now that LLM is ready
                bodysynth_client.enable_processing()
            else:
                print("[WARNING] LLM Singer not ready - check Ollama and Piper")
                # Enable processing anyway so other components work
                bodysynth_client.enable_processing()
        else:
            print("[INFO] LLM Singer disabled")
            # Enable processing immediately if no LLM
            bodysynth_client.enable_processing()
        
        # Track loop start time and length for GUI
        gui_loop_start_time = time.time()
        current_loop_length = LOOP_LENGTH
        
        def update_gui():
            nonlocal current_loop_length
            with data_lock:
                monitor.update_signal.emit(dict(synth_voice_data))
            
            # Update loop position - use drum sequencer time if available, else use our own
            if drum_enabled and drum_sequencer and drum_sequencer.loop_start_time:
                elapsed = time.time() - drum_sequencer.loop_start_time
            else:
                elapsed = time.time() - gui_loop_start_time
            
            loop_position = (elapsed % current_loop_length) / current_loop_length
            monitor.head_widget.update_loop_position(loop_position)
            
            # Check for LLM audio to play (main thread context)
            if llm_singer:
                llm_singer.check_audio_queue()
        
        def on_loop_length_changed(new_length):
            nonlocal current_loop_length
            current_loop_length = new_length
            if drum_enabled and drum_sequencer:
                drum_sequencer.loop_length = new_length
                drum_sequencer.loop_start_time = time.time()
                drum_sequencer.triggered_this_loop.clear()
        
        def on_scale_changed(scale_name):
            """Handle scale changes from GUI"""
            if wave_enabled and wavetable_synth:
                wavetable_synth.set_scale(scale_name)
        
        def on_click_changed(enabled):
            """Handle click enable/disable from GUI"""
            if drum_enabled and drum_sequencer:
                drum_sequencer.set_click_enabled(enabled)
        
        def on_drum_gain_changed(gain):
            """Handle drum gain changes from GUI"""
            if drum_enabled and drum_sequencer:
                drum_sequencer.set_gain(gain)
        
        def on_wave_gain_changed(gain):
            """Handle wave gain changes from GUI"""
            if wave_enabled and wavetable_synth:
                wavetable_synth.set_gain(gain)
        
        def on_llm_prompt_changed(prompt):
            """Handle LLM prompt changes from GUI"""
            if llm_singer:
                llm_singer.set_base_prompt(prompt)
        
        def on_llm_interval_changed(interval):
            """Handle LLM interval changes from GUI"""
            if llm_singer:
                llm_singer.set_generation_interval(interval)
        
        def on_llm_autotune_changed(amount):
            """Handle LLM auto-tune changes from GUI"""
            if llm_singer:
                llm_singer.set_autotune_amount(amount)
        
        def on_llm_reverb_changed(amount):
            """Handle LLM reverb changes from GUI"""
            if llm_singer:
                llm_singer.set_reverb_amount(amount)
        
        def on_llm_transpose_changed(octaves):
            """Handle LLM transpose changes from GUI"""
            if llm_singer:
                llm_singer.set_transpose(octaves)
        
        def on_llm_gain_changed(gain):
            """Handle LLM gain changes from GUI"""
            if llm_singer:
                llm_singer.set_gain(gain)
        
        def on_drum_distortion_changed(distortion):
            """Handle drum distortion changes from GUI"""
            if drum_enabled and drum_sequencer:
                drum_sequencer.set_distortion(distortion)
        
        def on_drum_compression_changed(compression):
            """Handle drum compression changes from GUI"""
            if drum_enabled and drum_sequencer:
                drum_sequencer.set_compression(compression)
        
        def on_wave_distortion_changed(distortion):
            """Handle wave distortion changes from GUI"""
            if wave_enabled and wavetable_synth:
                wavetable_synth.set_distortion(distortion)
        
        def on_wave_smoothing_changed(smoothing):
            """Handle wave smoothing changes from GUI"""
            if wave_enabled and wavetable_synth:
                wavetable_synth.set_smoothing(smoothing)
        
        def on_wave_lfo_changed(lfo_amount):
            """Handle wave LFO changes from GUI"""
            if wave_enabled and wavetable_synth:
                wavetable_synth.set_lfo_amount(lfo_amount)
        
        def on_wave_lfo_slowdown_changed(lfo_slowdown):
            """Handle wave LFO slowdown changes from GUI"""
            if wave_enabled and wavetable_synth:
                wavetable_synth.set_lfo_slowdown(lfo_slowdown)
        
        def on_wave_min_reverb_changed(min_reverb):
            """Handle wave min reverb changes from GUI"""
            if wave_enabled and wavetable_synth:
                wavetable_synth.set_min_reverb(min_reverb)
        
        def on_wave_max_armlen_changed(max_armlen):
            """Handle wave max arm length changes from GUI"""
            if wave_enabled and wavetable_synth:
                wavetable_synth.set_max_armlen(max_armlen)
        
        def on_wave_min_armlen_changed(min_armlen):
            """Handle wave min arm length changes from GUI"""
            if wave_enabled and wavetable_synth:
                wavetable_synth.set_min_armlen(min_armlen)
        
        def on_wave_octave_high_changed(octave_high):
            """Handle wave high octave range changes from GUI"""
            if wave_enabled and wavetable_synth:
                wavetable_synth.set_octave_range_high(octave_high)
        
        def on_wave_octave_low_changed(octave_low):
            """Handle wave low octave range changes from GUI"""
            if wave_enabled and wavetable_synth:
                wavetable_synth.set_octave_range_low(octave_low)
        
        # Sync scale changes to shared state, wave, and LLM
        def on_scale_changed_full(scale_name):
            # Update shared state
            shared_state.set_scale(scale_name)
            # Update individual components (they can read from shared state or get direct updates)
            if wave_enabled and wavetable_synth:
                wavetable_synth.set_scale(scale_name)
            if llm_singer:
                llm_singer.set_scale(scale_name)
        
        # Save settings on change
        def on_settings_changed():
            from bodysynth_settings import SettingsManager
            settings_mgr = SettingsManager()
            settings = monitor.get_all_settings()
            settings_mgr.save(settings)
        
        # Connect GUI signals
        monitor.loop_length_changed.connect(on_loop_length_changed)
        monitor.scale_changed.connect(on_scale_changed)
        monitor.click_enabled_changed.connect(on_click_changed)
        monitor.drum_gain_changed.connect(on_drum_gain_changed)
        monitor.wave_gain_changed.connect(on_wave_gain_changed)
        monitor.drum_distortion_changed.connect(on_drum_distortion_changed)
        monitor.drum_compression_changed.connect(on_drum_compression_changed)
        monitor.wave_distortion_changed.connect(on_wave_distortion_changed)
        monitor.wave_smoothing_changed.connect(on_wave_smoothing_changed)
        monitor.wave_lfo_changed.connect(on_wave_lfo_changed)
        monitor.wave_lfo_slowdown_changed.connect(on_wave_lfo_slowdown_changed)
        monitor.wave_min_reverb_changed.connect(on_wave_min_reverb_changed)
        monitor.wave_max_armlen_changed.connect(on_wave_max_armlen_changed)
        monitor.wave_min_armlen_changed.connect(on_wave_min_armlen_changed)
        monitor.wave_octave_high_changed.connect(on_wave_octave_high_changed)
        monitor.wave_octave_low_changed.connect(on_wave_octave_low_changed)
        monitor.settings_changed.connect(on_settings_changed)
        
        # Connect LLM Singer signals (if enabled)
        if llm_enabled:
            monitor.llm_prompt_changed.connect(on_llm_prompt_changed)
            monitor.llm_interval_changed.connect(on_llm_interval_changed)
            monitor.llm_autotune_changed.connect(on_llm_autotune_changed)
            monitor.llm_reverb_changed.connect(on_llm_reverb_changed)
            monitor.llm_transpose_changed.connect(on_llm_transpose_changed)
            monitor.llm_gain_changed.connect(on_llm_gain_changed)
        
        # Override scale signal to use the full version
        monitor.scale_changed.disconnect(on_scale_changed)
        monitor.scale_changed.connect(on_scale_changed_full)
        
        # Load and apply saved settings AFTER signals are connected
        from bodysynth_settings import SettingsManager, DEFAULT_SETTINGS
        settings_manager = SettingsManager()
        saved_settings = settings_manager.load()
        
        # Block signals temporarily to avoid triggering saves during load
        monitor.blockSignals(True)
        monitor.set_all_settings(saved_settings)
        monitor.blockSignals(False)
        
        # Now manually apply settings to audio engines (since signals were blocked)
        print("[Settings] Applying saved settings to audio engines...")
        on_loop_length_changed(saved_settings.get("loop_length", DEFAULT_SETTINGS["loop_length"]))
        on_scale_changed_full(saved_settings.get("scale", DEFAULT_SETTINGS["scale"]))
        on_click_changed(saved_settings.get("click_enabled", DEFAULT_SETTINGS["click_enabled"]))
        on_drum_gain_changed(saved_settings.get("drum_gain", DEFAULT_SETTINGS["drum_gain"]))
        on_drum_distortion_changed(saved_settings.get("drum_distortion", DEFAULT_SETTINGS["drum_distortion"]))
        on_drum_compression_changed(saved_settings.get("drum_compression", DEFAULT_SETTINGS["drum_compression"]))
        on_wave_gain_changed(saved_settings.get("wave_gain", DEFAULT_SETTINGS["wave_gain"]))
        on_wave_distortion_changed(saved_settings.get("wave_distortion", DEFAULT_SETTINGS["wave_distortion"]))
        on_wave_smoothing_changed(saved_settings.get("wave_smoothing", DEFAULT_SETTINGS["wave_smoothing"]))
        on_wave_lfo_changed(saved_settings.get("wave_lfo_amount", DEFAULT_SETTINGS["wave_lfo_amount"]))
        on_wave_min_reverb_changed(saved_settings.get("wave_min_reverb", DEFAULT_SETTINGS["wave_min_reverb"]))
        on_wave_max_armlen_changed(saved_settings.get("wave_max_armlen", DEFAULT_SETTINGS["wave_max_armlen"]))
        on_wave_min_armlen_changed(saved_settings.get("wave_min_armlen", DEFAULT_SETTINGS["wave_min_armlen"]))
        on_wave_octave_high_changed(saved_settings.get("wave_octave_high", DEFAULT_SETTINGS["wave_octave_high"]))
        on_wave_octave_low_changed(saved_settings.get("wave_octave_low", DEFAULT_SETTINGS["wave_octave_low"]))
        if llm_enabled and llm_singer:
            on_llm_prompt_changed(saved_settings.get("llm_prompt", DEFAULT_SETTINGS["llm_prompt"]))
            on_llm_interval_changed(saved_settings.get("llm_interval", DEFAULT_SETTINGS["llm_interval"]))
            on_llm_autotune_changed(saved_settings.get("llm_autotune", DEFAULT_SETTINGS["llm_autotune"]))
            on_llm_reverb_changed(saved_settings.get("llm_reverb", DEFAULT_SETTINGS["llm_reverb"]))
            on_llm_transpose_changed(saved_settings.get("llm_transpose", DEFAULT_SETTINGS["llm_transpose"]))
            on_llm_gain_changed(saved_settings.get("llm_gain", DEFAULT_SETTINGS["llm_gain"]))
        
        print(f"[Settings] Settings loaded and applied successfully")
        
        timer = QTimer()
        timer.timeout.connect(update_gui)
        timer.start(50)  # Update every 50ms
    else:
        # No GUI - but still need to initialize LLM if enabled
        if llm_enabled:
            print("[INFO] Initializing LLM Singer (this may take a moment)...")
            llm_singer = LLMSinger(pyo_server, channels=CHANNELS)
            llm_singer.start()  # This will block until pre-generation is complete
            bodysynth_client.set_llm_singer(llm_singer)
            
            if llm_singer.is_ready():
                print(f"[INFO] LLM Singer ready with {len(llm_singer.pregenerated_phrases)} phrases!")
                # Enable frame processing now that LLM is ready
                bodysynth_client.enable_processing()
            else:
                print("[WARNING] LLM Singer not ready - check Ollama and Piper")
                # Enable processing anyway so other components work
                bodysynth_client.enable_processing()
        else:
            # No LLM, enable processing immediately
            bodysynth_client.enable_processing()
    
    # Run main loop
    try:
        if gui_enabled and app:
            print('[INFO] Running with GUI. Close window or press Ctrl+C to stop.')
            
            # Allow Ctrl-C to interrupt the Qt event loop
            # The timer that updates the GUI (every 50ms) allows Python to process signals
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            
            app.exec()
        else:
            print('[INFO] Running without GUI. Press Ctrl+C to stop.')
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print('\n[INFO] Stopping...')
    finally:
        # Cleanup
        if drum_sequencer:
            drum_sequencer.stop()
        if wavetable_synth:
            wavetable_synth.stop()
        if llm_singer:
            llm_singer.stop()
        if osc_sampler:
            osc_sampler.stop()
        if pyo_server:
            pyo_server.stop()
            pyo_server.shutdown()
        print("[INFO] Shutdown complete")


if __name__ == '__main__':
    main()
