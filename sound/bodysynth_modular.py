"""
Modular bodysynth system - combines drum sequencer, wavetable synth, and visualization
Components can be independently enabled/disabled via command-line arguments
"""

import numpy as np
import time
import threading
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
from bodysynth_drum_class import DrumSequencer, LOOP_LENGTH, NUM_LANES, TOP_LEFT, BOTTOM_RIGHT
from bodysynth_wave_class import WavetableSynth
from bodysynth_llm_singer import LLMSinger
from bodysynth_gui import SynthMonitor

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


class BodysynthClient:
    """Client that processes frames and dispatches to enabled components"""
    
    def __init__(self, drum_enabled=True, wave_enabled=True, gui_enabled=True):
        self.drum_enabled = drum_enabled
        self.wave_enabled = wave_enabled
        self.gui_enabled = gui_enabled
        
        self.drum_sequencer = None
        self.wavetable_synth = None
        self.llm_singer = None
        
        # Person caching: track last seen time and data for each person
        self.person_last_seen = {}  # {person_id: timestamp}
        self.person_cache = {}  # {person_id: {'headpos': ..., 'armline': ...}}
        self.cache_time = CACHE_TIME  # Cache for 5 seconds
        
        print(f"[BodysynthClient] Initialized - Drum: {drum_enabled}, Wave: {wave_enabled}, GUI: {gui_enabled}")
    
    def set_drum_sequencer(self, drum_sequencer):
        """Set the drum sequencer instance"""
        self.drum_sequencer = drum_sequencer
    
    def set_wavetable_synth(self, wavetable_synth):
        """Set the wavetable synth instance"""
        self.wavetable_synth = wavetable_synth
    
    def set_llm_singer(self, llm_singer):
        """Set the LLM singer instance"""
        self.llm_singer = llm_singer
    
    def on_init(self):
        print(f"[INIT] Connected to server")
    
    def on_frame(self, frame: Frame):
        """Process incoming frame and dispatch to enabled components"""
        snap = extract_head_and_arm(frame)
        
        current_time = time.time()
        
        # Update last seen time and cache data for all currently detected people
        active_ids = set([arm['p'] for arm in snap['armlines']])
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
        
        # Combine live and cached data
        snap['headpos'] = [h for h in snap['headpos'] if h['p'] in kept_person_ids] + cached_headpos
        snap['armlines'] = [a for a in snap['armlines'] if a['p'] in kept_person_ids] + cached_armlines
        
        drum_info = None
        wave_info = {}
        
        # Process drum sequencer if enabled
        if self.drum_enabled and self.drum_sequencer:
            drum_info = self.drum_sequencer.process_frame(snap['headpos'])
        
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
    parser.add_argument("--gui", action="store_true", help="Enable GUI visualization")
    parser.add_argument("--all", action="store_true", help="Enable all components (drum + wave + llm + gui)")
    
    args = parser.parse_args()
    
    # Determine which components to enable
    drum_enabled = args.drum or args.all
    wave_enabled = args.wave or args.all
    llm_enabled = args.llm or args.all
    gui_enabled = args.gui or args.all

    
    # If nothing specified, enable all by default
    if not (args.drum or args.wave or args.llm or args.gui or args.all):
        print("[INFO] No components specified, enabling all by default")
        drum_enabled = wave_enabled = llm_enabled = gui_enabled = True
    
    print(f"[INFO] Enabled components: Drum={drum_enabled}, Wave={wave_enabled}, LLM={llm_enabled}, GUI={gui_enabled}")
    
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
    if drum_enabled or wave_enabled or llm_enabled:
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
    
    # Setup GUI first if enabled (before creating audio objects)
    monitor = None
    timer = None
    
    # Initialize components in main thread (critical for pyo)
    drum_sequencer = None
    wavetable_synth = None
    llm_singer = None
    
    if drum_enabled:
        drum_sequencer = DrumSequencer(pyo_server, num_lanes=NUM_LANES, loop_length=LOOP_LENGTH, channels=CHANNELS)
    
    if wave_enabled:
        wavetable_synth = WavetableSynth(pyo_server, gain=0.3, sample_rate=actual_sample_rate, channels=CHANNELS)
    
    # LLM Singer will be initialized after GUI
    llm_singer = None
    
    # Create client
    bodysynth_client = BodysynthClient(drum_enabled=drum_enabled, 
                                       wave_enabled=wave_enabled, 
                                       gui_enabled=gui_enabled)
    
    if drum_sequencer:
        bodysynth_client.set_drum_sequencer(drum_sequencer)
    if wavetable_synth:
        bodysynth_client.set_wavetable_synth(wavetable_synth)
    
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
    
    # Setup GUI monitor if enabled
    
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
        
        monitor.show()
        
        # Initialize LLM Singer with lyrics callback to GUI (if enabled)
        if llm_enabled:
            def on_lyrics_generated(lyrics):
                """Callback when lyrics are generated"""
                monitor.llm_lyrics_update.emit(lyrics)
            
            llm_singer = LLMSinger(pyo_server, channels=CHANNELS, on_lyrics_callback=on_lyrics_generated)
            llm_singer.start()
            bodysynth_client.set_llm_singer(llm_singer)
            print("[INFO] LLM Singer initialized with GUI callback")
        else:
            print("[INFO] LLM Singer disabled")
        
        # Track loop start time and length for GUI
        gui_loop_start_time = time.time()
        current_loop_length = [LOOP_LENGTH]  # Use list to make it mutable in closure
        
        def update_gui():
            with data_lock:
                monitor.update_signal.emit(dict(synth_voice_data))
            
            # Update loop position - use drum sequencer time if available, else use our own
            if drum_enabled and drum_sequencer and drum_sequencer.loop_start_time:
                elapsed = time.time() - drum_sequencer.loop_start_time
            else:
                elapsed = time.time() - gui_loop_start_time
            
            loop_position = (elapsed % current_loop_length[0]) / current_loop_length[0]
            monitor.head_widget.update_loop_position(loop_position)
            
            # Check for LLM audio to play (main thread context)
            if llm_singer:
                llm_singer.check_audio_queue()
        
        def on_loop_length_changed(new_length):
            """Handle loop length changes from GUI"""
            current_loop_length[0] = new_length
            if drum_enabled and drum_sequencer:
                drum_sequencer.loop_length = new_length
                # Reset loop timing
                drum_sequencer.loop_start_time = time.time()
                drum_sequencer.triggered_this_loop.clear()
                print(f"[INFO] Loop length changed to {new_length:.1f}s")
        
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
        
        def on_llm_gain_changed(gain):
            """Handle LLM gain changes from GUI"""
            if llm_singer:
                llm_singer.set_gain(gain)
        
        # Connect GUI signals
        monitor.loop_length_changed.connect(on_loop_length_changed)
        monitor.scale_changed.connect(on_scale_changed)
        monitor.click_enabled_changed.connect(on_click_changed)
        monitor.drum_gain_changed.connect(on_drum_gain_changed)
        monitor.wave_gain_changed.connect(on_wave_gain_changed)
        
        # Connect LLM Singer signals (if enabled)
        if llm_enabled:
            monitor.llm_prompt_changed.connect(on_llm_prompt_changed)
            monitor.llm_interval_changed.connect(on_llm_interval_changed)
            monitor.llm_autotune_changed.connect(on_llm_autotune_changed)
            monitor.llm_reverb_changed.connect(on_llm_reverb_changed)
            monitor.llm_gain_changed.connect(on_llm_gain_changed)
        
        # Sync scale changes to LLM Singer as well
        def on_scale_changed_full(scale_name):
            if wave_enabled and wavetable_synth:
                wavetable_synth.set_scale(scale_name)
            if llm_singer:
                llm_singer.set_scale(scale_name)
        
        monitor.scale_changed.disconnect(on_scale_changed)
        monitor.scale_changed.connect(on_scale_changed_full)
        
        timer = QTimer()
        timer.timeout.connect(update_gui)
        timer.start(50)  # Update every 50ms
    
    # Run main loop
    try:
        if gui_enabled and app:
            print('[INFO] Running with GUI. Close window to stop.')
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
        if pyo_server:
            pyo_server.stop()
            pyo_server.shutdown()
        print("[INFO] Shutdown complete")


if __name__ == '__main__':
    main()
