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
from bodysynth_gui import SynthMonitor

# --- Configuration ---
SAMPLE_RATE = 44100
CHANNELS = 2

# GUI data
synth_voice_data = {}
data_lock = threading.Lock()


def extract_head_and_arm(frame):
    """Extract head positions and armline joints from frame"""
    body_model = frame.body_model if frame.body_model else "BODY_34"
    
    curr_armlines = []
    curr_headpos = []
    for person in frame.people:
        if person.confidence > 70:
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
        
        # Person debouncing: track last seen time for each person
        self.person_last_seen = {}  # {person_id: timestamp}
        self.debounce_time = 1  # 500ms debounce
        
        print(f"[BodysynthClient] Initialized - Drum: {drum_enabled}, Wave: {wave_enabled}, GUI: {gui_enabled}")
    
    def set_drum_sequencer(self, drum_sequencer):
        """Set the drum sequencer instance"""
        self.drum_sequencer = drum_sequencer
    
    def set_wavetable_synth(self, wavetable_synth):
        """Set the wavetable synth instance"""
        self.wavetable_synth = wavetable_synth
    
    def on_init(self):
        print(f"[INIT] Connected to server")
    
    def on_frame(self, frame: Frame):
        """Process incoming frame and dispatch to enabled components"""
        snap = extract_head_and_arm(frame)
        
        current_time = time.time()
        
        # Update last seen time for all currently detected people
        active_ids = set([arm['p'] for arm in snap['armlines']])
        for person_id in active_ids:
            self.person_last_seen[person_id] = current_time
        
        # Determine which people to keep (currently active OR within debounce window)
        kept_person_ids = set()
        for person_id, last_seen in list(self.person_last_seen.items()):
            time_since_seen = current_time - last_seen
            if time_since_seen <= self.debounce_time:
                kept_person_ids.add(person_id)
            else:
                # Remove from tracking after debounce period
                del self.person_last_seen[person_id]
        
        # Filter snap data to only include kept people
        snap['headpos'] = [h for h in snap['headpos'] if h['p'] in kept_person_ids]
        snap['armlines'] = [a for a in snap['armlines'] if a['p'] in kept_person_ids]
        
        drum_info = None
        wave_info = {}
        
        # Process drum sequencer if enabled
        if self.drum_enabled and self.drum_sequencer:
            drum_info = self.drum_sequencer.process_frame(snap['headpos'])
        
        # Process wavetable synth if enabled
        if self.wave_enabled and self.wavetable_synth:
            wave_info = self.wavetable_synth.process_frame(snap['armlines'])
        
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
    parser.add_argument("--gui", action="store_true", help="Enable GUI visualization")
    parser.add_argument("--all", action="store_true", help="Enable all components (drum + wave + gui)")
    
    args = parser.parse_args()
    
    # Determine which components to enable
    drum_enabled = args.drum or args.all
    wave_enabled = args.wave or args.all
    gui_enabled = args.gui or args.all
    
    # If nothing specified, enable all by default
    if not (args.drum or args.wave or args.gui or args.all):
        print("[INFO] No components specified, enabling all by default")
        drum_enabled = wave_enabled = gui_enabled = True
    
    print(f"[INFO] Enabled components: Drum={drum_enabled}, Wave={wave_enabled}, GUI={gui_enabled}")
    
    # Setup GUI/QApplication FIRST before any pyo initialization
    app = None
    if gui_enabled:
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        print("[INFO] QApplication created")
    
    # Initialize pyo audio server if any audio component is enabled
    pyo_server = None
    if drum_enabled or wave_enabled:
        s = Server(sr=SAMPLE_RATE, nchnls=CHANNELS, duplex=0)
        s.setOutputDevice(3)
        s.boot().start()
        pyo_server = s
        print("[INFO] Pyo audio server started on output device 3")
    
    # Setup GUI first if enabled (before creating audio objects)
    monitor = None
    timer = None
    
    # Initialize components in main thread (critical for pyo)
    drum_sequencer = None
    wavetable_synth = None
    
    if drum_enabled:
        drum_sequencer = DrumSequencer(pyo_server, num_lanes=NUM_LANES, loop_length=LOOP_LENGTH)
    
    if wave_enabled:
        wavetable_synth = WavetableSynth(pyo_server, gain=0.3)
    
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
        
        # Track loop start time for GUI
        gui_loop_start_time = time.time()
        
        def update_gui():
            with data_lock:
                monitor.update_signal.emit(dict(synth_voice_data))
            
            # Update loop position - use drum sequencer time if available, else use our own
            if drum_enabled and drum_sequencer and drum_sequencer.loop_start_time:
                elapsed = time.time() - drum_sequencer.loop_start_time
            else:
                elapsed = time.time() - gui_loop_start_time
            
            loop_position = (elapsed % LOOP_LENGTH) / LOOP_LENGTH
            monitor.head_widget.update_loop_position(loop_position)
        
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
        if pyo_server:
            pyo_server.stop()
            pyo_server.shutdown()
        print("[INFO] Shutdown complete")


if __name__ == '__main__':
    main()
