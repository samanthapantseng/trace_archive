"""
OSC Sampler - Plays audio samples triggered by OSC messages
Listens for OSC messages in the format: [SAMPLE_NAME]|time=...|pos=(x,y,z)
Example: [HEAD_BUMP]|time=2025-11-05 09:48:02.000|pos=(333, 1321, -3008)
"""

import os
import re
import glob
from pathlib import Path
from pyo import *
from pythonosc import dispatcher
from pythonosc import osc_server
import threading


class OSCSampler:
    """
    OSC-controlled sampler that plays audio files from a samples directory.
    
    Listens for OSC messages in the format:
        /sample[filename]
    
    When received, it plays the corresponding audio file from the samples folder.
    """
    
    def __init__(self, samples_dir=None, osc_ip="0.0.0.0", osc_port=8000, pyo_server=None, gain=1.0, sample_mapping=None):
        """
        Initialize the OSC sampler.
        
        Args:
            samples_dir: Path to directory containing audio samples (default: ./samples relative to this script)
            osc_ip: IP address to listen on for OSC messages (default: 0.0.0.0 for all interfaces)
            osc_port: Port to listen on for OSC messages (default: 8000)
            pyo_server: Existing pyo Server instance (if None, will use global server)
            gain: Overall gain/volume for sample playback (0.0-1.0)
            sample_mapping: Dict mapping OSC names to sample filenames (e.g., {'HEAD_BUMP': 'piano', 'KICK': 'kick_drum'})
                           Keys should be uppercase OSC message names, values are sample names (without extension)
        """
        # If no samples_dir specified, use ./samples relative to this script's location
        if samples_dir is None:
            script_dir = Path(__file__).parent
            self.samples_dir = script_dir / "samples"
        else:
            self.samples_dir = Path(samples_dir)
        self.osc_ip = osc_ip
        self.osc_port = osc_port
        self.pyo_server = pyo_server
        self.gain_value = gain
        
        # Default sample mapping: OSC name -> sample filename (without extension)
        default_mapping = {
            'HEAD_BUMP': 'piano',
            'ARM_STRETCH': 'bass',
            'ARM_SPREAD': 'wood',
            'HANDSHAKE': 'metal',
            'HIGH_FIVE': 'drum',
            'HUG': 'underwater',
            'TITANIC': 'titanic'
        }
        
        # Use provided mapping, or default if None
        self.sample_mapping = sample_mapping if sample_mapping is not None else default_mapping
        
        # Dictionary to store loaded samples {filename: SfPlayer}
        self.samples = {}
        
        # Load all samples from directory
        self._load_samples()
        
        # Setup mixer after samples loaded
        if self.samples:
            # Create mixer and connect all sample outputs (each sample has 4 voices)
            self.mixer = Mixer(outs=2, chnls=2)
            mixer_idx = 0
            for name, data in self.samples.items():
                # Connect all voices for this sample to the mixer
                for output in data['outputs']:
                    self.mixer.addInput(mixer_idx, output)
                    self.mixer.setAmp(mixer_idx, 0, 0.8)  # Left channel
                    self.mixer.setAmp(mixer_idx, 1, 0.8)  # Right channel
                    mixer_idx += 1
            
            # Apply overall gain
            self.output_gain = Sig(self.gain_value)
            self.mixed_output = self.mixer * self.output_gain
            # Connect to audio output (like LLM Singer does with self.output.out())
            self.mixed_output.out()
            print(f"[OSCSampler] Audio mixer connected to output")
        else:
            self.mixer = None
            self.output_gain = Sig(self.gain_value)
            self.mixed_output = None
        
        # Setup OSC server
        self.osc_server = None
        self.server_thread = None
        self._setup_osc_server()
        
        print(f"[OSCSampler] Initialized with {len(self.samples)} samples")
        print(f"[OSCSampler] Listening on {self.osc_ip}:{self.osc_port}")
        print(f"[OSCSampler] Sample name determined by OSC address (e.g., /HEAD_BUMP)")
    
    def _load_samples(self):
        """Load all audio files from the samples directory"""
        print(f"[OSCSampler] Looking for samples in: {self.samples_dir.absolute()}")
        
        if not self.samples_dir.exists():
            print(f"[OSCSampler] WARNING: Samples directory not found: {self.samples_dir}")
            print(f"[OSCSampler] Creating directory: {self.samples_dir}")
            self.samples_dir.mkdir(parents=True, exist_ok=True)
            return
        
        # List all files in directory
        all_files = list(self.samples_dir.iterdir())
        print(f"[OSCSampler] Files in directory: {[f.name for f in all_files if f.is_file()]}")
        
        # Supported audio formats (lowercase and uppercase)
        audio_extensions = ['*.wav', '*.WAV', '*.mp3', '*.MP3', '*.aif', '*.AIF', 
                           '*.aiff', '*.AIFF', '*.flac', '*.FLAC', '*.ogg', '*.OGG']
        
        sample_files = []
        for ext in audio_extensions:
            found = list(self.samples_dir.glob(ext))
            if found:
                print(f"[OSCSampler] Found {len(found)} files matching {ext}")
            sample_files.extend(found)
        
        if not sample_files:
            print(f"[OSCSampler] WARNING: No audio files found in {self.samples_dir.absolute()}")
            print(f"[OSCSampler] Supported formats: .wav, .mp3, .aif, .aiff, .flac, .ogg")
            # List what files ARE in the directory
            all_files = list(self.samples_dir.iterdir())
            print(f"[OSCSampler] Files in directory: {[f.name for f in all_files if f.is_file()]}")
            return
        
        # Load each sample
        for idx, sample_path in enumerate(sample_files):
            sample_name = sample_path.stem  # Filename without extension
            
            try:
                # Load audio file and convert to DataTable + TrigEnv (like LLM Singer does)
                # Get sound file info (sndinfo returns [sr, dur, chnls, path])
                info = sndinfo(str(sample_path))
                duration = info[1]  # Duration in seconds
                
                # Create a SndTable from the audio file
                table = SndTable(str(sample_path))
                
                # Create multiple triggers/players for polyphony (allows 4 simultaneous playbacks)
                triggers = []
                outputs = []
                for voice in range(4):
                    trig = Trig()
                    player = TrigEnv(trig, table=table, dur=duration, mul=0.7)
                    panned = Pan(player, outs=2, pan=0.5)
                    # Stop the audio chain to prevent playing on startup
                    panned.stop()
                    triggers.append(trig)
                    outputs.append(panned)
                
                self.samples[sample_name.lower()] = {
                    'triggers': triggers,
                    'outputs': outputs,
                    'path': sample_path,
                    'duration': duration,
                    'current_voice': 0  # Round-robin voice selection
                }
                
                print(f"[OSCSampler] Loaded: {sample_name} ({sample_path.name}) - {duration:.2f}s")
                
            except Exception as e:
                print(f"[OSCSampler] ERROR loading {sample_path}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"[OSCSampler] Total samples loaded: {len(self.samples)}")
    
    def _setup_osc_server(self):
        """Setup OSC server to receive trigger messages"""
        disp = dispatcher.Dispatcher()
        
        # Handle all OSC messages with default handler
        disp.set_default_handler(self._osc_sample_handler)
        
        try:
            self.osc_server = osc_server.ThreadingOSCUDPServer(
                (self.osc_ip, self.osc_port), disp
            )
            
            # Start server in a separate thread
            self.server_thread = threading.Thread(
                target=self.osc_server.serve_forever,
                daemon=True
            )
            self.server_thread.start()
            
            print(f"[OSCSampler] OSC server started on {self.osc_ip}:{self.osc_port}")
            
        except Exception as e:
            print(f"[OSCSampler] ERROR starting OSC server: {e}")
            self.osc_server = None
    
    def _osc_sample_handler(self, address, *args):
        """
        Handle incoming OSC messages for sample triggering.
        
        The OSC address (channel) determines which sample to play.
        Example: Message on /HEAD_BUMP plays the head_bump sample (or mapped sample)
        
        Message format: [SAMPLE_NAME]|time=2025-11-05 09:48:02.000|pos=(333, 1321, -3008)
        """
        # Extract sample name from OSC address (remove leading /)
        osc_name = address.lstrip('/').upper()
        
        if not osc_name:
            print(f"[OSCSampler] Received message on root address, ignoring")
            return
        
        # Combine args into message string for logging
        full_message = ' '.join(str(arg) for arg in args) if args else ""
        
        # Extract additional info (optional, for logging)
        time_match = re.search(r'time=([^|]+)', full_message)
        pos_match = re.search(r'pos=\(([^)]+)\)', full_message)
        
        timestamp = time_match.group(1).strip() if time_match else "N/A"
        position = pos_match.group(1).strip() if pos_match else "N/A"
        
        print(f"[OSCSampler] /{osc_name} | time={timestamp} | pos=({position})")
        
        # Check if there's a mapping for this OSC name
        if osc_name in self.sample_mapping:
            sample_name = self.sample_mapping[osc_name].lower()
            print(f"[OSCSampler] Mapped {osc_name} -> {sample_name}")
        else:
            # No mapping, use OSC name as sample name
            sample_name = osc_name.lower()
        
        # Trigger the sample (use lowercase for matching)
        self.trigger_sample(sample_name, velocity=1.0)
    
    def trigger_sample(self, sample_name: str, velocity: float = 1.0, metadata: dict = None):
        """Trigger a loaded sample to play (using round-robin voice selection)
        
        Args:
            sample_name: Name of the sample to play
            velocity: Playback velocity/amplitude (0.0 to 1.0)
            metadata: Optional dict with time, pos, etc.
        """
        if sample_name not in self.samples:
            print(f"Warning: Sample '{sample_name}' not loaded")
            return
        
        # Use round-robin voice selection for polyphony
        sample = self.samples[sample_name]
        voice_idx = sample['current_voice']
        trigger = sample['triggers'][voice_idx]
        output = sample['outputs'][voice_idx]
        
        # Advance to next voice
        sample['current_voice'] = (voice_idx + 1) % 4
        
        # Start the audio chain and trigger playback
        output.play()
        trigger.play()
        print(f"Playing sample: {sample_name} (voice {voice_idx})")
    
    def set_gain(self, gain):
        """Set overall output gain"""
        self.gain_value = max(0.0, min(1.0, gain))
        self.output_gain.value = self.gain_value
        print(f"[OSCSampler] Gain set to {self.gain_value:.2f}")
    
    def list_samples(self):
        """Print list of available samples"""
        if not self.samples:
            print("[OSCSampler] No samples loaded")
            return
        
        print(f"[OSCSampler] Available samples ({len(self.samples)}):")
        for name in sorted(self.samples.keys()):
            path = self.samples[name]['path']
            print(f"  - {name} ({path.name})")
    
    def out(self):
        """Return the audio output for connection to pyo Server"""
        if self.mixed_output:
            return self.mixed_output
        else:
            # Return a silent signal if no samples loaded
            return Sig(0)
    
    def stop(self):
        """Stop the OSC server and cleanup"""
        if self.osc_server:
            self.osc_server.shutdown()
            print("[OSCSampler] OSC server stopped")
        
        # Stop all sample outputs
        for sample_data in self.samples.values():
            try:
                sample_data['output'].stop()
            except:
                pass
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop()


# Example usage and testing
if __name__ == "__main__":
    print("=== OSC Sampler Test ===")
    print("Starting pyo audio server...")
    
    # Initialize pyo server with output device 10
    s = Server(duplex=0, audio='portaudio')
    s.setOutputDevice(10)
    s.boot()
    s.start()
    
    # Define sample mapping: OSC message name -> sample filename (without extension)
    # Example: /HEAD_BUMP will play piano.mp3
    sample_mapping = {
        'HEAD_BUMP': 'piano',
        'ARM_STRETCH': 'bass',
        'ARM_SPREAD': 'wood',
        'HANDSHAKE': 'metal',
        'HIGH_FIVE': 'drum',
        'HUG': 'underwater',
        'TITANIC': 'titanic'
    }
    
    # Create OSC sampler (samples_dir defaults to ./samples relative to this script)
    sampler = OSCSampler(
        osc_ip="0.0.0.0",
        osc_port=8000,
        gain=0.8,
        sample_mapping=sample_mapping
    )
    
    # Audio is already connected in __init__, no need to call .out() again
    if not sampler.mixed_output:
        print("[OSCSampler] WARNING: No audio output (no samples loaded)")
    
    # List available samples
    sampler.list_samples()
    
    # Show sample mapping
    if sample_mapping:
        print("\n=== Sample Mapping ===")
        for osc_name, sample_name in sample_mapping.items():
            print(f"  /{osc_name} -> {sample_name}")
    
    print("\n=== Ready ===")
    print("Send OSC messages to trigger samples:")
    print("  Address determines sample: /HEAD_BUMP, /KICK, /SNARE, etc.")
    print("  Message format: [SAMPLE_NAME]|time=...|pos=(x,y,z)")
    print("  Example: Send to /HEAD_BUMP with message '[HEAD_BUMP]|time=2025-11-05 09:48:02.000|pos=(333, 1321, -3008)'")
    print("\nYou can test with:")
    print("  python3 test_osc_loop.py")
    print("\nPress Ctrl+C to quit...")
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sampler.stop()
        s.stop()
        s.shutdown()
