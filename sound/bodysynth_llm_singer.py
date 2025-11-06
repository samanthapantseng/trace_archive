"""
LLM-powered singing narrator for body synthesis
Pre-generates phrases on startup to avoid resource usage during runtime.
All LLM generation and TTS synthesis happens during initialization or prompt changes.
During playback, it cycles through pre-generated phrases with no CPU overhead.
"""

import numpy as np
import time
import threading
import queue
from pyo import *
import io
import wave
import sys
import os
import subprocess

# Add the LLM examples path to sys.path
llm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../client/examples/llm'))
if llm_path not in sys.path:
    sys.path.insert(0, llm_path)

from ollamaClient import OllamaClient

# --- Configuration ---
DEFAULT_BASE_PROMPT = "You are an alchemist trying to find a dance to create a bad spell on a kingdom. Describe what you see in poetic, mystical terms in one short sentence."
DEFAULT_GENERATION_INTERVAL = 15.0  # seconds
DEFAULT_AUTOTUNE_AMOUNT = 0.5  # 0.0-1.0
DEFAULT_REVERB_AMOUNT = 0.7  # 0.0-1.0
DEFAULT_GAIN = 0.6  # 0.0-1.0
DEFAULT_TRANSPOSE = 0  # Octaves to transpose (-4 to 0)
SAMPLE_RATE = 44100

# Pre-generation settings
NUM_PREGENERATED_PHRASES = 1  # How many phrases to pre-generate PER people count (0-5)
PREGENERATED_CACHE_DIR = "llm_singer_cache"  # Directory to store pre-generated audio

# Musical scales (semitones from root)
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


class LLMSinger:
    """LLM-powered singing narrator with auto-tune and reverb"""
    
    def __init__(self, pyo_server, base_prompt=DEFAULT_BASE_PROMPT, 
                 generation_interval=DEFAULT_GENERATION_INTERVAL,
                 autotune_amount=DEFAULT_AUTOTUNE_AMOUNT,
                 reverb_amount=DEFAULT_REVERB_AMOUNT,
                 gain=DEFAULT_GAIN,
                 transpose=DEFAULT_TRANSPOSE,
                 scale_name="Dorian",
                 channels=2,
                 on_lyrics_callback=None):
        self.pyo_server = pyo_server
        self.base_prompt = base_prompt
        self.generation_interval = generation_interval
        self.autotune_amount = autotune_amount
        self.reverb_amount = reverb_amount
        self.gain = gain
        self.transpose = transpose  # Octaves to transpose
        self.current_scale = scale_name
        self.channels = channels
        self.on_lyrics_callback = on_lyrics_callback
        
        # Store last generated lyrics
        self.last_lyrics = ""
        
        # LLM client
        self.llm_client = OllamaClient(model_name="llama3.2:1b")
        self.llm_ready = False
        
        # Piper TTS settings - using male voice
        self.piper_model = "en_US-danny-low"  # Male voice (Danny)
        self.piper_length_scale = 1.3  # 1.0 = normal speed, >1.0 = slower, <1.0 = faster
        
        # Pre-generated audio cache
        self.pregenerated_phrases = {}  # Dict of {people_count: [(text, audio_data), ...]}
        self.phrase_indices = {}  # Track which phrase index to play next per people count
        self.cache_dir = PREGENERATED_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Synchronization for pre-generation
        self.pregeneration_complete = threading.Event()
        self.pregeneration_complete.clear()
        
        # Thread-safe queue for audio data (to pass from background thread to main thread)
        self.audio_playback_queue = queue.Queue()
        
        # Root note for scale (A4 = 440 Hz)
        self.root_freq = 440.0
        self.current_note_index = 0  # Which note in the scale we're currently on
        

        
        # Melody generation
        self.current_melody = []  # List of note indices for the current phrase
        self.melody_note_duration = 0.5  # Duration of each note in seconds
        
        # Pyo audio chain - use TrigEnv like drums do
        # Pre-allocate a large table (10 seconds at 44100 Hz = 441000 samples)
        max_samples = SAMPLE_RATE * 10
        dummy_audio = np.zeros(max_samples, dtype=np.float32)
        self.current_table = DataTable(size=max_samples, init=dummy_audio.tolist())
        self.trig = Trig()
        # TrigEnv will play the entire table once when triggered
        self.player = TrigEnv(self.trig, table=self.current_table, dur=1.0, mul=0.0)
        
        # Step 1: Flatten voice to root frequency (440 Hz)
        # Estimate typical speech frequency (around 150-200 Hz for male, 200-250 Hz for female)
        # We'll use 200 Hz as average speech fundamental
        self.speech_fundamental = 200.0
        self.flatten_shift = self.root_freq - self.speech_fundamental  # Shift up to 440 Hz
        self.flattener = FreqShift(self.player, shift=self.flatten_shift, mul=1)
        
        # Step 2: Apply melodic pitch shifting on top of flattened voice
        # Use a simple constant shift that we'll update per phrase
        self.current_pitch_shift = 0.0
        self.pitch_shift = FreqShift(self.flattener, shift=self.current_pitch_shift, mul=1)
        
        # Mix between dry (flattened) and melody-shifted signal based on autotune_amount
        self.autotune_mix = Interp(self.flattener, self.pitch_shift, interp=self.autotune_amount)
        
        # Apply transpose (octave shift) BEFORE reverb
        # Each octave shifts by root_freq Hz (440 Hz)
        transpose_shift = self.transpose * self.root_freq
        self.transpose_shifter = FreqShift(self.autotune_mix, shift=transpose_shift, mul=1)
        
        # Add reverb effect after transpose
        self.reverb = Freeverb(self.transpose_shifter, size=0.8, damp=0.7, bal=self.reverb_amount)
        
        # Pan to stereo and output
        self.output = Pan(self.reverb, outs=self.channels, pan=0.5)
        self.output.out()
        
        print(f"[LLMSinger] Pre-created Pyo audio objects (using TrigEnv)")
        print(f"[LLMSinger] Server running: {self.pyo_server.getIsStarted()}")
        
        # Timing - set to negative so first generation happens immediately
        self.last_generation_time = -generation_interval
        self.is_playing = False
        self.is_generating = False
        self.lock = threading.Lock()
        
        # Background thread for LLM generation
        self.generation_thread = None
        self.running = False
        
        print(f"[LLMSinger] Initializing with interval={generation_interval}s")
        
        # Initialize LLM and pre-generate phrases in background
        threading.Thread(target=self._init_llm_and_pregenerate, daemon=True).start()
    
    def _init_llm_and_pregenerate(self):
        """Initialize LLM connection and pre-generate phrases in background"""
        print("[LLMSinger] Connecting to Ollama...")
        if self.on_lyrics_callback:
            self.on_lyrics_callback("Connecting to Ollama...")
        
        if self.llm_client.connect():
            self.llm_ready = True
            print("[LLMSinger] LLM ready")
            if self.on_lyrics_callback:
                self.on_lyrics_callback("Pre-generating phrases...")
            
            # Pre-generate phrases
            self._pregenerate_phrases()
            
            # Signal that pre-generation is complete
            self.pregeneration_complete.set()
            
            if self.on_lyrics_callback:
                self.on_lyrics_callback(f"Ready! {len(self.pregenerated_phrases)} phrases cached")
        else:
            print("[LLMSinger] LLM connection failed")
            if self.on_lyrics_callback:
                self.on_lyrics_callback("⚠️ LLM connection failed - check Ollama")
            # Even on failure, set the event so system doesn't hang
            self.pregeneration_complete.set()
    
    def _pregenerate_phrases(self):
        """Pre-generate phrases for different people counts (0-5+)"""
        print(f"[LLMSinger] Pre-generating {NUM_PREGENERATED_PHRASES} phrase(s) per people count...")
        
        # Store phrases by people count: {0: [(text, audio)], 1: [(text, audio)], ...}
        self.pregenerated_phrases = {}
        
        # Scenarios for each people count
        scenarios = {
            0: "empty space with no movement",
            1: "a lone figure moving through space",
            2: "2 figures dancing together",
            3: "3 figures dancing in harmony",
            4: "4 figures in coordinated motion",
            5: "5 or more figures creating patterns"
        }
        
        for people_count, scenario in scenarios.items():
            self.pregenerated_phrases[people_count] = []
            
            for phrase_idx in range(NUM_PREGENERATED_PHRASES):
                try:
                    prompt = f"{self.base_prompt}\n\nScene: {scenario}\n\nDescribe in ONE SHORT SENTENCE (maximum 10 words):"
                    response, error = self.llm_client.generate(prompt, temperature=0.9, max_tokens=30)
                    
                    if error:
                        continue
                    
                    text = response.strip().replace('\n', ' ')
                    print(f"[LLMSinger] {people_count}p #{phrase_idx}: {text}")
                    
                    audio_data = self._text_to_speech(text)
                    if audio_data is not None:
                        self.pregenerated_phrases[people_count].append((text, audio_data))
                except Exception as e:
                    print(f"[LLMSinger] Error {people_count}p #{phrase_idx}: {e}")
        
        total_phrases = sum(len(phrases) for phrases in self.pregenerated_phrases.values())
        print(f"[LLMSinger] Pre-generation complete! {total_phrases} total phrases ready")
    
    def start(self):
        """Start the generation loop"""
        if self.running:
            return
        
        self.running = True
        print("[LLMSinger] Started")
        print("[LLMSinger] Waiting for pre-generation to complete...")
        
        # Wait up to 60 seconds for pre-generation to complete
        if self.pregeneration_complete.wait(timeout=60):
            print(f"[LLMSinger] Pre-generation complete! {len(self.pregenerated_phrases)} phrases ready")
        else:
            print("[LLMSinger] WARNING: Pre-generation timeout after 60s")
    
    def is_ready(self):
        """Check if the LLM Singer is ready with pre-generated phrases"""
        return self.pregeneration_complete.is_set() and bool(self.pregenerated_phrases)
    
    def check_audio_queue(self):
        """Check for audio in the queue and play it. Must be called from main thread."""
        try:
            # Non-blocking check
            audio_data = self.audio_playback_queue.get_nowait()
            print(f"[LLMSinger] Retrieved audio from queue, calling _play_audio()")
            self._play_audio(audio_data)
        except queue.Empty:
            pass  # No audio to play
    
    def stop(self):
        """Stop the LLM Singer."""
        self.running = False
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=2.0)
    
    
    def should_play(self):
        """Check if it's time to play the next phrase"""
        # Wait for pre-generation to complete before allowing playback
        if not self.pregeneration_complete.is_set():
            return False
        
        # Need at least one phrase to play
        if not self.pregenerated_phrases:
            return False
        
        if self.is_playing:
            return False
        
        current_time = time.time()
        time_since_last = current_time - self.last_generation_time
        should_play = time_since_last >= self.generation_interval
        
        if should_play:
            print(f"[LLMSinger] Time to play next phrase! ({time_since_last:.1f}s since last)")
        
        return should_play
    
    def process_frame(self, frame_data):
        """Process a frame and potentially play matching phrase"""
        if not self.should_play():
            return
        
        # Update timestamp immediately
        self.last_generation_time = time.time()
        
        # Analyze frame to get people count
        people_count = self._get_people_count(frame_data)
        
        print(f"[LLMSinger] Processing frame for playback (people: {people_count})")
        
        # Play matching phrase in background thread to avoid blocking
        threading.Thread(
            target=self._play_matching_phrase,
            args=(people_count,),
            daemon=True
        ).start()
    
    def _get_people_count(self, frame_data):
        """Get number of people in frame"""
        if not frame_data or 'people' not in frame_data:
            return 0
        return min(len(frame_data['people']), 5)  # Cap at 5 for matching
    
    def _play_matching_phrase(self, people_count):
        """Play pre-generated phrase matching people count"""
        with self.lock:
            if self.is_playing:
                return
            self.is_playing = True
        
        try:
            if not self.pregenerated_phrases:
                return
            
            # Get phrase list matching people count
            if people_count not in self.pregenerated_phrases:
                available = sorted(self.pregenerated_phrases.keys())
                people_count = min(available, key=lambda x: abs(x - people_count))
            
            phrase_list = self.pregenerated_phrases[people_count]
            if not phrase_list:
                return
            
            # Get current index for this people count
            if people_count not in self.phrase_indices:
                self.phrase_indices[people_count] = 0
            
            idx = self.phrase_indices[people_count]
            text, audio_data = phrase_list[idx]
            self.phrase_indices[people_count] = (idx + 1) % len(phrase_list)
            
            print(f"[LLMSinger] {people_count}p: {text}")
            
            self.last_lyrics = text
            if self.on_lyrics_callback:
                self.on_lyrics_callback(text)
            
            self.audio_playback_queue.put(audio_data)
            
        except Exception as e:
            print(f"[LLMSinger] Error: {e}")
        finally:
            self.is_playing = False
    
    def _text_to_speech(self, text):
        """Convert text to speech using Piper"""
        temp_path = "llm_singer_debug.wav"
        result = subprocess.run(
            ['piper', '--model', self.piper_model, '--length_scale', str(self.piper_length_scale), '--output_file', temp_path],
            input=text,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return None
        
        # Read the audio file
        with wave.open(temp_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            audio_bytes = wf.readframes(wf.getnframes())
        
        # Convert to numpy array and normalize
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Handle stereo
        if n_channels == 2:
            audio_data = audio_data[::2]
        
        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            audio_data = self._resample(audio_data, sample_rate, SAMPLE_RATE)
        
        return audio_data
    
    def _resample(self, audio, orig_sr, target_sr):
        """Simple resampling using linear interpolation"""
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        
        x_old = np.linspace(0, len(audio) - 1, len(audio))
        x_new = np.linspace(0, len(audio) - 1, target_length)
        
        return np.interp(x_new, x_old, audio)
    
    def _play_audio(self, audio_data):
        """Play audio with melodic pitch shifting"""
        try:
            print(f"[LLMSinger] Playing audio: {len(audio_data)} samples at {SAMPLE_RATE}Hz")
            print(f"[LLMSinger] Audio range: min={audio_data.min():.3f}, max={audio_data.max():.3f}")
            
            # Pad audio to match table size if needed, or truncate if too long
            table_size = self.current_table.getSize()
            if len(audio_data) > table_size:
                print(f"[LLMSinger] Warning: Audio too long, truncating from {len(audio_data)} to {table_size}")
                audio_data = audio_data[:table_size]
            elif len(audio_data) < table_size:
                # Pad with zeros
                padded = np.zeros(table_size, dtype=np.float32)
                padded[:len(audio_data)] = audio_data
                audio_data = padded
            
            # Replace table data using the existing table
            self.current_table.replace(audio_data.tolist())
            
            # Calculate duration in seconds (use actual audio length, not padded)
            duration = len(audio_data) / SAMPLE_RATE
            
            print(f"[LLMSinger] Table size: {self.current_table.getSize()}, Duration: {duration:.2f}s")
            print(f"[LLMSinger] Player gain: {self.gain}")
            
            # Generate melody for this phrase - pick a random note from the scale
            melody = self._generate_melody(duration)
            self.current_melody = melody
            
            # Use the first note of the melody for the entire phrase
            # (simplified - no time-varying modulation to avoid Pyo issues)
            if melody:
                note_idx = melody[0][0]
                shift = self._calculate_scale_shift(note_idx)
                self.pitch_shift.shift = shift
                print(f"[LLMSinger] Using note {note_idx} with shift: {shift:.1f} Hz")
            
            # Update TrigEnv with duration
            self.player.setDur(duration)
            self.player.setMul(self.gain)
            
            # Trigger playback (like drums do)
            self.trig.play()
            
            print(f"[LLMSinger] Triggered playback with melody!")
            
        except Exception as e:
            print(f"[LLMSinger] Playback error: {e}")
            import traceback
            traceback.print_exc()
    
    def set_base_prompt(self, prompt):
        """Update the base prompt and regenerate phrases"""
        self.base_prompt = prompt
        print(f"[LLMSinger] Base prompt updated, regenerating phrases...")
        
        # Regenerate phrases in background
        if self.on_lyrics_callback:
            self.on_lyrics_callback("Regenerating phrases for new prompt...")
        
        threading.Thread(target=self._regenerate_phrases, daemon=True).start()
    
    def _regenerate_phrases(self):
        """Regenerate all phrases with the new prompt"""
        # Clear old phrases and indices
        old_count = sum(len(phrases) for phrases in self.pregenerated_phrases.values())
        self.pregenerated_phrases.clear()
        self.phrase_indices.clear()
        
        print(f"[LLMSinger] Cleared {old_count} old phrases, generating new ones...")
        
        # Pre-generate new phrases (will create dict by people_count)
        self._pregenerate_phrases()
        
        total_phrases = sum(len(phrases) for phrases in self.pregenerated_phrases.values())
        if self.on_lyrics_callback:
            self.on_lyrics_callback(f"Ready! {total_phrases} new phrases cached")
    
    def set_generation_interval(self, interval):
        """Update generation interval in seconds"""
        self.generation_interval = interval
        print(f"[LLMSinger] Generation interval set to {interval}s")
    
    def set_autotune_amount(self, amount):
        """Set auto-tune amount (0.0-1.0)"""
        self.autotune_amount = float(np.clip(amount, 0.0, 1.0))
        if hasattr(self, 'autotune_mix'):
            self.autotune_mix.interp = self.autotune_amount
        print(f"[LLMSinger] Auto-tune amount: {self.autotune_amount:.2f}")
    
    def set_reverb_amount(self, amount):
        """Set reverb amount (0.0-1.0)"""
        self.reverb_amount = float(np.clip(amount, 0.0, 1.0))
        if self.reverb:
            self.reverb.bal = self.reverb_amount
        print(f"[LLMSinger] Reverb amount: {self.reverb_amount:.2f}")
    
    def set_gain(self, gain):
        """Set output gain (0.0-1.0)"""
        self.gain = gain
        if self.player:
            self.player.mul = gain
        print(f"[LLMSinger] Gain: {self.gain:.2f}")
    
    def set_transpose(self, octaves):
        """Set transpose in octaves (-4 to 0)"""
        self.transpose = float(np.clip(octaves, -4, 0))
        if hasattr(self, 'transpose_shifter'):
            # Each octave shifts by root_freq Hz (440 Hz)
            transpose_shift = self.transpose * self.root_freq
            self.transpose_shifter.shift = transpose_shift
        print(f"[LLMSinger] Transpose: {self.transpose:.2f} octaves ({self.transpose * self.root_freq:.1f} Hz)")
    
    def set_scale(self, scale_name):
        """Set the musical scale for auto-tune"""
        if scale_name in SCALES:
            self.current_scale = scale_name
            # Update pitch shift for current note in new scale
            shift = self._calculate_scale_shift(self.current_note_index)
            if hasattr(self, 'pitch_shift'):
                self.pitch_shift.shift = shift
            print(f"[LLMSinger] Scale: {scale_name}, shift: {shift:.1f} Hz")
        else:
            print(f"[LLMSinger] Unknown scale: {scale_name}")
    
    def _generate_melody(self, duration_seconds):
        """Generate a random melody in the current scale
        
        Args:
            duration_seconds: Duration of the phrase in seconds
        
        Returns:
            List of (note_index, duration) tuples
        """
        scale = SCALES[self.current_scale]
        num_notes = int(duration_seconds / self.melody_note_duration)
        
        # Ensure at least one note
        if num_notes < 1:
            num_notes = 1
        
        melody = []
        for i in range(num_notes):
            # Random walk through the scale (prefer stepwise motion)
            # Use notes across 2 octaves for more pronounced effect
            if i == 0:
                # Start on root, fifth, or octave (more dramatic starting notes)
                note_idx = np.random.choice([0, 4, 7, len(scale)])
            else:
                # Move by step (±1 or ±2 scale degrees) with occasional larger jumps
                last_note = melody[-1][0]
                if np.random.random() < 0.6:  # 60% stepwise motion
                    step = np.random.choice([-2, -1, 1, 2])
                else:  # 40% larger jumps (more dramatic)
                    step = np.random.choice([-5, -4, -3, 3, 4, 5, 7])  # Include octave jumps
                
                note_idx = (last_note + step) % (len(scale) * 2)  # Use 2 octaves
            
            melody.append((note_idx, self.melody_note_duration))
        
        print(f"[LLMSinger] Generated melody: {[n[0] for n in melody]}")
        return melody
    
    def _calculate_scale_shift(self, note_index):
        """Calculate frequency shift to reach a specific note in the current scale
        
        Args:
            note_index: Index of the note in the scale (0 = root, 1 = second note, etc.)
                       Can span multiple octaves (e.g., 0-13 for 2 octaves of a 7-note scale)
        
        Returns:
            Frequency shift in Hz
        """
        # Get the scale degrees (semitones from root)
        scale = SCALES[self.current_scale]
        
        # Calculate which octave and which note in the scale
        octave = note_index // len(scale)
        scale_note = note_index % len(scale)
        
        # Get semitones for this note within its octave
        semitones = scale[scale_note]
        
        # Add octave offset (12 semitones per octave)
        total_semitones = semitones + (octave * 12)
        
        # Calculate target frequency using equal temperament formula
        # f = f0 * 2^(semitones/12)
        target_freq = self.root_freq * (2.0 ** (total_semitones / 12.0))
        
        # The shift is the difference from root (already flattened to root)
        shift = target_freq - self.root_freq
        
        return float(shift)
