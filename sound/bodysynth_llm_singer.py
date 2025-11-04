"""
LLM-powered singing narrator for body synthesis
Analyzes frames periodically, generates descriptions via LLM, and sings them with auto-tuned voice
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
DEFAULT_GENERATION_INTERVAL = 30.0  # seconds
DEFAULT_AUTOTUNE_AMOUNT = 0.5  # 0.0-1.0
DEFAULT_REVERB_AMOUNT = 0.7  # 0.0-1.0
DEFAULT_GAIN = 0.6  # 0.0-1.0
SAMPLE_RATE = 44100

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
                 scale_name="Dorian",
                 channels=2,
                 on_lyrics_callback=None):
        self.pyo_server = pyo_server
        self.base_prompt = base_prompt
        self.generation_interval = generation_interval
        self.autotune_amount = autotune_amount
        self.reverb_amount = reverb_amount
        self.gain = gain
        self.current_scale = scale_name
        self.channels = channels
        self.on_lyrics_callback = on_lyrics_callback
        
        # Store last generated lyrics
        self.last_lyrics = ""
        
        # LLM client
        self.llm_client = OllamaClient(model_name="llama3.2:1b")
        self.llm_ready = False
        
        # Piper TTS settings
        self.piper_model = "en_US-lessac-medium"
        self.piper_length_scale = 1.3  # 1.0 = normal speed, >1.0 = slower, <1.0 = faster
        
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
        
        # Add reverb effect after auto-tune
        self.reverb = Freeverb(self.autotune_mix, size=0.8, damp=0.7, bal=self.reverb_amount)
        
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
        
        # Initialize LLM in background
        threading.Thread(target=self._init_llm, daemon=True).start()
    
    def _init_llm(self):
        """Initialize LLM connection in background"""
        print("[LLMSinger] Connecting to Ollama...")
        if self.on_lyrics_callback:
            self.on_lyrics_callback("Connecting to Ollama...")
        
        if self.llm_client.connect():
            self.llm_ready = True
            print("[LLMSinger] LLM ready")
            if self.on_lyrics_callback:
                self.on_lyrics_callback("LLM ready - waiting for first frame...")
        else:
            print("[LLMSinger] LLM connection failed")
            if self.on_lyrics_callback:
                self.on_lyrics_callback("⚠️ LLM connection failed - check Ollama")
    
    def start(self):
        """Start the generation loop"""
        if self.running:
            return
        
        self.running = True
        self.generation_thread = threading.Thread(target=self._generation_loop, daemon=True)
        self.generation_thread.start()
        print("[LLMSinger] Started")
    
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
    
    def _generation_loop(self):
        """Background loop that triggers generation at intervals"""
        while self.running:
            current_time = time.time()
            
            # Check if it's time to generate
            if current_time - self.last_generation_time >= self.generation_interval:
                if self.llm_ready and not self.is_playing:
                    self.last_generation_time = current_time
                    # Trigger generation (will be called with frame data from main thread)
                    print(f"[LLMSinger] Generation cycle triggered")
            
            time.sleep(0.5)
    
    def should_generate(self):
        """Check if it's time to generate new content"""
        if not self.llm_ready:
            return False
        
        if self.is_playing:
            return False
        
        if self.is_generating:
            return False
        
        current_time = time.time()
        time_since_last = current_time - self.last_generation_time
        should_gen = time_since_last >= self.generation_interval
        
        if should_gen:
            print(f"[LLMSinger] Time to generate! ({time_since_last:.1f}s since last)")
        
        return should_gen
    
    def process_frame(self, frame_data):
        """Process a frame and potentially generate singing"""
        if not self.should_generate():
            return
        
        # Mark as generating and update timestamp immediately
        self.is_generating = True
        self.last_generation_time = time.time()
        
        print(f"[LLMSinger] Processing frame for generation")
        
        # Generate in background thread to avoid blocking
        threading.Thread(
            target=self._generate_and_sing,
            args=(frame_data,),
            daemon=True
        ).start()
    
    def _generate_and_sing(self, frame_data):
        """Generate description and sing it (runs in background thread)"""
        with self.lock:
            if self.is_playing:
                self.is_generating = False
                return
            
            self.is_playing = True
        
        try:
            # Analyze frame data
            description = self._analyze_frame(frame_data)
            
            # Generate LLM response
            print(f"[LLMSinger] Analyzing: {description}")
            prompt = f"{self.base_prompt}\n\nScene: {description}\n\nDescribe in ONE SHORT SENTENCE (maximum 10 words):"
            
            response, error = self.llm_client.generate(prompt, temperature=0.8, max_tokens=30)
            
            if error:
                print(f"[LLMSinger] LLM error: {error}")
                self.is_playing = False
                self.is_generating = False
                return
            
            # Clean up response
            text = response.strip().replace('\n', ' ')
            print(f"[LLMSinger] Singing: {text}")
            
            # Store and callback with lyrics
            self.last_lyrics = text
            if self.on_lyrics_callback:
                self.on_lyrics_callback(text)
            
            # Generate speech (melody will be applied during playback)
            audio_data = self._text_to_speech(text)
            
            if audio_data is not None:
                print(f"[LLMSinger] Generated {len(audio_data)} audio samples")
                # Queue audio for playback in main thread
                self.audio_playback_queue.put(audio_data)
                print(f"[LLMSinger] Audio queued for playback")
            else:
                print(f"[LLMSinger] Failed to generate audio")
            
        except Exception as e:
            print(f"[LLMSinger] Error: {e}")
        finally:
            self.is_playing = False
            self.is_generating = False
    
    def _analyze_frame(self, frame_data):
        """Analyze frame data and create description"""
        if not frame_data or 'people' not in frame_data:
            return "empty space with no movement"
        
        num_people = len(frame_data['people'])
        
        if num_people == 0:
            return "empty space with no movement"
        elif num_people == 1:
            return "a lone figure moving through space"
        else:
            return f"{num_people} figures dancing together"
    
    def _text_to_speech(self, text):
        """Convert text to speech using Piper and return audio data"""
        try:
            print(f"[LLMSinger] Converting to speech: '{text}'")
            # Save to project folder for debugging
            import os
            
            temp_path = "llm_singer_debug.wav"
            
            print(f"[LLMSinger] TTS temp file: {temp_path}")
            
            # Generate speech using Piper
            # Pass text via stdin to piper
            result = subprocess.run(
                ['piper', '--model', self.piper_model, '--length_scale', str(self.piper_length_scale), '--output_file', temp_path],
                input=text,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"[LLMSinger] Piper error: {result.stderr}")
                return None
            
            print(f"[LLMSinger] TTS generation complete")
            
            # Check if file exists and has content
            if not os.path.exists(temp_path):
                print(f"[LLMSinger] Error: TTS file not created")
                return None
            
            file_size = os.path.getsize(temp_path)
            print(f"[LLMSinger] TTS file size: {file_size} bytes")
            
            if file_size == 0:
                print(f"[LLMSinger] Error: TTS file is empty")
                return None
            
            # Read the audio file
            with wave.open(temp_path, 'rb') as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                n_frames = wf.getnframes()
                audio_bytes = wf.readframes(n_frames)
            
            # Convert to numpy array
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            audio_data = audio_data / 32768.0  # Normalize to -1.0 to 1.0
            
            # Handle stereo if needed
            if n_channels == 2:
                audio_data = audio_data[::2]  # Take left channel
            
            # Resample if needed
            if sample_rate != SAMPLE_RATE:
                audio_data = self._resample(audio_data, sample_rate, SAMPLE_RATE)
            
            # Don't delete temp file for debugging
            print(f"[LLMSinger] TTS audio saved to: {temp_path}")
            # try:
            #     os.unlink(temp_path)
            # except:
            #     pass
            
            return audio_data
            
        except Exception as e:
            import traceback
            print(f"[LLMSinger] TTS error: {e}")
            traceback.print_exc()
            return None
    
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
        """Update the base prompt"""
        self.base_prompt = prompt
        print(f"[LLMSinger] Base prompt updated")
    
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
            if i == 0:
                # Start on root or fifth
                note_idx = np.random.choice([0, 4])
            else:
                # Move by step (±1 or ±2 scale degrees) with occasional jumps
                last_note = melody[-1][0]
                if np.random.random() < 0.7:  # 70% stepwise motion
                    step = np.random.choice([-2, -1, 1, 2])
                else:  # 30% jumps
                    step = np.random.choice([-4, -3, 3, 4])
                
                note_idx = (last_note + step) % len(scale)
            
            melody.append((note_idx, self.melody_note_duration))
        
        print(f"[LLMSinger] Generated melody: {[n[0] for n in melody]}")
        return melody
    
    def _calculate_scale_shift(self, note_index):
        """Calculate frequency shift to reach a specific note in the current scale
        
        Args:
            note_index: Index of the note in the scale (0 = root, 1 = second note, etc.)
        
        Returns:
            Frequency shift in Hz
        """
        # Get the scale degrees (semitones from root)
        scale = SCALES[self.current_scale]
        
        # Wrap note index to scale length
        note_index = note_index % len(scale)
        
        # Get semitones for this note
        semitones = scale[note_index]
        
        # Calculate target frequency using equal temperament formula
        # f = f0 * 2^(semitones/12)
        target_freq = self.root_freq * (2.0 ** (semitones / 12.0))
        
        # The shift is the difference from root (already flattened to root)
        shift = target_freq - self.root_freq
        
        return float(shift)
    
    def cycle_scale_note(self):
        """Move to the next note in the scale"""
        scale = SCALES[self.current_scale]
        self.current_note_index = (self.current_note_index + 1) % len(scale)
        
        # Update the pitch shift
        shift = self._calculate_scale_shift(self.current_note_index)
        if hasattr(self, 'pitch_shift'):
            self.pitch_shift.shift = shift
        
        semitones = scale[self.current_note_index]
        print(f"[LLMSinger] Note: {self.current_note_index} (semitone: {semitones}, shift: {shift:.1f} Hz)")
        
        return shift
    
    def set_scale_note(self, note_index):
        """Set a specific note in the scale
        
        Args:
            note_index: Index of the note in the scale (0-based)
        """
        scale = SCALES[self.current_scale]
        self.current_note_index = note_index % len(scale)
        
        # Update the pitch shift
        shift = self._calculate_scale_shift(self.current_note_index)
        if hasattr(self, 'pitch_shift'):
            self.pitch_shift.shift = shift
        
        semitones = scale[self.current_note_index]
        print(f"[LLMSinger] Note: {self.current_note_index} (semitone: {semitones}, shift: {shift:.1f} Hz)")
