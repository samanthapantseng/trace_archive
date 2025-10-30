"""
Simple drum sample tester - generates and loops samples continuously
for testing sample generation without external triggers or visualizations
"""

import numpy as np
import time
from pyo import *
import wave
import os

# --- Configuration ---
SAMPLE_RATE = 44100
CHANNELS = 2
LOOP_LENGTH = 2.4
NUM_LANES = 3

def generate_click():
    """Generate a sharp click sound for metronome"""
    duration = 0.05  # Very short click
    sample_length = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, sample_length)
    
    # High pitched short beep with fast decay
    freq = 1200  # Hz
    envelope = np.exp(-t / 0.01)  # Very fast decay
    
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


def main():
    # Initialize pyo audio server
    s = Server(sr=SAMPLE_RATE, nchnls=CHANNELS, duplex=0)
    s.setOutputDevice(3)
    s.boot().start()
    print("[INFO] Pyo audio server started on output device 3")
    
    # Generate click sample
    click_sample = generate_click()
    click_table = DataTable(size=len(click_sample), init=click_sample.tolist())
    print(f"[INFO] Generated click: duration={len(click_sample)/SAMPLE_RATE:.3f}s")
    
    # Generate drum samples for each lane
    sample_tables = []
    drum_types = ['kick', 'snare', 'cymbal']
    drum_params = [
        {'base_freq': 60, 'duration': 0.5, 'drum_type': 'kick', 'distortion': 0.2, 'tone': 0.3, 'decay_rate': 1.5},
        {'base_freq': 200, 'duration': 0.3, 'drum_type': 'snare', 'distortion': 0.1, 'tone': 0.6, 'decay_rate': 1.2},
        {'base_freq': 5000, 'duration': 0.8, 'drum_type': 'cymbal', 'distortion': 0.0, 'tone': 0.7, 'decay_rate': 0.7}
    ]
    
    # Create output directory if it doesn't exist
    output_dir = "drum_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    for lane_idx in range(NUM_LANES):
        params = drum_params[lane_idx]
        sample = generate_drum_variant(**params)
        table = DataTable(size=len(sample), init=sample.tolist())
        sample_tables.append(table)
        
        # Save as WAV file
        wav_filename = os.path.join(output_dir, f"{params['drum_type']}_lane{lane_idx}.wav")
        with wave.open(wav_filename, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(SAMPLE_RATE)
            # Convert float32 [-1, 1] to int16 [-32768, 32767]
            sample_int16 = (sample * 32767).astype(np.int16)
            wav_file.writeframes(sample_int16.tobytes())
        
        print(f"[INFO] Lane {lane_idx} ({params['drum_type']}): freq={params['base_freq']:.1f}Hz, "
              f"duration={params['duration']:.2f}s, samples={len(sample)}, saved to {wav_filename}")

    
    # Create loopers for continuous playback
    # Click plays at the start of each loop
    click_looper = Looper(table=click_table, dur=LOOP_LENGTH, mul=0.8).out()
    
    # Each lane plays at different times in the loop
    lane_triggers = []
    lane_start_times = []
    
    # Create a metro that fires at the loop rate
    loop_metro = Metro(time=LOOP_LENGTH).play()
    
    for lane_idx, table in enumerate(sample_tables):
        # Space samples evenly across the loop
        # kick at 0.0s, snare at 0.8s, cymbal at 1.6s
        start_offset = (lane_idx / NUM_LANES) * LOOP_LENGTH
        lane_start_times.append(start_offset)
        
        # Create a delayed metro for this lane
        lane_metro = Metro(time=LOOP_LENGTH, poly=1).play(delay=start_offset)
        
        # Use TrigEnv to play the sample when triggered
        sample_player = TrigEnv(lane_metro, table=table, dur=drum_params[lane_idx]['duration'], mul=0.5)
        sample_player.out()
        lane_triggers.append(sample_player)
        
        print(f"[INFO] Lane {lane_idx} ({drum_params[lane_idx]['drum_type']}) triggers at {start_offset:.2f}s in each {LOOP_LENGTH}s loop")
    
    # Monitoring thread to print when samples play
    import threading
    stop_monitoring = threading.Event()
    
    def monitor_playback():
        start_time = time.time()
        loop_count = 0
        last_loop_time = start_time
        
        while not stop_monitoring.is_set():
            current_time = time.time()
            elapsed = current_time - start_time
            loop_position = elapsed % LOOP_LENGTH
            
            # Check for new loop
            if current_time - last_loop_time >= LOOP_LENGTH:
                loop_count += 1
                print(f"\n=== LOOP {loop_count} START (click) ===")
                last_loop_time = current_time
            
            # Check if any sample should be playing
            for lane_idx, lane_time in enumerate(lane_start_times):
                # Check if we just crossed this lane's trigger point
                prev_position = (elapsed - 0.1) % LOOP_LENGTH
                if prev_position > loop_position or (loop_position >= lane_time and prev_position < lane_time):
                    if abs(loop_position - lane_time) < 0.15:  # Within small window
                        drum_name = drum_params[lane_idx]['drum_type']
                        print(f"  → Playing {drum_name} (lane {lane_idx}) at t={loop_position:.2f}s")
            
            time.sleep(0.1)
    
    monitor_thread = threading.Thread(target=monitor_playback, daemon=True)
    monitor_thread.start()
    
    print(f"\n[INFO] Playing {NUM_LANES} drum samples in a {LOOP_LENGTH}s loop")
    print("[INFO] Click plays at loop start")
    print("[INFO] Press Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('\n[INFO] Stopping...')
    finally:
        stop_monitoring.set()
        monitor_thread.join(timeout=1.0)
        s.stop()
        s.shutdown()


if __name__ == '__main__':
    main()
