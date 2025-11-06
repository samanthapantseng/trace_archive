# trace-archive


# Bodysynth Sound System

A modular real-time music generation system that uses body tracking data from senseSpace to create interactive audio experiences.

## Overview

The bodysynth system combines multiple audio components:
- **Drum Sequencer**: Pattern-based drum machine with kick, snare, and hi-hat
- **Wavetable Synthesizer**: Generates tones from arm geometry with LFO modulation
- **LLM Singer**: AI-powered singing using Bark TTS with pitch correction
- **OSC Sampler**: Gesture-triggered audio sample playback
- **GUI**: Real-time visualization and parameter control

## Requirements

### Python Dependencies
```bash
# Core audio
pip install pyo

# GUI
pip install PyQt6

# LLM Singer (optional)
pip install transformers torch scipy
pip install TTS  # For Bark model
```

### Audio Setup
The system should be configured to use the default audio device (in my case 10) (line 18 in `bodysynth_modular.py`). To find your default audio device:
```python
from pyo import *
s = Server()
s.printDevices()
```

Update `AUDIO_DEVICE` in `bodysynth_modular.py` to match your output device.

### Sample Files
Place audio samples in the `samples/` directory:
- `samples/piano.wav` - Piano sound (HEAD_BUMP gesture)
- `samples/bass.wav` - Bass sound (ARM_STRETCH gesture)
- `samples/wood.wav` - Wood sound (ARM_SPREAD gesture)
- `samples/water.wav` - Water sound (HANDSHAKE gesture)
- `samples/glass.wav` - Glass sound (HIGH_FIVE gesture)

These mappings can be configured in `bodysynth_sampler_osc.py`

## Running the System

### Basic Usage

Run with standard components enabled:
```bash
python bodysynth_modular.py --wave --osc --gui
```

### Component Selection

Available flags:
- `--drum` - Enable drum sequencer
- `--wave` - Enable wavetable synthesizer
- `--llm` - Enable LLM singer
- `--osc` - Enable OSC sampler
- `--gui` - Enable GUI visualization
- `--all` - Enable all components
- `--server` - Set the server adress, default 192.168.1.4
- `--rec` - Enables reading senseSpace data from recording, specify path after flag
- `--port` - Set the port, default 12345


- **Prompt**: Text prompt for lyric generation
- **Play Interval**: Seconds between phrase playback
- **Auto-tune**: Pitch correction amount (0-100%)
- **Reverb**: Reverb wetness (0-100%)
- **Transpose**: Pitch shift in semitones (-12 to +12)
- **Gain**: Overall singer volume (0-100%)


## Keyboard Shortcuts

- **Ctrl+C**: Exit

## Settings

Settings are automatically saved to `bodysynth_settings.json` when changed in the GUI.

