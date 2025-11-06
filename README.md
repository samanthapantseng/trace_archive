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

#### Quick Install (Core Components)
```bash
pip install -r requirements.txt
```


#### senseSpaceLib
The system requires `senseSpaceLib` from the parent repository. Install from the repository root:
```bash
cd /path/to/senseSpace/libs/senseSpaceLib
pip install -e .
```


### Audio Setup
The system should be configured to use the default audio device (in my case 10) (line 18 in `bodysynth_modular.py`). To find your default audio device:
```python
from pyo import *
pa_get_devices_infos()
```

Update `AUDIO_DEVICE` in `bodysynth_modular.py` to match your output device.

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


### OSC Sampler

The OSC sampler receives gesture data from the gesture part over OSC on port 8000.

Whenerver a message on a specific topic arrives it playse the corresponding sample from the `samples/` folder:

Current config
- HEAD_BUMP: `samples/piano.wav`
- ARM_STRETCH: `samples/bass.wav`
- ARM_SPREAD: `samples/wood.wav`
- HANDSHAKE: `samples/water.wav`
- HIGH_FIVE: `samples/glass.wav`

These mappings can be configured in `bodysynth_sampler_osc.py`


## Keyboard Shortcuts

- **Ctrl+C**: Exit

## Settings

Settings are automatically saved to `bodysynth_settings.json` when changed in the GUI.




### External Dependencies

#### For LLM Singer

**Option 1: Piper TTS + Ollama** (`bodysynth_llm_singer.py`)
- **Piper TTS**: Download and install from https://github.com/rhasspy/piper
  - Requires `piper` binary in PATH
  - Model: `en_US-danny-low` (downloads automatically on first run)
- **Ollama**: Install and run with `llama3.2:1b` model
  ```bash
  # Install Ollama from https://ollama.ai
  ollama pull llama3.2:1b
  ollama run llama3.2:1b
  ```

**Option 2: Bark TTS + Ollama** (`bodysynth_llm_singer_bark.py`)
- Install Bark and dependencies (see requirements.txt)
- Ollama setup same as above
- First run downloads Bark models automatically

**Option 3: HuggingFace API** (`bodysynth_llm_singer_hf.py`)
- Requires HuggingFace account and API token
- Get free token from: https://huggingface.co/settings/tokens
- No local models needed (uses cloud API)


#### Manual Installation
```bash
# Core audio engine
pip install pyo>=1.0.5

# GUI framework
pip install PyQt6>=6.6.0
pip install pyqtgraph>=0.13.3

# Scientific computing
pip install numpy>=1.24.0

# OSC communication (for gesture-triggered samples)
pip install python-osc>=1.8.3

# LLM Singer - Bark Version (optional, only needed for --llm flag)
pip install git+https://github.com/suno-ai/bark.git
pip install transformers>=4.35.0
pip install torch>=2.1.0
pip install scipy>=1.10.0

# LLM Singer - HuggingFace API Version (alternative to Bark)
pip install huggingface-hub>=0.19.0
```