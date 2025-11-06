"""
Settings persistence for bodysynth modular system
Saves and loads all GUI parameters to/from a JSON file
"""

import json
import os
from typing import Dict, Any

DEFAULT_SETTINGS_FILE = "bodysynth_settings.json"

# Default values for all parameters
DEFAULT_SETTINGS = {
    # Global
    "loop_length": 2.4,
    "scale": "Dorian",
    "click_enabled": False,
    
    # Drum parameters
    "drum_gain": 0.8,
    "drum_distortion": 1.0,
    "drum_compression": 1.0,
    
    # Wave parameters
    "wave_gain": 0.3,
    "wave_distortion": 0.0,
    "wave_smoothing": 0.0,  # 0.0 = no smoothing (poly order 12), 1.0 = max smoothing (poly order 4)
    "wave_lfo_amount": 0.2,
    "wave_min_reverb": 0.2,
    "wave_max_armlen": 2400.0,
    "wave_min_armlen": 450.0,
    "wave_octave_high": 2.0,   # octaves above root for short arms
    "wave_octave_low": -2.0,   # octaves below root for long arms
    
    # LLM Singer parameters
    "llm_gain": 0.6,
    "llm_prompt": "You are an alchemist trying to find a dance to create a bad spell on a kingdom. Describe what you see in poetic, mystical terms in one short sentence.",
    "llm_interval": 15.0,
    "llm_autotune": 0.5,
    "llm_reverb": 0.7,
    "llm_transpose": 0.0,
}


class SettingsManager:
    """Manages loading and saving of bodysynth settings"""
    
    def __init__(self, settings_file=None):
        """
        Args:
            settings_file: Path to settings JSON file (default: bodysynth_settings.json in current dir)
        """
        if settings_file is None:
            # Use the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            settings_file = os.path.join(script_dir, DEFAULT_SETTINGS_FILE)
        
        self.settings_file = settings_file
        self.settings = DEFAULT_SETTINGS.copy()
    
    def load(self) -> Dict[str, Any]:
        """Load settings from file, falling back to defaults if file doesn't exist
        
        Returns:
            Dict of settings
        """
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    loaded_settings = json.load(f)
                    # Merge with defaults (in case new settings were added)
                    self.settings = DEFAULT_SETTINGS.copy()
                    self.settings.update(loaded_settings)
                    print(f"[Settings] Loaded from {self.settings_file}")
            except Exception as e:
                print(f"[Settings] Error loading from {self.settings_file}: {e}")
                print(f"[Settings] Using defaults")
                self.settings = DEFAULT_SETTINGS.copy()
        else:
            print(f"[Settings] No settings file found, using defaults")
            self.settings = DEFAULT_SETTINGS.copy()
        
        return self.settings.copy()
    
    def save(self, settings: Dict[str, Any] = None):
        """Save settings to file
        
        Args:
            settings: Dict of settings to save (if None, saves current settings)
        """
        if settings is not None:
            self.settings = settings
        
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
            print(f"[Settings] Saved to {self.settings_file}")
        except Exception as e:
            print(f"[Settings] Error saving to {self.settings_file}: {e}")
    
    def get(self, key: str, default=None) -> Any:
        """Get a setting value
        
        Args:
            key: Setting key
            default: Default value if key not found
        
        Returns:
            Setting value
        """
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a setting value
        
        Args:
            key: Setting key
            value: Setting value
        """
        self.settings[key] = value
    
    def update(self, settings_dict: Dict[str, Any]):
        """Update multiple settings at once
        
        Args:
            settings_dict: Dict of settings to update
        """
        self.settings.update(settings_dict)
