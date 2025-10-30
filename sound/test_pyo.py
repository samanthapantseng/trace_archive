#!/usr/bin/env python3
"""
Minimal pyo test to verify audio output is working
"""

from pyo import *
import time

# List available audio devices
print("=== Available Audio Devices ===")
try:
    pa_list_devices()
except:
    print("Could not list devices")
print("================================\n")

# Create server with output device 3
s = Server(sr=44100, nchnls=2, duplex=0)
s.setOutputDevice(3)
s.boot()
s.start()

print(f"Using audio backend: {s.getServerID()}")
print(f"Using output device: 3")
print(f"Sample rate: {s.getSamplingRate()}")
print(f"Number of channels: {s.getNchnls()}")
print(f"Buffer size: {s.getBufferSize()}")
print()

print("Playing 440Hz sine wave for 3 seconds...")
sine = Sine(freq=440, mul=0.3).out()
time.sleep(30)

print("Playing 220Hz sine wave for 3 seconds...")
sine.setFreq(220)
time.sleep(3)

print("Stopping...")
sine.stop()
s.stop()
s.shutdown()

print("Done!")
