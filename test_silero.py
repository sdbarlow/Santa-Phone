#!/usr/bin/env python3
"""
Silero VAD Diagnostic Tool
Tests if Silero is working correctly with your microphone.
"""

import pyaudio
import numpy as np
import torch
from silero_vad import load_silero_vad
import os

# Suppress ALSA warnings
os.environ['ALSA_CARD'] = 'Device'
from ctypes import *
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
try:
    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except:
    pass

print("🔄 Loading Silero VAD...")
model = load_silero_vad()
print("✅ Silero loaded!\n")

# Test 1: Check model with synthetic speech-like signal
print("=" * 50)
print("TEST 1: Synthetic signal test")
print("=" * 50)

# Create a simple sine wave (not speech, should be low probability)
t = np.linspace(0, 512/16000, 512)
sine_wave = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440Hz tone
prob = model(torch.from_numpy(sine_wave), 16000).item()
print(f"440Hz sine wave: {prob:.4f} (should be LOW, <0.3)")

model.reset_states()

# Create noise
noise = np.random.randn(512).astype(np.float32) * 0.1
prob = model(torch.from_numpy(noise), 16000).item()
print(f"Random noise: {prob:.4f} (should be LOW, <0.3)")

model.reset_states()

# Test 2: Live microphone test
print("\n" + "=" * 50)
print("TEST 2: Live microphone test")
print("=" * 50)
print("Speak into the microphone for 10 seconds...")
print("Say 'Hello Santa' or count '1, 2, 3, 4, 5'\n")

# Two approaches - try native 16kHz first, then fallback to 44.1kHz with resample

def test_native_16khz():
    """Try recording directly at 16kHz"""
    print("Attempting native 16kHz recording...")
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=1,
            frames_per_buffer=512
        )
        
        model.reset_states()
        
        for i in range(150):  # ~5 seconds at 16kHz
            chunk = stream.read(512, exception_on_overflow=False)
            audio_int16 = np.frombuffer(chunk, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            
            volume = np.sqrt(np.mean(audio_int16.astype(np.float64)**2))
            
            prob = model(torch.from_numpy(audio_float), 16000).item()
            
            bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
            print(f"\rVol: {int(volume):5d} | Silero: [{bar}] {prob:.2f}", end="", flush=True)
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        return True
        
    except Exception as e:
        print(f"\n❌ Native 16kHz failed: {e}")
        return False


def test_44khz_with_resample():
    """Record at 44.1kHz and resample (like santa_phone.py does)"""
    print("\nTesting 44.1kHz with resampling...")
    
    MIC_SAMPLE_RATE = 44100
    MIC_CHUNK_SIZE = int(512 * MIC_SAMPLE_RATE / 16000)  # ~1411 samples
    
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=MIC_SAMPLE_RATE,
            input=True,
            input_device_index=1,
            frames_per_buffer=MIC_CHUNK_SIZE
        )
        
        model.reset_states()
        
        for i in range(150):  # ~5 seconds
            chunk = stream.read(MIC_CHUNK_SIZE, exception_on_overflow=False)
            audio_int16 = np.frombuffer(chunk, dtype=np.int16)
            
            volume = np.sqrt(np.mean(audio_int16.astype(np.float64)**2))
            
            # Resample to 16kHz (same method as santa_phone.py)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            indices = np.linspace(0, len(audio_float32) - 1, 512)
            audio_resampled = np.interp(indices, np.arange(len(audio_float32)), audio_float32)
            
            prob = model(torch.from_numpy(audio_resampled.astype(np.float32)), 16000).item()
            
            bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
            print(f"\rVol: {int(volume):5d} | Silero: [{bar}] {prob:.2f}", end="", flush=True)
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        return True
        
    except Exception as e:
        print(f"\n❌ 44.1kHz test failed: {e}")
        return False


def test_with_proper_resample():
    """Use scipy for proper resampling"""
    print("\nTesting with scipy.signal.resample...")
    
    try:
        from scipy.signal import resample
    except ImportError:
        print("❌ scipy not installed, skipping this test")
        return False
    
    MIC_SAMPLE_RATE = 44100
    MIC_CHUNK_SIZE = int(512 * MIC_SAMPLE_RATE / 16000)
    
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=MIC_SAMPLE_RATE,
            input=True,
            input_device_index=1,
            frames_per_buffer=MIC_CHUNK_SIZE
        )
        
        model.reset_states()
        
        for i in range(150):
            chunk = stream.read(MIC_CHUNK_SIZE, exception_on_overflow=False)
            audio_int16 = np.frombuffer(chunk, dtype=np.int16)
            
            volume = np.sqrt(np.mean(audio_int16.astype(np.float64)**2))
            
            # Proper resample with scipy
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            audio_resampled = resample(audio_float32, 512)
            
            prob = model(torch.from_numpy(audio_resampled.astype(np.float32)), 16000).item()
            
            bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
            print(f"\rVol: {int(volume):5d} | Silero: [{bar}] {prob:.2f}", end="", flush=True)
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        return True
        
    except Exception as e:
        print(f"\n❌ scipy resample test failed: {e}")
        return False


# Run tests
print("\n--- Method 1: Native 16kHz ---")
if not test_native_16khz():
    print("\n--- Method 2: 44.1kHz with np.interp (current method) ---")
    test_44khz_with_resample()
    
    print("\n--- Method 3: 44.1kHz with scipy.signal.resample ---")
    test_with_proper_resample()

print("\n\n" + "=" * 50)
print("DIAGNOSIS")
print("=" * 50)
print("""
If Silero scores stayed near 0 even while you were speaking loudly:
1. The resampling might be corrupting the audio
2. Try using native 16kHz if your mic supports it
3. The mic might be picking up distorted audio

If Silero worked correctly (scores went up when speaking):
1. The issue is in santa_phone.py's state management
2. Silero state might not be resetting properly
""")