#!/usr/bin/env python3
"""
Ambient Noise Calibration Tool for Santa Phone
Measures volume levels for 5 minutes and recommends threshold settings.
"""

import pyaudio
import numpy as np
import time
import os
from collections import deque

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

# Audio config (match santa_phone.py)
MIC_SAMPLE_RATE = 44100
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK_SIZE = 1411  # Same as santa_phone.py

# Calibration duration
CALIBRATION_DURATION_SECONDS = 5 * 60  # 5 minutes


def measure_ambient_noise():
    print("🎤 Ambient Noise Calibration Tool")
    print("=" * 50)
    print(f"Duration: {CALIBRATION_DURATION_SECONDS // 60} minutes")
    print("\n⚠️  For accurate results:")
    print("   - Place mic in final position")
    print("   - Keep the environment as it will be during use")
    print("   - Don't speak directly into the mic")
    print("=" * 50)
    print("\nStarting in 3 seconds...\n")
    time.sleep(3)
    
    audio = pyaudio.PyAudio()
    
    # Try to find the right input device
    device_index = None
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"Found input device [{i}]: {info['name']}")
            if device_index is None:
                device_index = i
    
    # Use device index 1 like santa_phone.py, or fallback
    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=MIC_SAMPLE_RATE,
            input=True,
            input_device_index=1,
            frames_per_buffer=CHUNK_SIZE
        )
        print(f"\n✅ Using device index 1")
    except:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=MIC_SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK_SIZE
        )
        print(f"\n✅ Using device index {device_index}")
    
    print(f"📊 Recording for {CALIBRATION_DURATION_SECONDS} seconds...\n")
    
    # Storage for measurements
    all_volumes = []
    recent_volumes = deque(maxlen=100)  # For rolling display
    
    start_time = time.time()
    last_print_time = start_time
    samples_collected = 0
    
    # Track peaks
    peak_volume = 0
    peak_time = 0
    
    try:
        while True:
            elapsed = time.time() - start_time
            remaining = CALIBRATION_DURATION_SECONDS - elapsed
            
            if remaining <= 0:
                break
            
            # Read audio chunk
            chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            
            # Calculate RMS volume
            audio_int16 = np.frombuffer(chunk, dtype=np.int16)
            audio_float = audio_int16.astype(np.float64)
            volume = np.sqrt(np.mean(audio_float**2))
            
            if not np.isfinite(volume):
                volume = 0
            
            all_volumes.append(volume)
            recent_volumes.append(volume)
            samples_collected += 1
            
            # Track peak
            if volume > peak_volume:
                peak_volume = volume
                peak_time = elapsed
            
            # Print status every second
            if time.time() - last_print_time >= 1.0:
                recent_avg = np.mean(recent_volumes) if recent_volumes else 0
                recent_max = max(recent_volumes) if recent_volumes else 0
                
                mins_remaining = int(remaining // 60)
                secs_remaining = int(remaining % 60)
                
                # Progress bar
                progress = elapsed / CALIBRATION_DURATION_SECONDS
                bar_width = 30
                filled = int(bar_width * progress)
                bar = "█" * filled + "░" * (bar_width - filled)
                
                print(f"\r[{bar}] {mins_remaining}:{secs_remaining:02d} remaining | "
                      f"Current: {int(volume):5d} | "
                      f"Avg: {int(recent_avg):5d} | "
                      f"Peak: {int(recent_max):5d}", end="", flush=True)
                
                last_print_time = time.time()
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Calibration interrupted early!")
    
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
    
    # Calculate statistics
    print("\n\n" + "=" * 50)
    print("📊 CALIBRATION RESULTS")
    print("=" * 50)
    
    if len(all_volumes) < 100:
        print("❌ Not enough samples collected!")
        return
    
    volumes_array = np.array(all_volumes)
    
    avg_volume = np.mean(volumes_array)
    median_volume = np.median(volumes_array)
    std_volume = np.std(volumes_array)
    min_volume = np.min(volumes_array)
    max_volume = np.max(volumes_array)
    
    # Percentiles
    p50 = np.percentile(volumes_array, 50)
    p75 = np.percentile(volumes_array, 75)
    p90 = np.percentile(volumes_array, 90)
    p95 = np.percentile(volumes_array, 95)
    p99 = np.percentile(volumes_array, 99)
    
    print(f"\nSamples collected: {len(all_volumes):,}")
    print(f"Duration: {elapsed:.1f} seconds")
    
    print(f"\n📈 Volume Statistics:")
    print(f"   Average:    {avg_volume:,.0f}")
    print(f"   Median:     {median_volume:,.0f}")
    print(f"   Std Dev:    {std_volume:,.0f}")
    print(f"   Min:        {min_volume:,.0f}")
    print(f"   Max:        {max_volume:,.0f} (at {peak_time:.1f}s)")
    
    print(f"\n📊 Percentiles:")
    print(f"   50th (median): {p50:,.0f}")
    print(f"   75th:          {p75:,.0f}")
    print(f"   90th:          {p90:,.0f}")
    print(f"   95th:          {p95:,.0f}")
    print(f"   99th:          {p99:,.0f}")
    
    # Calculate recommended thresholds
    # VOLUME_THRESHOLD: Should be above 95th percentile of ambient noise
    # VOLUME_THRESHOLD_LOW: Should be above median but catch quiet speech
    # VOLUME_THRESHOLD_TRIGGERED: Can be lower to maintain capture
    
    recommended_main = max(int(p95 * 2), int(avg_volume * 2.5))
    recommended_low = max(int(p90 * 1.5), int(avg_volume * 1.8))
    recommended_triggered = max(int(p75), int(avg_volume * 0.8))
    
    # Ensure proper ordering
    if recommended_low >= recommended_main:
        recommended_low = int(recommended_main * 0.7)
    if recommended_triggered >= recommended_low:
        recommended_triggered = int(recommended_low * 0.6)
    
    print(f"\n" + "=" * 50)
    print("🎯 RECOMMENDED THRESHOLDS")
    print("=" * 50)
    print(f"\nBased on your ambient noise levels:\n")
    print(f"VOLUME_THRESHOLD = {recommended_main}        # Main trigger (2x p95)")
    print(f"VOLUME_THRESHOLD_LOW = {recommended_low}      # 'Speak up' zone")
    print(f"VOLUME_THRESHOLD_TRIGGERED = {recommended_triggered}   # Continue capture")
    
    print(f"\n📋 Copy this to santa_phone.py:")
    print("-" * 40)
    print(f"VOLUME_THRESHOLD = {recommended_main}")
    print(f"VOLUME_THRESHOLD_LOW = {recommended_low}")
    print(f"VOLUME_THRESHOLD_TRIGGERED = {recommended_triggered}")
    print("-" * 40)
    
    # Warnings
    print(f"\n⚠️  Notes:")
    if avg_volume > 2000:
        print(f"   - High ambient noise detected ({avg_volume:.0f} avg)")
        print(f"   - Consider a quieter location or directional mic")
    if std_volume > avg_volume:
        print(f"   - High variance in noise levels")
        print(f"   - Environment may have intermittent loud sounds")
    if max_volume > avg_volume * 5:
        print(f"   - Detected loud spikes (max {max_volume:.0f} vs avg {avg_volume:.0f})")
        print(f"   - May cause false triggers")
    
    print(f"\n✅ Calibration complete!")


if __name__ == "__main__":
    measure_ambient_noise()