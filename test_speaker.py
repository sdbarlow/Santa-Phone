import pyaudio
from silero_vad import load_silero_vad
import numpy as np
import torch
import wave
from collections import deque
import os
import warnings
import signal
import subprocess
import time
from elevenlabs.client import ElevenLabs
from groq import Groq
import threading
import psutil
from dotenv import load_dotenv

load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global variables for audio playback control
current_playback_process = None
playback_lock = threading.Lock()

conversation_history = []
conversation_lock = threading.Lock()
MAX_HISTORY_MESSAGES = 20

listening_active = True

# Signal handler
def signal_handler(sig, frame):
    print("\n\n👋 Force stopping...")
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)

os.environ['ALSA_CARD'] = 'Device'
from ctypes import *
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
asound = cdll.LoadLibrary('libasound.so.2')
asound.snd_lib_error_set_handler(c_error_handler)

# Configuration
MIC_SAMPLE_RATE = 44100
SILERO_SAMPLE_RATE = 16000
VAD_CHUNK_SIZE = 512

MIC_CHUNK_SIZE = int(VAD_CHUNK_SIZE * MIC_SAMPLE_RATE / SILERO_SAMPLE_RATE)

FORMAT = pyaudio.paInt16
CHANNELS = 1

# Silero VAD settings
SILERO_THRESHOLD = 0.90
VOLUME_THRESHOLD = 1244       # Must be clearly louder than ambient
VOLUME_THRESHOLD_LOW = 763   # Wide gap for "speak up" zone
VOLUME_THRESHOLD_TRIGGERED = 391  # Keep capturing through brief quiet moments
SILENCE_DURATION_MS = 350
SILENCE_CHUNKS = int((SILENCE_DURATION_MS / 1000) * SILERO_SAMPLE_RATE / VAD_CHUNK_SIZE)

# Pre-recorded messages
SPEAK_UP_REMINDER = "speak_up_reminder.mp3"  # Santa saying "speak loud and clearly"

print("🔄 Initializing Silero VAD...")
silero_model = load_silero_vad()
print("✅ Silero VAD ready!\n")

print("🔄 Initializing ElevenLabs...")
elevenlabs_client = ElevenLabs(
    api_key=os.environ.get("ELEVENLABS_API_KEY", "your-key-here")
)
print("✅ ElevenLabs ready!")

print("🔄 Initializing Groq Whisper...")
groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY", "your-key-here")
)
print("✅ Groq ready!\n")

SANTA_VOICE_ID = "Gqe8GJJLg3haJkTwYj2L"

SANTA_SYSTEM_PROMPT = '''You are Santa Claus speaking to children at a toy store during the Christmas season. Keep responses to EXACTLY 1 sentence - never more.

IMPORTANT CONTEXT:
- A pre-recorded greeting ("Ho ho ho! Merry Christmas! Who do we have here?") has ALREADY played before the child speaks
- A thinking sound like "Hmm..." or "Well now..." may have ALREADY played before your response
- Do NOT repeat greetings or thinking phrases - go directly into your response
- Never start with "Ho ho ho", "Well", "Hmm", "Let me see", or similar - those are already covered

Personality:
- Warm, jolly, and genuinely interested in each child
- Speak naturally like a kind grandfather, not overly formal
- Remember details children share (names, wishes, siblings, pets)

Guidelines:
- Ask about their Christmas wishes, how they've been good, siblings, pets, or school
- If they tell you their name, use it warmly in your response
- If they ask for something, respond warmly but don't make specific promises
- Keep the magic alive - reference the North Pole, elves, reindeer, Mrs. Claus naturally
- Be encouraging about good behavior without being preachy
- If they seem shy, be extra gentle and patient

Remember: ONE sentence only. No greeting prefixes. Make every word count.'''

max_tokens=50 

SANTA_OPENERS = [
    "opener_ho_ho_ho.mp3",
    "opener_oh_ho_ho.mp3",
]

# Thinking sounds - played while LLM is processing
# These are neutral "acknowledgment" sounds, not full phrases
# The LLM is told these already played, so it won't repeat them
THINKING_SOUNDS = [
    "thinking_ho_ho_ho.mp3",
]

import random

# Global state tracking
current_state = {
    'transcribing': False,
    'llm_thinking': False,
    'tts_generating': False,  # Track when TTS is generating audio
    'playing_audio': False,
    'was_interrupted': False,  # Track if we interrupted playback
    'last_transcription': '',
    'last_transcription_time': 0,
    'interrupted': False,
    'last_activity_time': time.time(),  # Track last speech activity
    'audio_playing_at_trigger': False,  # Was audio playing when speech triggered?
    'playback_end_time': 0,  # When did playback end?
    'stop_thinking_sound': False,  # Signal to stop thinking sound
    'reset_silero': False,  # Signal VAD to reset Silero state
}
state_lock = threading.Lock()

# Echo detection settings
ECHO_DETECTION_WINDOW = 0.5  # Seconds after playback ends to suspect echo
MIN_REAL_SPEECH_FRAMES = 30  # Minimum frames for real speech (not echo)

INTERRUPTION_MERGE_WINDOW = 2.0  # Merge if interrupted within 2 seconds
CONVERSATION_TIMEOUT = 45.0  # Clear conversation after 45 seconds of silence

import queue

# Global queue for audio recordings
audio_queue = queue.Queue()

# Thinking sound process (separate from main playback)
thinking_sound_process = None
thinking_sound_lock = threading.Lock()


def play_thinking_sound():
    """Play a random thinking sound while LLM processes - runs in background thread"""
    global thinking_sound_process
    
    # Find available thinking sounds
    available_sounds = [s for s in THINKING_SOUNDS if os.path.exists(s)]
    
    if not available_sounds:
        print("⚠️ No thinking sounds found, skipping...")
        return
    
    # Pick a random one
    sound_file = random.choice(available_sounds)
    print(f"🤔 Playing thinking sound: {sound_file}")
    
    with thinking_sound_lock:
        # Check if we should stop before starting
        with state_lock:
            if current_state['stop_thinking_sound'] or current_state['interrupted']:
                print("🤔 Thinking sound cancelled before start")
                return
        
        try:
            thinking_sound_process = subprocess.Popen(
                ['ffplay', '-nodisp', '-autoexit', '-volume', '8', sound_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for it to finish or be stopped
            thinking_sound_process.wait()
            
        except Exception as e:
            print(f"⚠️ Thinking sound error: {e}")
        finally:
            thinking_sound_process = None


def stop_thinking_sound():
    """Stop the thinking sound if it's playing"""
    global thinking_sound_process
    
    with thinking_sound_lock:
        if thinking_sound_process and thinking_sound_process.poll() is None:
            print("🤔 Stopping thinking sound")
            try:
                thinking_sound_process.kill()
            except:
                pass
            thinking_sound_process = None


def play_speak_up_reminder():
    """Play pre-recorded reminder to speak louder"""
    global current_playback_process
    
    if not os.path.exists(SPEAK_UP_REMINDER):
        print(f"⚠️ Speak up reminder file not found: {SPEAK_UP_REMINDER}")
        return
    
    print("🔊 Playing speak up reminder...")
    
    with playback_lock:
        # Stop any current playback first
        if current_playback_process and current_playback_process.poll() is None:
            try:
                current_playback_process.kill()
            except:
                pass
        
        current_playback_process = subprocess.Popen(
            ['ffplay', '-nodisp', '-autoexit', '-volume', '10', SPEAK_UP_REMINDER],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    with state_lock:
        current_state['playing_audio'] = True


def play_opener(wait_for_finish=True):
    """Play a random Santa opener greeting
    
    Args:
        wait_for_finish: If True, wait until audio finishes (but still interruptible)
    """
    global current_playback_process
    
    # Find available openers
    available_openers = [s for s in SANTA_OPENERS if os.path.exists(s)]
    
    if not available_openers:
        print("⚠️ No opener audio files found, skipping greeting...")
        return
    
    # Pick a random opener
    opener_file = random.choice(available_openers)
    print(f"🎅 Playing opener: {opener_file}")
    
    with playback_lock:
        # Stop any current playback first
        if current_playback_process and current_playback_process.poll() is None:
            try:
                current_playback_process.kill()
            except:
                pass
        
        current_playback_process = subprocess.Popen(
            ['ffplay', '-nodisp', '-autoexit', '-volume', '10', opener_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    with state_lock:
        current_state['playing_audio'] = True
        current_state['last_activity_time'] = time.time()
    
    if wait_for_finish:
        # Poll for completion instead of blocking - allows interruption
        while current_playback_process and current_playback_process.poll() is None:
            # Check if we were interrupted
            with state_lock:
                if current_state['interrupted']:
                    print("🎅 Opener interrupted!")
                    break
            time.sleep(0.05)  # Small sleep to avoid busy-waiting
        
        with state_lock:
            current_state['playing_audio'] = False
            current_state['playback_end_time'] = time.time()
            current_state['reset_silero'] = True  # Reset Silero after playback
        
        print("🎅 Opener finished, ready for child to speak!")


def downsample_for_vad(chunk, from_rate, to_rate, target_samples=512):
    """Downsample audio chunk and ensure exactly target_samples output"""
    audio_int16 = np.frombuffer(chunk, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    
    num_samples_original = len(audio_float32)
    indices = np.linspace(0, num_samples_original - 1, target_samples)
    audio_resampled = np.interp(indices, np.arange(num_samples_original), audio_float32)
    
    assert len(audio_resampled) == target_samples, f"Expected {target_samples} samples, got {len(audio_resampled)}"
    
    return audio_resampled


def stop_all_audio(clear_state=True):
    """Stop any currently playing audio
    
    Args:
        clear_state: If False, caller is responsible for clearing playing_audio flag
                    (use when caller already holds state_lock)
    """
    global current_playback_process
    
    print(f"\n🔍 [stop_all_audio] ENTRY: current_playback_process={current_playback_process}", flush=True)
    print(f"🔍 [stop_all_audio] Thread ID: {threading.current_thread().ident}", flush=True)
    print(f"🔍 [stop_all_audio] Acquiring playback_lock...", flush=True)
    
    with playback_lock:
        print(f"🔍 [stop_all_audio] Lock acquired!", flush=True)
        print(f"🔍 [stop_all_audio] process={current_playback_process}, poll={current_playback_process.poll() if current_playback_process else 'N/A'}", flush=True)
        
        if current_playback_process and current_playback_process.poll() is None:
            print("🛑 [stop_all_audio] Stopping current audio playback...", flush=True)
            try:
                parent = psutil.Process(current_playback_process.pid)
                children = parent.children(recursive=True)
                print(f"🔪 [stop_all_audio] Killing parent PID {parent.pid} and {len(children)} children", flush=True)
                
                for child in children:
                    print(f"🔪 [stop_all_audio] Killing child PID {child.pid}", flush=True)
                    child.kill()
                
                parent.kill()
                print("✅ [stop_all_audio] Processes killed", flush=True)
            except (psutil.NoSuchProcess, ProcessLookupError) as e:
                print(f"⚠️ [stop_all_audio] Process lookup error: {e}", flush=True)
            
            current_playback_process = None
            print("✅ [stop_all_audio] current_playback_process set to None", flush=True)
        else:
            print(f"⚠️ [stop_all_audio] No active process to stop (exists: {current_playback_process is not None}, running: {current_playback_process.poll() is None if current_playback_process else 'N/A'})", flush=True)
    
    # Clear playing_audio flag when audio is stopped (only if caller doesn't hold state_lock)
    if clear_state:
        with state_lock:
            current_state['playing_audio'] = False
            print("✅ [stop_all_audio] playing_audio flag cleared", flush=True)
    else:
        print("✅ [stop_all_audio] Skipping state clear (caller holds lock)", flush=True)
    
    print(f"🔍 [stop_all_audio] Lock released, exiting", flush=True)


def continuous_vad_loop():
    """Continuously listen for speech and queue recordings"""
    print("🎤 Starting continuous VAD listener...")
    
    while True:
        print("\n🔄 Resetting Silero state...")
        silero_model.reset_states()
        
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=MIC_SAMPLE_RATE,
            input=True,
            input_device_index=1,
            frames_per_buffer=MIC_CHUNK_SIZE
        )
        
        triggered = False
        voiced_frames = []
        ring_buffer = deque(maxlen=10)
        silence_count = 0
        chunk_count = 0
        consecutive_quiet_chunks = 0  # Track quiet periods to reset Silero
        last_silero_reset = time.time()
        
        SILERO_RESET_INTERVAL = 5.0  # Reset every 5 seconds of non-triggered listening
        QUIET_CHUNKS_FOR_RESET = 100  # Reset after ~3 seconds of quiet
        
        print("👂 Listening...")
        
        while True:
            chunk_count += 1
            chunk = stream.read(MIC_CHUNK_SIZE, exception_on_overflow=False)
            
            # Check if we need to reset Silero (after TTS playback)
            with state_lock:
                if current_state['reset_silero']:
                    current_state['reset_silero'] = False
                    print("\n🔄 Resetting Silero state (post-playback)...")
                    silero_model.reset_states()
                    last_silero_reset = time.time()
            
            # Calculate volume
            audio_int16 = np.frombuffer(chunk, dtype=np.int16)
            audio_float = audio_int16.astype(np.float64)
            volume = np.sqrt(np.mean(audio_float**2))
            
            if not np.isfinite(volume):
                volume = 0
            
            # Track quiet periods and reset Silero periodically when not triggered
            if not triggered:
                if volume < VOLUME_THRESHOLD_LOW:
                    consecutive_quiet_chunks += 1
                else:
                    consecutive_quiet_chunks = 0
                
                # Reset Silero if it's been quiet for a while (state might be stale)
                if consecutive_quiet_chunks >= QUIET_CHUNKS_FOR_RESET:
                    silero_model.reset_states()
                    consecutive_quiet_chunks = 0
                    last_silero_reset = time.time()
                    # Don't print to avoid spam, but uncomment for debugging:
                    # print("\n🔄 Silero reset (quiet period)")
                
                # Also reset periodically based on time
                if time.time() - last_silero_reset > SILERO_RESET_INTERVAL:
                    silero_model.reset_states()
                    last_silero_reset = time.time()
                    # print("\n🔄 Silero reset (periodic)")
            
            # Print volume every 50 chunks
            if chunk_count % 50 == 0:
                print(f"\r📊 vol={int(volume)}, triggered={triggered}", end="", flush=True)
            
            # Use lower threshold once triggered to keep capturing
            active_volume_threshold = VOLUME_THRESHOLD_TRIGGERED if triggered else VOLUME_THRESHOLD
            
            # Check for low volume speech (between LOW and main threshold) - only when not triggered
            if not triggered and VOLUME_THRESHOLD_LOW <= volume < VOLUME_THRESHOLD:
                # Check if this might be speech using Silero
                audio_resampled = downsample_for_vad(chunk, MIC_SAMPLE_RATE, SILERO_SAMPLE_RATE, VAD_CHUNK_SIZE)
                audio_tensor = torch.from_numpy(audio_resampled.astype(np.float32))
                
                if len(audio_tensor) == VAD_CHUNK_SIZE:
                    speech_prob = silero_model(audio_tensor, SILERO_SAMPLE_RATE).item()
                    if speech_prob > SILERO_THRESHOLD:
                        print(f"\n🔈 Low volume speech detected (vol={int(volume)}, prob={speech_prob:.2f})")
                        print("📢 Playing speak up reminder...")
                        # Play reminder in background thread to not block VAD
                        threading.Thread(target=play_speak_up_reminder, daemon=True).start()
                        # Reset Silero state after playing reminder
                        silero_model.reset_states()
                        # Skip a bit to avoid repeated triggers
                        time.sleep(0.5)
                        continue
            
            # Skip if volume is too low
            if volume < active_volume_threshold:
                if not triggered:
                    continue
                else:
                    silence_count += 1
                    voiced_frames.append(chunk)
                    if silence_count > SILENCE_CHUNKS:
                        print(f"\n✅ Speech ended!")
                        break  # Exit inner loop, save recording
                    continue
            
            # Volume above threshold
            if chunk_count % 10 == 0:  # Only print occasionally to reduce spam
                print(f"\n🔊 VOLUME: {int(volume)}", flush=True)
            
            # Downsample and check Silero
            audio_resampled = downsample_for_vad(chunk, MIC_SAMPLE_RATE, SILERO_SAMPLE_RATE, VAD_CHUNK_SIZE)
            audio_tensor = torch.from_numpy(audio_resampled.astype(np.float32))
            
            if len(audio_tensor) != VAD_CHUNK_SIZE:
                continue
            
            speech_prob = silero_model(audio_tensor, SILERO_SAMPLE_RATE).item()
            is_speech = speech_prob > SILERO_THRESHOLD
            
            if chunk_count % 10 == 0:
                print(f"🎯 Silero: {speech_prob:.2f}, is_speech={is_speech}", flush=True)
            
            if not triggered:
                ring_buffer.append(chunk)
                
                if is_speech:
                    print(f"\n🚨 SPEECH DETECTED!", flush=True)
                    
                    # First, collect state and set flags (quick lock acquire/release)
                    need_to_stop_audio = False
                    need_to_stop_thinking = False
                    should_clear_conversation = False
                    
                    with state_lock:
                        # Check if conversation timed out
                        time_since_last = time.time() - current_state['last_activity_time']
                        if time_since_last > CONVERSATION_TIMEOUT:
                            print(f"⏰ Conversation timed out ({time_since_last:.1f}s), starting fresh!")
                            should_clear_conversation = True
                        
                        # Update activity time
                        current_state['last_activity_time'] = time.time()
                        
                        # Track if audio was playing when we triggered (possible echo)
                        was_playing = current_state['playing_audio'] or current_state['tts_generating']
                        time_since_playback = time.time() - current_state['playback_end_time']
                        current_state['audio_playing_at_trigger'] = was_playing or (time_since_playback < ECHO_DETECTION_WINDOW)
                        
                        if current_state['audio_playing_at_trigger']:
                            print("⚠️ Audio was playing - this might be echo, will verify...")
                        
                        if current_state['playing_audio']:
                            print("🛑 Interrupting audio playback...")
                            current_state['was_interrupted'] = True
                            current_state['playing_audio'] = False
                            need_to_stop_audio = True
                        if current_state['tts_generating']:
                            print("🛑 Interrupting TTS generation...")
                            current_state['was_interrupted'] = True
                            current_state['interrupted'] = True
                        if current_state['transcribing']:
                            print("🛑 Interrupting transcription...")
                            current_state['interrupted'] = True
                        if current_state['llm_thinking']:
                            print("🛑 Interrupting LLM...")
                            current_state['interrupted'] = True
                            current_state['stop_thinking_sound'] = True
                            need_to_stop_thinking = True
                    
                    # Now perform actions OUTSIDE the state_lock to avoid deadlock
                    if should_clear_conversation:
                        with conversation_lock:
                            conversation_history.clear()
                    
                    if need_to_stop_audio:
                        stop_all_audio(clear_state=False)  # Safe now - we don't hold state_lock
                    
                    if need_to_stop_thinking:
                        stop_thinking_sound()
                    
                    triggered = True
                    voiced_frames.extend(ring_buffer)
                    voiced_frames.append(chunk)
            else:
                voiced_frames.append(chunk)
                
                if is_speech:
                    silence_count = 0
                else:
                    silence_count += 1
                
                if silence_count > SILENCE_CHUNKS:
                    print(f"\n✅ Speech ended!")
                    break  # Exit inner loop, save recording
        
        # Package audio data (don't save to disk)
        audio_data = b''.join(voiced_frames)
        sample_width = audio.get_sample_size(FORMAT)
        
        print(f"📦 Captured {len(voiced_frames)} frames")
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Check if this might be echo (audio was playing + short capture)
        with state_lock:
            might_be_echo = current_state['audio_playing_at_trigger']
            current_state['audio_playing_at_trigger'] = False  # Reset for next capture
        
        # Filter out likely echo: short recordings that triggered during playback
        if might_be_echo and len(voiced_frames) < MIN_REAL_SPEECH_FRAMES:
            print(f"🔇 Discarding likely echo ({len(voiced_frames)} frames < {MIN_REAL_SPEECH_FRAMES} minimum)")
            continue  # Skip to next VAD session without queuing
        
        # Add audio data to queue for processing (as tuple with metadata)
        audio_queue.put((audio_data, MIC_SAMPLE_RATE, sample_width, CHANNELS, might_be_echo))
        print(f"📦 Added to queue (queue size: {audio_queue.qsize()})")
        
        # Continue immediately to next VAD session (no waiting!)


def transcribe_audio(audio_tuple):
    """Transcribe audio using Groq Whisper - can be interrupted"""
    print("📝 Transcribing with Groq...")
    
    # Handle both old (4-tuple) and new (5-tuple) format
    if len(audio_tuple) == 5:
        audio_data, sample_rate, sample_width, channels, might_be_echo = audio_tuple
    else:
        audio_data, sample_rate, sample_width, channels = audio_tuple
        might_be_echo = False
    
    with state_lock:
        current_state['transcribing'] = True
        current_state['interrupted'] = False
    
    transcribe_start = time.time()
    
    try:
        # Create in-memory WAV file
        import io
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        wav_buffer.seek(0)
        
        transcription = groq_client.audio.transcriptions.create(
            file=("audio.wav", wav_buffer.read()),
            model="whisper-large-v3-turbo",
            response_format="json",
            language="en"
        )
        
        # Check if we were interrupted during transcription
        with state_lock:
            if current_state['interrupted']:
                print("⚠️ Transcription was interrupted!")
                current_state['transcribing'] = False
                return None
        
        transcribe_duration = time.time() - transcribe_start
        print(f"⏱️ Transcription took {transcribe_duration:.2f}s")
        
        text = transcription.text.strip()
        
        with state_lock:
            current_state['transcribing'] = False
        
        print("✅ Transcription complete!")
        return text
        
    except Exception as e:
        print(f"⚠️ Transcription error: {e}")
        with state_lock:
            current_state['transcribing'] = False
        return None


def text_to_speech_and_play_streaming(text):
    """Convert text to speech and stream directly to ffplay - no disk I/O"""
    global current_playback_process
    
    print(f"🔊 [TTS] Starting streaming TTS for: '{text[:50]}...'")
    tts_start = time.time()
    
    # Mark that we're generating TTS
    with state_lock:
        current_state['tts_generating'] = True
    
    try:
        # Check if interrupted before starting
        with state_lock:
            if current_state['interrupted']:
                print("⚠️ [TTS] Skipping - already interrupted!")
                current_state['tts_generating'] = False
                return False
        
        # Stop any existing playback
        stop_all_audio()
        
        # Start ffplay process that reads from stdin pipe
        with playback_lock:
            current_playback_process = subprocess.Popen(
                ['ffplay', '-nodisp', '-autoexit', '-volume', '10', '-i', 'pipe:0'],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"🎵 [TTS] Started ffplay pipe (PID: {current_playback_process.pid})")
        
        # Set playing state immediately since we're streaming
        with state_lock:
            current_state['playing_audio'] = True
        
        # Generate audio and stream directly to ffplay
        audio_generator = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id=SANTA_VOICE_ID,
            model_id="eleven_turbo_v2",
            optimize_streaming_latency=3,  # Max optimization for lowest latency
            output_format="mp3_44100_128"
        )
        
        first_chunk_time = None
        bytes_written = 0
        
        for chunk in audio_generator:
            # Check if interrupted during streaming
            with state_lock:
                if current_state['interrupted']:
                    print("⚠️ [TTS] Streaming interrupted!")
                    current_state['tts_generating'] = False
                    current_state['playing_audio'] = False
            
            # Handle interruption outside the lock
            if current_state['interrupted']:
                # Close stdin to stop ffplay gracefully
                try:
                    current_playback_process.stdin.close()
                except:
                    pass
                stop_all_audio(clear_state=False)  # State already cleared above
                return False
            
            if isinstance(chunk, bytes):
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    print(f"🎵 [TTS] First audio chunk received in {first_chunk_time - tts_start:.2f}s")
                
                try:
                    current_playback_process.stdin.write(chunk)
                    current_playback_process.stdin.flush()
                    bytes_written += len(chunk)
                except (BrokenPipeError, OSError) as e:
                    print(f"⚠️ [TTS] Pipe error (playback stopped?): {e}")
                    break
        
        # Close stdin to signal end of audio
        try:
            current_playback_process.stdin.close()
        except:
            pass
        
        print(f"🎵 [TTS] Streamed {bytes_written} bytes in {time.time() - tts_start:.2f}s")
        
        # Wait for playback to actually finish, then record end time
        try:
            current_playback_process.wait(timeout=30)  # Wait up to 30s for playback
        except:
            pass
        
        with state_lock:
            current_state['tts_generating'] = False
            current_state['playing_audio'] = False
            current_state['playback_end_time'] = time.time()
            current_state['reset_silero'] = True  # Signal VAD to reset Silero
            print(f"🎵 [TTS] Playback finished, recorded end time")
        
        return True
        
    except Exception as e:
        print(f"⚠️ [TTS] Exception: {e}")
        with state_lock:
            current_state['tts_generating'] = False
            current_state['playing_audio'] = False
        return False


def get_santa_response(child_text):
    global conversation_history
    
    print("🎅 Santa is thinking...")
    
    with state_lock:
        current_state['llm_thinking'] = True
        current_state['interrupted'] = False
    
    llm_start = time.time()
    
    with conversation_lock:
        conversation_history.append({"role": "user", "content": child_text})
        
        if len(conversation_history) > MAX_HISTORY_MESSAGES:
            conversation_history[:] = conversation_history[-MAX_HISTORY_MESSAGES:]
        
        messages = [
            {"role": "system", "content": SANTA_SYSTEM_PROMPT}
        ] + conversation_history
    
    try:
        # CHANGED: Using faster 8B model instead of 70B
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",  # Much faster than llama-3.3-70b-versatile
            temperature=0.9,
            max_tokens=max_tokens,
            stream=True
        )
        
        santa_response = ""
        
        for chunk in chat_completion:
            # Check if interrupted
            with state_lock:
                if current_state['interrupted']:
                    print("⚠️ LLM interrupted!")
                    current_state['llm_thinking'] = False
                    current_state['stop_thinking_sound'] = True
                    stop_thinking_sound()
                    return None
            
            if chunk.choices[0].delta.content:
                santa_response += chunk.choices[0].delta.content
        
        # Stop thinking sound before TTS starts
        with state_lock:
            current_state['llm_thinking'] = False
            current_state['stop_thinking_sound'] = True
        stop_thinking_sound()
        
        with conversation_lock:
            conversation_history.append({"role": "assistant", "content": santa_response})
        
        print(f"⏱️ Groq LLM responded in {time.time() - llm_start:.2f}s")
        print(f"🎅 Santa says: {santa_response}")
        
        # START TTS IN BACKGROUND THREAD
        threading.Thread(
            target=text_to_speech_and_play_streaming,
            args=(santa_response,),
            daemon=True
        ).start()
        
        return santa_response
        
    except Exception as e:
        print(f"⚠️ LLM error: {e}")
        with state_lock:
            current_state['llm_thinking'] = False
        return None


if __name__ == "__main__":
    try:
        print("\n🎅 Starting Santa Phone...\n")
        
        # Start continuous VAD in background thread
        vad_thread = threading.Thread(target=continuous_vad_loop, daemon=True)
        vad_thread.start()
        
        # Accumulator for audio that was interrupted before transcription
        pending_audio_chunks = []
        
        while True:
            # Wait for audio from queue
            audio_tuple = audio_queue.get()
            print(f"\n📥 Processing audio from queue...")
            
            # Extract audio data and echo flag
            audio_data = audio_tuple[0]
            sample_rate = audio_tuple[1]
            sample_width = audio_tuple[2]
            channels = audio_tuple[3]
            might_be_echo = audio_tuple[4] if len(audio_tuple) == 5 else False
            
            # Check if there's more audio waiting (rapid speech)
            # If so, accumulate before transcribing
            accumulated_audio = [audio_data]
            
            # Give a tiny moment for any rapid follow-up audio to arrive
            time.sleep(0.05)
            
            while not audio_queue.empty():
                try:
                    next_tuple = audio_queue.get_nowait()
                    print(f"📦 Merging additional audio from queue...")
                    accumulated_audio.append(next_tuple[0])
                except:
                    break
            
            # Also add any pending audio from interrupted transcriptions
            if pending_audio_chunks:
                print(f"📦 Including {len(pending_audio_chunks)} pending audio chunk(s) from interrupted transcription")
                accumulated_audio = pending_audio_chunks + accumulated_audio
                pending_audio_chunks = []
            
            # Combine all audio
            if len(accumulated_audio) > 1:
                print(f"🔗 Combining {len(accumulated_audio)} audio segments...")
                combined_audio = b''.join(accumulated_audio)
            else:
                combined_audio = accumulated_audio[0]
            
            # Create combined tuple for transcription
            combined_tuple = (combined_audio, sample_rate, sample_width, channels, might_be_echo)
            
            # Stop any existing audio playback before thinking sound
            stop_all_audio()
            stop_thinking_sound()
            
            # Start thinking sound immediately - gives feedback while we transcribe
            with state_lock:
                current_state['stop_thinking_sound'] = False
            thinking_thread = threading.Thread(target=play_thinking_sound, daemon=True)
            thinking_thread.start()
            
            # Transcribe
            text = transcribe_audio(combined_tuple)
            
            # Check if transcription was interrupted
            with state_lock:
                was_interrupted_during_transcription = current_state['interrupted']
            
            if was_interrupted_during_transcription:
                # Save this audio for later - it will be merged with the next capture
                print(f"💾 Saving interrupted audio for merge with next capture")
                pending_audio_chunks.append(combined_audio)
                continue
            
            if not text or len(text.strip()) < 3:
                print("⚠️ No clear speech detected, continuing to listen...\n")
                continue
            
            # Filter out likely echo based on transcription content
            if might_be_echo:
                text_lower = text.lower().strip()
                # Common echo artifacts: very short, or contains Santa's typical phrases
                echo_indicators = [
                    len(text_lower) < 10,  # Very short transcriptions
                    "ho ho" in text_lower,  # Santa's laugh
                    "merry christmas" in text_lower,
                    "north pole" in text_lower,
                    "mrs. claus" in text_lower,
                    "reindeer" in text_lower,
                    "elves" in text_lower,
                ]
                if any(echo_indicators):
                    print(f"🔇 Discarding likely echo transcription: '{text}'")
                    continue
            
            # Check if this was an interruption of Santa speaking
            with state_lock:
                was_santa_interrupted = current_state['was_interrupted']
                current_state['was_interrupted'] = False  # Reset the flag
                
                if was_santa_interrupted:
                    print(f"\n🔊 Santa was interrupted while speaking!")
                    print(f"📝 Interruption text: '{text}'")
                    # Clear any pending text merge since this is fresh input after interrupting Santa
                    current_state['last_transcription'] = ''
                    merged_text = text
                elif current_state['last_transcription'] and not current_state['playing_audio']:
                    print(f"\n🔗 MERGING WITH PREVIOUS TRANSCRIPTION!")
                    print(f"📝 Previous text: '{current_state['last_transcription']}'")
                    print(f"📝 New text: '{text}'")
                    merged_text = current_state['last_transcription'] + " " + text
                    print(f"✅ Combined text: '{merged_text}'")
                else:
                    print(f"\n📝 Sending text to Santa: '{text}'")
                    merged_text = text
                
                # Update last transcription for potential future merges
                current_state['last_transcription'] = text
            
            print(f"\n👦 Child said: '{merged_text}'")
            
            # Get Santa's response
            response = get_santa_response(merged_text)
            
            if response is None:
                print("⚠️ Response interrupted, continuing to listen...\n")
                continue
            
            # Reset conversation timer after successful response
            last_response_time = time.time()
            
            print("="*50 + "\n")
            
    except KeyboardInterrupt:
        print("\n\n👋 Santa is going back to the North Pole!")
        stop_all_audio()