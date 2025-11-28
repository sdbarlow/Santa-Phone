import pyaudio
from silero_vad import load_silero_vad
import numpy as np
import torch
import wave
from collections import deque
import audioop
import os
import warnings
import signal
import anthropic
import subprocess
import time
from elevenlabs.client import ElevenLabs
from elevenlabs import stream as elevenlabs_stream
from groq import Groq
import random
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
SILERO_THRESHOLD = 0.9
VOLUME_THRESHOLD = 7000  # Main threshold for triggering speech capture
VOLUME_THRESHOLD_LOW = 2000  # If speech detected between LOW and THRESHOLD, play reminder
VOLUME_THRESHOLD_TRIGGERED = 1000  # Lower threshold once speech is detected
SILENCE_DURATION_MS = 500
SILENCE_CHUNKS = int((SILENCE_DURATION_MS / 1000) * SILERO_SAMPLE_RATE / VAD_CHUNK_SIZE)

# Pre-recorded messages
SPEAK_UP_REMINDER = "speak_up_reminder.mp3"  # Santa saying "speak loud and clearly"

print("🔄 Initializing Silero VAD...")
silero_model = load_silero_vad()
print("✅ Silero VAD ready!\n")

print("🔄 Initializing Claude...")
claude_client = anthropic.Anthropic(
    api_key=os.environ.get("CLAUDE_API_KEY", "your-key-here")
)
print("✅ Claude ready!\n")

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

Personality:
- Warm, jolly, and genuinely interested in each child
- Use "ho ho ho" sparingly (only for greetings or excitement)
- Speak naturally like a kind grandfather, not overly formal
- Remember details children share (names, wishes, siblings, pets)

Guidelines:
- Ask about their Christmas wishes, how they've been good, siblings, pets, or school
- If they tell you their name, use it in future responses
- If they ask for something, respond warmly but don't make specific promises
- Keep the magic alive - reference the North Pole, elves, reindeer, Mrs. Claus naturally
- Be encouraging about good behavior without being preachy
- If they seem shy, be extra gentle and patient

Remember: ONE sentence only. Make every word count.'''

max_tokens=50 

SANTA_OPENERS = [
    "opener_ho_ho_ho.mp3",
    "opener_oh_ho_ho.mp3",
]


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
    'last_activity_time': time.time()  # Track last speech activity
}
state_lock = threading.Lock()

INTERRUPTION_MERGE_WINDOW = 2.0  # Merge if interrupted within 2 seconds
CONVERSATION_TIMEOUT = 45.0  # Clear conversation after 45 seconds of silence

import queue

# Global queue for audio recordings
audio_queue = queue.Queue()


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
        
        print("👂 Listening...")
        
        while True:
            chunk_count += 1
            chunk = stream.read(MIC_CHUNK_SIZE, exception_on_overflow=False)
            
            # Calculate volume
            audio_int16 = np.frombuffer(chunk, dtype=np.int16)
            audio_float = audio_int16.astype(np.float64)
            volume = np.sqrt(np.mean(audio_float**2))
            
            if not np.isfinite(volume):
                volume = 0
            
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
                    
                    # Check what's currently running and interrupt
                    with state_lock:
                        # Check if conversation timed out
                        time_since_last = time.time() - current_state['last_activity_time']
                        if time_since_last > CONVERSATION_TIMEOUT:
                            print(f"⏰ Conversation timed out ({time_since_last:.1f}s), starting fresh!")
                            with conversation_lock:
                                conversation_history.clear()
                        
                        # Update activity time
                        current_state['last_activity_time'] = time.time()
                        
                        if current_state['playing_audio']:
                            print("🛑 Interrupting audio playback...")
                            current_state['was_interrupted'] = True
                            current_state['playing_audio'] = False
                            stop_all_audio(clear_state=False)
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
        
        # Add audio data to queue for processing (as tuple with metadata)
        audio_queue.put((audio_data, MIC_SAMPLE_RATE, sample_width, CHANNELS))
        print(f"📦 Added to queue (queue size: {audio_queue.qsize()})")
        
        # Continue immediately to next VAD session (no waiting!)


def transcribe_audio(audio_tuple):
    """Transcribe audio using Groq Whisper - can be interrupted"""
    print("📝 Transcribing with Groq...")
    
    audio_data, sample_rate, sample_width, channels = audio_tuple
    
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
    """Convert text to speech, save to file, and play - interruptible"""
    global current_playback_process
    
    print(f"🔊 [TTS] Starting TTS for: '{text[:50]}...'")
    tts_start = time.time()
    
    # Mark that we're generating TTS
    with state_lock:
        current_state['tts_generating'] = True
    
    try:
        # Generate audio and save to file
        audio_generator = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id=SANTA_VOICE_ID,
            model_id="eleven_turbo_v2",
            optimize_streaming_latency=2,
            output_format="mp3_44100_128"
        )
        
        # Save to temporary file, checking for interruption
        temp_file = "santa_response.mp3"
        with open(temp_file, "wb") as f:
            for chunk in audio_generator:
                # Check if interrupted during generation
                with state_lock:
                    if current_state['interrupted']:
                        print("⚠️ [TTS] Generation interrupted!")
                        current_state['tts_generating'] = False
                        return False
                
                if isinstance(chunk, bytes):
                    f.write(chunk)
        
        print(f"🎵 [TTS] Audio saved in {time.time() - tts_start:.2f}s")
        
        # Check again before starting playback
        with state_lock:
            if current_state['interrupted']:
                print("⚠️ [TTS] Skipping playback - interrupted!")
                current_state['tts_generating'] = False
                return False
            current_state['tts_generating'] = False
        
        # Now play it
        print(f"🎧 [TTS] Starting playback...")
        
        stop_all_audio()
        
        with playback_lock:
            current_playback_process = subprocess.Popen(
                ['ffplay', '-nodisp', '-autoexit', '-volume', '10', temp_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"🎵 [TTS] Started playback (PID: {current_playback_process.pid})")
        
        # ONLY NOW set playing_audio = True (when actually playing)
        with state_lock:
            current_state['playing_audio'] = True
        
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
            conversation_history = conversation_history[-MAX_HISTORY_MESSAGES:]
        
        messages = [
            {"role": "system", "content": SANTA_SYSTEM_PROMPT}
        ] + conversation_history
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
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
                    return None
            
            if chunk.choices[0].delta.content:
                santa_response += chunk.choices[0].delta.content
        
        with state_lock:
            current_state['llm_thinking'] = False
        
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
        
        while True:
            # Wait for audio from queue
            audio_tuple = audio_queue.get()
            print(f"\n📥 Processing audio from queue...")
            
            # Transcribe
            text = transcribe_audio(audio_tuple)
            
            if not text or len(text.strip()) < 3:
                print("⚠️ No clear speech detected, continuing to listen...\n")
                continue
            
            # Check if this was an interruption
            with state_lock:
                was_interruption = current_state['was_interrupted']
                current_state['was_interrupted'] = False  # Reset the flag
                
                if was_interruption:
                    print(f"\n🔊 Santa was interrupted while speaking!")
                    print(f"📝 New interruption text: '{text}'")
                    # Clear any pending merge since this is fresh input
                    current_state['last_transcription'] = ''
                    merged_text = text
                elif current_state['last_transcription'] and not current_state['playing_audio']:
                    print(f"\n🔗 MERGING TRANSCRIPTIONS!")
                    print(f"📝 Previous text: '{current_state['last_transcription']}'")
                    print(f"📝 New text: '{text}'")
                    merged_text = current_state['last_transcription'] + " " + text
                    print(f"✅ Combined text: '{merged_text}'")
                    print(f"🎅 Sending merged text to Santa...\n")
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
            
            print("="*50 + "\n")
            
    except KeyboardInterrupt:
        print("\n\n👋 Santa is going back to the North Pole!")
        stop_all_audio()