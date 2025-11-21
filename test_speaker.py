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
import psutil  # Add this import

# subprocess.run(['amixer', '-D', 'bluealsa', 'sset', 'Master', '90%'])

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global variables for audio playback control
current_playback_process = None
playback_lock = threading.Lock()

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
MIC_SAMPLE_RATE = 44100  # Your mic's actual supported rate
SILERO_SAMPLE_RATE = 16000  # Silero VAD requirement
VAD_CHUNK_SIZE = 512  # Silero needs exactly 512 samples at 16kHz

# Calculate mic chunk size - use a ratio that produces exactly 512 samples
MIC_CHUNK_SIZE = int(VAD_CHUNK_SIZE * MIC_SAMPLE_RATE / SILERO_SAMPLE_RATE)  # = 1411.2 → 1411

FORMAT = pyaudio.paInt16
CHANNELS = 1

# Silero VAD settings
SILERO_THRESHOLD = 0.9
VOLUME_THRESHOLD_IDLE = 2000    # When Santa is NOT speaking
VOLUME_THRESHOLD_SPEAKING = 20000  # When Santa IS speaking
SILENCE_DURATION_MS = 500
SILENCE_CHUNKS = int((SILENCE_DURATION_MS / 1000) * SILERO_SAMPLE_RATE / VAD_CHUNK_SIZE)

santa_max_volume = 0
santa_max_volume_lock = threading.Lock()

print("🔄 Initializing Silero VAD...")
silero_model = load_silero_vad()
print("✅ Silero VAD ready!\n")

# Initialize Claude
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

SANTA_SYSTEM_PROMPT = 'You are Santa Claus. Keep responses to EXACTLY 1 sentence. Be brief and magical.'
max_tokens=50 

SANTA_OPENERS = [
    "opener_ho_ho_ho.mp3",
    "opener_oh_ho_ho.mp3",
]

def stop_all_audio():
    """Stop any currently playing audio"""
    global current_playback_process
    
    with playback_lock:
        if current_playback_process and current_playback_process.poll() is None:
            print("🛑 Stopping current audio playback...")
            try:
                # Kill the process and all its children
                parent = psutil.Process(current_playback_process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass
            current_playback_process = None

def play_audio_file(audio_file, description="audio", volume=10):
    """Play an audio file with interrupt capability"""
    global current_playback_process
    
    stop_all_audio()
    
    print(f"🎵 Playing {description}...")
    
    with playback_lock:
        current_playback_process = subprocess.Popen(
            ['ffplay', '-nodisp', '-autoexit', '-volume', str(volume), audio_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    # Wait for completion (can be interrupted by stop_all_audio)
    current_playback_process.wait()
    
    with playback_lock:
        current_playback_process = None

def is_generic_greeting(text):
    """Check if the input is a generic greeting that can use a predetermined response"""
    text_lower = text.lower().strip()

    # Normalize apostrophes
    text_lower = text_lower.replace('’', "'")
    
    # Remove common punctuation
    text_clean = text_lower.replace(',', '').rstrip('!.?')

    greetings = {
        'hi','hello','hey','hi santa','hello santa','hey santa','hi there','hello there',
        'good morning','good afternoon','good evening','merry christmas','happy holidays',
        'santa','hiiii','hellooo','hii santa','hellooo santa','santaaa','omg hi santa',
        'yay santa','oh hi santa','hi santa wow','hi santa yay','santa are you there',
        'is this santa','santa can you hear me','can you hear me santa','hello santa are you there',
        'um hi','uh hi','um hello','uh hello','hi um','hello um','santa hi','santa hello',
        "hi it's me",'hi hi hi','hello hello','hey hey hey','hiiii santaaaa','yay hi santa',
        'woo hi santa','hi santa guess what','guess what santa','hey santa it’s me',
        'hi santa it’s me','yo santa','sup santa','hey dude santa','hi mr santa','hello mr santa',
        'hey santa claus','hi santa claus','hello santa claus','ho ho ho santa','hi ho ho ho',
        'hey big guy santa','hi tana','hewwo santa','santa hewwo','hi cwause','hi santaaa',
        'hi santa pwease','hello santa pwease','santa good morning','santa good evening',
        'santa good afternoon','season’s greetings santa','happy christmas santa',
        'happy holidays santa',"hi it's me again","hello it's me again",'santa it’s me',
        "hi santa i'm here","hi santa i'm called you","hello santa i'm ready",
        "hi santa i'm excited","hi santa i'm nervous","hello santa i'm shy",
        'jingle bells santa','hi santa merry christmas','hello santa merry christmas',
        'merry christmas santa','happy holidays santa','ok hi santa','okay hello santa',
        'okay hi santa','hi santa okay','hi santa wait','wait santa hi','oh okay hi santa',
        'oh hello santa','is that you santa','santa is that you','hi santa is this real',
        'hello santa is this really you','hello santa i picked up','hi santa i answered',
        'hi santa i got the phone',"hi santa look it's me","hi santa i'm calling",
        "hello santa i'm calling you",'hi santa can we talk',"hello santa let's talk",
        'hey santa yoooo','hi santaaaaa wow','wow santa hi','wow it’s santa',
        'oh my gosh santa hi','hi santa hi','hello santa hello','santa hello hi',
        'hi santa hello','hello santa hi', 'hi is this santa',
'hello is this santa',
'hey is this santa',

'hi is that santa',
'hello is that santa',
'hey is that santa',

'hi santa is that you',
'hello santa is that you',
'hey santa is that you',

'hi santa is this you',
'hello santa is this you',
'hey santa is this you',

'is this santa hi',
'is this santa hello',
'is this santa hey',

'is that santa hi',
'is that santa hello',
'is that santa hey',

'this is santa right',
'is this really santa',
'is that really santa',
'wait is this santa',
'wait is that santa',
    }

    return text_clean in greetings

def play_random_opener_async():
    """Play a random Santa opener in background thread"""
    opener = random.choice(SANTA_OPENERS)
    print(f"🎵 Playing opener: {opener} (in background)")
    
    def play_in_background():
        play_audio_file(opener, "opener")
    
    # Start playing in background thread (don't wait for it)
    thread = threading.Thread(target=play_in_background, daemon=True)
    thread.start()
    # Return immediately without waiting

def play_random_greeting_response():
    """Play a random predetermined greeting response"""
    greeting_files = [
        "greeting_whats_your_name.mp3",
        "greeting_wonderful_to_hear.mp3",
        "greeting_merry_christmas.mp3",
        "greeting_happy_to_talk.mp3",
        "greeting_glad_you_called.mp3",
        "greeting_nice_to_meet.mp3",
        "greeting_wonderful_day.mp3",
        "greeting_jolly_hello.mp3",
        "greeting_warm_welcome.mp3",
    ]
    
    # Filter to only existing files
    available_files = [f for f in greeting_files if os.path.exists(f)]
    
    if not available_files:
        print("⚠️  No greeting response files found!")
        return
    
    chosen_file = random.choice(available_files)
    play_audio_file(chosen_file, "greeting")

def downsample_for_vad(chunk, from_rate, to_rate, target_samples=512):
    """Downsample audio chunk and ensure exactly target_samples output"""
    audio_int16 = np.frombuffer(chunk, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    
    # Use numpy interpolation to get exactly target_samples
    num_samples_original = len(audio_float32)
    indices = np.linspace(0, num_samples_original - 1, target_samples)
    audio_resampled = np.interp(indices, np.arange(num_samples_original), audio_float32)
    
    # Verify we have exactly the right number of samples
    assert len(audio_resampled) == target_samples, f"Expected {target_samples} samples, got {len(audio_resampled)}"
    
    return audio_resampled


def record_with_vad():
    """Record audio using Silero VAD with resampling and volume threshold"""
    global santa_max_volume
    
    print("🎤 Listening for speech...")
    
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
    
    while True:
        chunk = stream.read(MIC_CHUNK_SIZE, exception_on_overflow=False)
        
        # Calculate volume
        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        audio_float = audio_int16.astype(np.float64)
        volume = np.sqrt(np.mean(audio_float**2))
        
        if not np.isfinite(volume):
            volume = 0
        
        # Check if Santa is speaking
        with playback_lock:
            is_santa_speaking = (current_playback_process is not None and 
                                current_playback_process.poll() is None)
        
        # Track Santa's volume
        if is_santa_speaking and volume > santa_max_volume:
            with santa_max_volume_lock:
                santa_max_volume = volume
                print(f"\n📊 New Santa volume peak: {int(santa_max_volume)}")
        
        # Dynamic threshold
        current_threshold = VOLUME_THRESHOLD_SPEAKING if is_santa_speaking else VOLUME_THRESHOLD_IDLE
        
        # Skip if volume is too low
        if volume < current_threshold:
            if not triggered:
                continue
            else:
                silence_count += 1
                voiced_frames.append(chunk)
                if silence_count > SILENCE_CHUNKS:
                    print("\n✅ Speech ended!")
                    break
                continue
        
        # Downsample
        audio_resampled = downsample_for_vad(chunk, MIC_SAMPLE_RATE, SILERO_SAMPLE_RATE, VAD_CHUNK_SIZE)
        audio_tensor = torch.from_numpy(audio_resampled.astype(np.float32))
        
        if len(audio_tensor) != VAD_CHUNK_SIZE:
            continue
        
        # Get speech probability
        speech_prob = silero_model(audio_tensor, SILERO_SAMPLE_RATE).item()
        is_speech = speech_prob > SILERO_THRESHOLD
        
        if is_speech:
            print(".", end="", flush=True)
        
        if not triggered:
            ring_buffer.append(chunk)
            
            if is_speech:
                triggered = True
                print(f"\n🗣️  Speech detected! (confidence: {speech_prob:.2f}, volume: {int(volume)}, threshold: {current_threshold})")
                voiced_frames.extend(ring_buffer)
                voiced_frames.append(chunk)
        else:
            voiced_frames.append(chunk)
            
            if is_speech:
                silence_count = 0
            else:
                silence_count += 1
            
            if silence_count > SILENCE_CHUNKS:
                print("\n✅ Speech ended!")
                break
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Save recording
    wf = wave.open('recording.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(MIC_SAMPLE_RATE)
    wf.writeframes(b''.join(voiced_frames))
    wf.close()
    
    # Show max Santa volume after recording
    with santa_max_volume_lock:
        if santa_max_volume > 0:
            print(f"📊 Santa's max volume: {int(santa_max_volume)}")
    
    return 'recording.wav'

def transcribe_audio(audio_file):
    """Transcribe audio using Groq Whisper"""
    print("📝 Transcribing with Groq...")
    
    transcribe_start = time.time()
    
    with open(audio_file, "rb") as file:
        transcription = groq_client.audio.transcriptions.create(
            file=(audio_file, file.read()),
            model="whisper-large-v3-turbo",
            response_format="json",
            language="en"
        )
    
    transcribe_duration = time.time() - transcribe_start
    print(f"⏱️  Transcription took {transcribe_duration:.2f}s")
    
    # Get confidence metrics from first segment
    avg_logprob = -1.0
    no_speech_prob = 0.0
    
    # if transcription.segments and len(transcription.segments) > 0:
    #     segment = transcription.segments[0]
    #     avg_logprob = segment.get('avg_logprob', -1.0)
    #     no_speech_prob = segment.get('no_speech_prob', 0.0)
    
    text = transcription.text.strip()
    
    # print(f"🔍 avg_logprob: {avg_logprob:.3f}, no_speech_prob: {no_speech_prob:.2e}")
    
    # if no_speech_prob > 1e-6:
    #     print(f"⚠️  High no_speech_prob ({no_speech_prob:.2e}), likely noise")
    #     return ""
    
    # if avg_logprob < -0.76:
    #     print(f"⚠️  Low confidence (avg_logprob: {avg_logprob:.2f}), likely noise")
    #     return ""
    
    print("✅ Transcription complete!")
    return text


def text_to_speech_and_play_streaming(text):
    """Convert text to speech and play using ElevenLabs"""
    print("🔊 Generating Santa's voice...")
    tts_start = time.time()
    
    try:
        # Use .convert() which returns complete audio as iterator
        audio_generator = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id=SANTA_VOICE_ID,
            model_id="eleven_turbo_v2",
            optimize_streaming_latency=4,
            output_format="mp3_22050_32" 
        )
        
        print("🎧 Starting ffplay...")
        
        global current_playback_process
        stop_all_audio()
        
        with playback_lock:
            current_playback_process = subprocess.Popen(
                ['ffplay', '-nodisp', '-autoexit', '-volume', '10', '-'],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        # Start volume monitoring in background
        
        chunk_count = 0
        total_bytes = 0
        first_chunk_time = None
        
        # Stream chunks as they arrive
        for chunk in audio_generator:
            if isinstance(chunk, bytes):
                chunk_count += 1
                total_bytes += len(chunk)
                
                if chunk_count == 1:
                    first_chunk_time = time.time() - tts_start
                    print(f"🎵 First audio in {first_chunk_time:.2f}s - playing now!")
                
                try:
                    current_playback_process.stdin.write(chunk)
                    current_playback_process.stdin.flush()
                except (BrokenPipeError, ValueError) as e:
                    print(f"❌ Error: {e}")
                    break
        
        print(f"🏁 Complete. Chunks: {chunk_count}, Bytes: {total_bytes}")
        
        # Close and wait
        if current_playback_process and current_playback_process.poll() is None:
            current_playback_process.stdin.close()
        
        if current_playback_process:
            current_playback_process.wait()
        
        print(f"⏱️  Total: {time.time() - tts_start:.2f}s")
        
        # Show current max
        with santa_max_volume_lock:
            print(f"📊 Santa's max volume so far: {int(santa_max_volume)}")
        
        with playback_lock:
            current_playback_process = None
        
        return True
        
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        import traceback
        traceback.print_exc()
        stop_all_audio()
        return False

# def text_to_speech_and_play_streaming(text):
#     """Convert text to speech and play using ElevenLabs streaming"""
#     print("🔊 Streaming Santa's voice...")
#     print(f"📝 Text to convert: '{text}'")
#     print(f"📏 Text length: {len(text)} characters")
#     tts_start = time.time()
    
#     try:
#         # Use .stream() method for real-time streaming
#         print("🌐 Starting ElevenLabs stream...")
#         audio_stream = elevenlabs_client.text_to_speech.stream(
#             text=text,
#             voice_id=SANTA_VOICE_ID,
#             model_id="eleven_turbo_v2_5",  # Newer, faster model
            # optimize_streaming_latency=4,  # Max optimization (0-4)
            # output_format="mp3_22050_32"  # Lower quality = faster
#         )
        
#         print("🎧 Stream created, starting ffplay...")
        
#         # Start ffplay for playback (better streaming support than mpg123)
#         global current_playback_process
        
#         stop_all_audio()
        
#         with playback_lock:
#             current_playback_process = subprocess.Popen(
#                 ['ffplay', '-nodisp', '-autoexit', '-'],  # -nodisp hides window, -autoexit closes when done
#                 stdin=subprocess.PIPE,
#                 stdout=subprocess.DEVNULL,
#                 stderr=subprocess.DEVNULL
#             )
        
#         print("✅ ffplay started, streaming chunks...")
        
#         chunk_count = 0
#         total_bytes = 0
#         first_chunk_time = None
        
#         # Stream audio chunks as they arrive
#         for chunk in audio_stream:
#             if isinstance(chunk, bytes):
#                 chunk_count += 1
#                 total_bytes += len(chunk)
                
#                 if chunk_count == 1:
#                     first_chunk_time = time.time() - tts_start
#                     print(f"🎵 First audio in {first_chunk_time:.2f}s - playing now!")
                
#                 try:
#                     current_playback_process.stdin.write(chunk)
#                     current_playback_process.stdin.flush()
#                 except (BrokenPipeError, ValueError) as e:
#                     print(f"❌ Error writing chunk {chunk_count}: {e}")
#                     break
        
#         print(f"🏁 Stream ended. Total chunks: {chunk_count}, Total bytes: {total_bytes}")
        
#         # Close stdin to signal end
#         if current_playback_process and current_playback_process.poll() is None:
#             current_playback_process.stdin.close()
#             print("✅ Stdin closed, waiting for playback to complete...")
        
#         # Wait for playback to finish
#         if current_playback_process:
#             current_playback_process.wait()
#             print("✅ ffplay finished")
        
#         tts_duration = time.time() - tts_start
        
#         print(f"⏱️  Total time: {tts_duration:.2f}s ({chunk_count} chunks, {total_bytes} bytes)")
        
#         with playback_lock:
#             current_playback_process = None
        
#         return True
        
#     except Exception as e:
#         print(f"❌ TTS Streaming Error: {e}")
#         import traceback
#         traceback.print_exc()
#         stop_all_audio()
#         return False
        
#     except Exception as e:
#         print(f"❌ TTS Streaming Error: {e}")
#         import traceback
#         traceback.print_exc()
#         stop_all_audio()
#         return False

def get_santa_response(child_text):
    """Get Santa's response from Groq LLM"""
    print("🎅 Santa is thinking...")
    llm_start = time.time()
    
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": SANTA_SYSTEM_PROMPT},
            {"role": "user", "content": child_text}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.9,
        max_tokens=max_tokens,
        stream=True
    )
    
    santa_response = ""
    
    # Collect full response
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            santa_response += chunk.choices[0].delta.content
    
    print(f"⏱️  Groq LLM responded in {time.time() - llm_start:.2f}s")
    print(f"🎅 Santa says: {santa_response}")
    
    # Start TTS in BACKGROUND thread (don't wait for it)
    threading.Thread(
        target=text_to_speech_and_play_streaming,
        args=(santa_response,),
        daemon=True
    ).start()
    
    return santa_response


# def get_santa_response(child_text):
#     """Get Santa's response with streaming LLM"""
#     # if interrupt_flag.is_set():
#     #     return None
#     print("🎅 Santa is thinking...")
#         llm_start = time.time()
#         chat_completion = groq_client.chat.completions.create(
#     messages=[
#                 {"role": "system", "content": SANTA_SYSTEM_PROMPT},
#                 {"role": "user", "content": child_text}
#             ],
#     model="llama-3.3-70b-versatile",
#     temperature=0.9,
#     max_tokens=max_tokens,
#     stream=True
#         )
#         santa_response = ""
#         tts_started = False
#     for chunk in chat_completion:
#     if chunk.choices[0].delta.content:
#                 santa_response += chunk.choices[0].delta.content
#     # Start TTS after ~20-30 characters (partial sentence)
#     if not tts_started and len(santa_response) >= 25:
#     print(f"⏱️  Starting TTS early at {time.time() - llm_start:.2f}s")
#     print(f"🎅 Santa says: {santa_response}...")
#     # Start TTS in background
#                     threading.Thread(
#     target=text_to_speech_and_play_streaming,
#     args=(santa_response,),
#     daemon=True
#                     ).start()
#                     tts_started = True
#     print(f"🎅 Full response: {santa_response}")
#     if not tts_started:
#             text_to_speech_and_play_streaming(santa_response)
#     return santa_response


# def test_microphone_volume_with_santa():
#     """Test mode - plays Santa greetings while monitoring mic volume"""
#     print("🎤 Testing microphone volume with Santa playback...")
#     print("This will help you set VOLUME_THRESHOLD above Santa's playback level")
#     print("Press Ctrl+C to stop\n")
    
#     greeting_files = [
#         "greeting_whats_your_name.mp3",
#         "greeting_wonderful_to_hear.mp3",
#         "greeting_merry_christmas.mp3",
#         "greeting_happy_to_talk.mp3",
#         "greeting_glad_you_called.mp3",
#         "greeting_nice_to_meet.mp3",
#         "greeting_wonderful_day.mp3",
#         "greeting_jolly_hello.mp3",
#         "greeting_warm_welcome.mp3",
#     ]
    
#     available_files = [f for f in greeting_files if os.path.exists(f)]
    
#     if not available_files:
#         print("⚠️  No greeting response files found!")
#         return
    
#     audio = pyaudio.PyAudio()
#     stream = audio.open(
#         format=FORMAT,
#         channels=CHANNELS,
#         rate=MIC_SAMPLE_RATE,
#         input=True,
#         input_device_index=1,
#         frames_per_buffer=MIC_CHUNK_SIZE
#     )
    
#     max_volume_seen = 0
    
#     def play_greeting_loop():
#         """Play greetings continuously in background"""
#         while True:
#             chosen_file = random.choice(available_files)
#             print(f"\n🎅 Playing: {chosen_file}")
#             subprocess.run(
#                 ['ffplay', '-nodisp', '-autoexit', '-volume', '90', chosen_file],
#                 stdout=subprocess.DEVNULL,
#                 stderr=subprocess.DEVNULL
#             )
#             time.sleep(2)  # Pause between greetings
    
#     # Start playing greetings in background thread
#     greeting_thread = threading.Thread(target=play_greeting_loop, daemon=True)
#     greeting_thread.start()
    
#     try:
#         print("\n📊 Monitoring volume levels...\n")
#         while True:
#             chunk = stream.read(MIC_CHUNK_SIZE, exception_on_overflow=False)
#             audio_int16 = np.frombuffer(chunk, dtype=np.int16)
            
#             # Calculate RMS volume safely
#             audio_float = audio_int16.astype(np.float64)
#             volume = np.sqrt(np.mean(audio_float**2))
            
#             # Handle NaN/inf
#             if not np.isfinite(volume):
#                 volume = 0
            
#             # Track maximum
#             if volume > max_volume_seen:
#                 max_volume_seen = volume
            
#             # Visual bar
#             bar_length = min(int(volume / 100), 50)
#             bar = "█" * bar_length
            
#             # Show current, max, and recommended threshold
#             recommended_threshold = int(max_volume_seen * 1.2)  # 20% above max Santa volume
            
#             print(f"\rVolume: {int(volume):5d} {bar:<50} | Max: {int(max_volume_seen):5d} | Recommend threshold: {recommended_threshold:5d}", 
#                   end="", flush=True)
            
#     except KeyboardInterrupt:
#         print("\n\n✅ Done testing")
#         print(f"\n📋 RESULTS:")
#         print(f"   Max volume from Santa playback: {int(max_volume_seen)}")
#         print(f"   Recommended VOLUME_THRESHOLD: {int(max_volume_seen * 1.2)}")
#         print(f"\n💡 Set this in your code:")
#         print(f"   VOLUME_THRESHOLD = {int(max_volume_seen * 1.2)}")
#     finally:
#         stream.stop_stream()
#         stream.close()
#         audio.terminate()

if __name__ == "__main__":
    
    try:
        while True:
                    total_start = time.time()
        # Record
                    audio_file = record_with_vad()
        # Transcribe
                    text = transcribe_audio(audio_file)
                    # Filter out empty or too short transcriptions
                    if not text or len(text.strip()) < 3:
                        print("⚠️  No clear speech detected, listening again...\n")
                        continue

                    print(f"\n👦 Child said: '{text}'")
                    if is_generic_greeting(text):
                        print("🎉 Detected generic greeting, playing predetermined response")
                        play_random_greeting_response()
                    else:
                    # play_random_opener_async()
                        santa_response = get_santa_response(text)
                        total_duration = time.time() - total_start
                        print(f"\n⏱️  TOTAL RESPONSE TIME: {total_duration:.2f}s")
                    print("="*50 + "\n")
    except KeyboardInterrupt:
        print("\n\n👋 Santa is going back to the North Pole!")