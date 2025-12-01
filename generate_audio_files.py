#!/usr/bin/env python3
"""
Generate pre-recorded audio files for Santa Phone using ElevenLabs.
Run this once to create the audio assets.
"""

import os
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

# Initialize ElevenLabs
elevenlabs_client = ElevenLabs(
    api_key=os.environ.get("ELEVENLABS_API_KEY", "your-key-here")
)

SANTA_VOICE_ID = "Gqe8GJJLg3haJkTwYj2L"

# Define all audio files to generate
AUDIO_FILES = {
    # Speak up reminder - gentle, encouraging
    "speak_up_reminder.mp3": "Ho ho, speak up a little louder for Santa, I want to hear you clearly!",
    
    # Opener variations
    "opener_ho_ho_ho.mp3": "Ho ho ho! Merry Christmas! Who do we have here?",
    "opener_oh_ho_ho.mp3": "Oh ho ho! Well hello there, little one! What's your name?",
    
    # Thinking sounds - short, warm acknowledgments (not full phrases)
    # These play while LLM processes, so keep them brief and neutral
    "thinking_ho_ho_ho.mp3": "Ho ho ho!",
    
    # Optional: Additional useful audio clips
    "goodbye.mp3": "Merry Christmas! Remember to be good, and I'll see you on Christmas Eve!",
    "next_child.mp3": "Ho ho ho! Is there another friend who wants to talk to Santa?",
}

# Directory to save audio files (same directory as santa_phone.py)
OUTPUT_DIR = "."  # Change this to your santa_phone.py directory path if needed


def generate_audio(text: str, filename: str):
    """Generate audio file from text using ElevenLabs"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    print(f"🎙️ Generating: {filename}")
    print(f"   Text: \"{text}\"")
    
    try:
        audio_generator = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id=SANTA_VOICE_ID,
            model_id="eleven_turbo_v2",
            output_format="mp3_44100_128"
        )
        
        # Save to file
        with open(filepath, "wb") as f:
            for chunk in audio_generator:
                if isinstance(chunk, bytes):
                    f.write(chunk)
        
        # Verify file was created
        file_size = os.path.getsize(filepath)
        print(f"   ✅ Saved: {filepath} ({file_size} bytes)\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        return False


def main():
    print("🎅 Santa Phone Audio Generator")
    print("=" * 40)
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}\n")
    
    success_count = 0
    fail_count = 0
    
    for filename, text in AUDIO_FILES.items():
        if generate_audio(text, filename):
            success_count += 1
        else:
            fail_count += 1
    
    print("=" * 40)
    print(f"✅ Generated: {success_count} files")
    if fail_count > 0:
        print(f"❌ Failed: {fail_count} files")
    
    print("\n📁 Files created:")
    for filename in AUDIO_FILES.keys():
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            print(f"   ✓ {filepath}")
        else:
            print(f"   ✗ {filepath} (missing)")


if __name__ == "__main__":
    main()