# voice_of_the_doctor.py
import os

from elevenlabs.client import ElevenLabs
from gtts import gTTS

# Default ElevenLabs voice ("Aria"). Override with ELEVENLABS_VOICE_ID in .env.
DEFAULT_VOICE_ID = "9BWtsMINqrJLrRacOk9x"


def text_to_speech_with_gtts(input_text, output_filepath, language="en"):
    """Free fallback voice. language: "en", "ur", ..."""
    gTTS(text=input_text, lang=language, slow=False).save(output_filepath)
    return output_filepath


def text_to_speech_with_elevenlabs(input_text, output_filepath, language="en"):
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")

    # Eleven v3 supports Urdu; the turbo model is used for English (faster/cheaper).
    if language == "ur":
        model_id = os.environ.get("ELEVENLABS_MODEL_UR", "eleven_v3")
    else:
        model_id = os.environ.get("ELEVENLABS_MODEL_EN", "eleven_turbo_v2_5")
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID)

    client = ElevenLabs(api_key=api_key)
    audio = client.text_to_speech.convert(
        voice_id=voice_id,
        text=input_text,
        model_id=model_id,
        output_format="mp3_44100_128",
    )
    with open(output_filepath, "wb") as f:
        for chunk in audio:
            if chunk:
                f.write(chunk)
    return output_filepath


def text_to_speech(input_text, output_filepath, language="en"):
    """Try ElevenLabs first; fall back to gTTS if it fails (bad key, quota, etc.)."""
    try:
        return text_to_speech_with_elevenlabs(input_text, output_filepath, language)
    except Exception as e:
        print(f"ElevenLabs failed ({e}); falling back to gTTS.")
        return text_to_speech_with_gtts(input_text, output_filepath, language)