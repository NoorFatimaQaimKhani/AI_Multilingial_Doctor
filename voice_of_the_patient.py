# voice_of_the_patient.py
from groq import Groq


def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY, language=None):
    """Transcribe an audio file with Groq's hosted Whisper.

    language: ISO code such as "en" or "ur". None lets Whisper auto-detect,
    which can confuse Urdu with Hindi/Arabic, so pass it when you know it.
    """
    client = Groq(api_key=GROQ_API_KEY)

    with open(audio_filepath, "rb") as audio_file:
        kwargs = {"model": stt_model, "file": audio_file}
        if language:
            kwargs["language"] = language
        transcription = client.audio.transcriptions.create(**kwargs)

    return transcription.text