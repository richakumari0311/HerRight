"""
Voice agent for HerRight.
Handles speech-to-text (STT) and text-to-speech (TTS) via Sarvam AI.
"""

import os
import base64
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

def speech_to_text(audio_bytes: bytes, language_code: str = "hi-IN") -> str:
    """
    Transcribe audio bytes to text using Sarvam STT.

    Args:
        audio_bytes: Raw audio in wav format.
        language_code: BCP-47 language code (e.g. hi-IN, en-IN, ta-IN).

    Returns:
        Transcribed text string.
    """
    response = client.speech_to_text.transcribe(
        file=("audio.wav", audio_bytes, "audio/wav"),
        model="saarika:v2.5",
        language_code=language_code,
    )
    return response.transcript


def text_to_speech(text: str, language_code: str = "hi-IN") -> bytes:
    """
    Convert text to speech audio bytes using Sarvam TTS.

    Args:
        text: Text to convert to speech.
        language_code: BCP-47 language code.

    Returns:
        Audio content as bytes (wav).
    """
    response = client.text_to_speech.convert(
        text=text,
        target_language_code=language_code,
        speaker="priya",
        model="bulbul:v3",
    )
    audio_base64 = response.audios[0]
    return base64.b64decode(audio_base64)