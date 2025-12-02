from fastrtc import audio_to_bytes
from openai import OpenAI

from realtime_phone_agents.config import settings
from realtime_phone_agents.stt.models.base import STTModel


class FasterWhisperSTT(STTModel):
    """Speech-to-Text model using Faster Whisper."""

    def __init__(self):
        self.client = OpenAI(
            api_key="",
            base_url=f"{settings.faster_whisper.api_url}/v1",
        )

    def stt(self, audio_data: bytes) -> str:
        """Convert speech audio to text."""
        response = self.client.audio.transcriptions.create(
            file=("audio.wav", audio_to_bytes(audio_data)),
            model=settings.faster_whisper.model,
            response_format="verbose_json",
        )
        return response.text
