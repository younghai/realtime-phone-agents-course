from realtime_phone_agents.tts.models.base import TTSModel
from realtime_phone_agents.tts.models.kokoro import KokoroTTSModel
from realtime_phone_agents.tts.models.orpheus_runpod import OrpheusTTSModel
from realtime_phone_agents.tts.models.together import TogetherTTS


def get_tts_model(model_name: str) -> TTSModel:
    """Get a TTS model by name.

    Available options:
        - "kokoro": Local Kokoro TTS via FastRTC
        - "orpheus-runpod": Orpheus TTS via RunPod deployment
        - "together": Together AI API (supports Orpheus, Kokoro, Cartesia)
    """
    if model_name == "kokoro":
        return KokoroTTSModel()
    elif model_name == "orpheus-runpod":
        return OrpheusTTSModel()
    elif model_name == "together":
        return TogetherTTS()
    else:
        raise ValueError(
            f"Invalid TTS model name: {model_name}. Available: kokoro, orpheus-runpod, together"
        )
