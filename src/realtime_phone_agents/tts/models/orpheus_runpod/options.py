from pydantic import BaseModel, Field

from realtime_phone_agents.config import settings

DEFAULT_HEADERS = {"Content-Type": "application/json"}
CUSTOM_TOKEN_PREFIX = "<custom_token_"


class OrpheusTTSOptions(BaseModel):
    """Orpheus TTS options with defaults from Pydantic settings."""

    api_url: str = Field(
        default_factory=lambda: settings.orpheus.api_url,
        description="Orpheus TTS API URL",
    )
    model: str = Field(
        default_factory=lambda: settings.orpheus.model, description="Orpheus TTS Model"
    )
    headers: dict = Field(
        default_factory=lambda: DEFAULT_HEADERS.copy(),
        description="HTTP headers for API requests",
    )
    voice: str = Field(
        default_factory=lambda: settings.orpheus.voice,
        description="Voice to use for synthesis",
    )
    temperature: float = Field(
        default_factory=lambda: settings.orpheus.temperature,
        description="Temperature for generation",
    )
    top_p: float = Field(
        default_factory=lambda: settings.orpheus.top_p,
        description="Top-p sampling parameter",
    )
    max_tokens: int = Field(
        default_factory=lambda: settings.orpheus.max_tokens,
        description="Maximum tokens to generate",
    )
    repetition_penalty: float = Field(
        default_factory=lambda: settings.orpheus.repetition_penalty,
        description="Repetition penalty",
    )
    sample_rate: int = Field(
        default_factory=lambda: settings.orpheus.sample_rate,
        description="Audio sample rate (Hz)",
    )
    debug: bool = Field(
        default_factory=lambda: settings.orpheus.debug, description="Enable debug mode"
    )

    model_config = {"arbitrary_types_allowed": True}
