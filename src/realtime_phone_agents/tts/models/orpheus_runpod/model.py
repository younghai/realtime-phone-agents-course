"""
Orpheus TTS Model for RunPod deployment.

This implementation is based on the OrpheusEngine from the amazing RealtimeTTS library.
Credit to Kolja Beigel and the RealtimeTTS team for their excellent work.

Original Repository: https://github.com/KoljaB/RealtimeTTS
License: MIT
"""

import json
import time
import asyncio
import threading
import traceback
from typing import Optional, Generator, AsyncGenerator
from loguru import logger

import requests
import numpy as np
from numpy.typing import NDArray

from realtime_phone_agents.tts.models.base import TTSModel
from realtime_phone_agents.tts.models.orpheus_runpod.options import (
    OrpheusTTSOptions,
    CUSTOM_TOKEN_PREFIX,
)
from realtime_phone_agents.tts.models.orpheus_runpod.token_decoders import (
    convert_to_audio,
)


class OrpheusTTSModel(TTSModel):
    def __init__(self, options: OrpheusTTSOptions | None = None):
        """
        Initialize the Orpheus TTS model.

        Args:
            options: Configuration options for TTS generation. If None, uses defaults.
        """
        self.options = options or OrpheusTTSOptions()

    def set_voice(self, voice: str) -> None:
        """
        Set the voice for the Orpheus TTS model.

        Args:
            voice: Voice identifier.
        """
        self.options.voice = voice

    def _format_prompt(self, prompt: str, voice: str) -> str:
        """
        Format the input prompt with Orpheus-specific tokens.

        Args:
            prompt: The text to synthesize.
            voice: Voice identifier.

        Returns:
            Formatted prompt string with special tokens.
        """
        return f"<|audio|>{voice}: {prompt}<|eot_id|>"

    def _generate_tokens_sync(
        self,
        text: str,
        options: OrpheusTTSOptions,
    ) -> Generator[str, None, None]:
        """
        Generate audio tokens synchronously via streaming API.

        Args:
            text: Text to convert to speech.
            options: TTS configuration options.

        Yields:
            Individual token strings as they arrive from the API.
        """
        logger.debug(f"Generating tokens for text: {text}")
        formatted_prompt = self._format_prompt(text, options.voice)

        payload = {
            "model": options.model,
            "prompt": formatted_prompt,
            "max_tokens": options.max_tokens,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "repeat_penalty": options.repetition_penalty,
            "stream": True,
        }

        try:
            logger.debug(f"Requesting API: {options.api_url}")
            response = requests.post(
                f"{options.api_url}/v1/completions",
                headers=options.headers,
                json=payload,
                stream=True,
                timeout=30,
            )
            response.raise_for_status()

            token_counter = 0
            start_time = time.time()

            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode("utf-8")
                if not line_str.startswith("data: "):
                    continue

                data_str = line_str[6:].strip()
                if data_str == "[DONE]":
                    logger.debug("Token generation complete")
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    continue

                if "choices" in data and data["choices"]:
                    token_text = data["choices"][0].get("text", "")
                    if token_text:
                        token_counter += 1
                        if token_counter == 1:
                            elapsed = time.time() - start_time
                            logger.info(f"Time to first token: {elapsed:.2f}s")
                        yield token_text

        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def _turn_token_into_id(self, token_string: str, index: int) -> Optional[int]:
        """
        Convert a token string to its numeric audio token ID.

        Args:
            token_string: Token text containing custom token markers.
            index: Current token position (used for offset calculation).

        Returns:
            Decoded token ID or None if invalid.
        """
        token_string = token_string.strip()
        last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)

        if last_token_start == -1:
            return None

        last_token = token_string[last_token_start:]

        if last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
            try:
                number_str = last_token[
                    14:-1
                ]  # Extract number from <custom_token_XXXXX>
                token_id = int(number_str) - 10 - ((index % 7) * 4096)
                return token_id
            except ValueError:
                return None

        return None

    def _convert_buffer(
        self,
        multiframe: list[int],
        count: int,
    ) -> NDArray[np.int16] | None:
        """
        Convert token IDs to PCM audio samples.

        Args:
            multiframe: List of audio token IDs.
            count: Current token count (for logging).

        Returns:
            Audio samples as int16 array or None if conversion fails.
        """
        try:
            audio_bytes = convert_to_audio(multiframe, count)
            if audio_bytes is None:
                logger.warning("Audio conversion returned None")
                return None

            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            return audio
        except Exception as e:
            logger.error(f"Buffer conversion failed: {e}")
            traceback.print_exc()
            return None

    def _token_decoder_sync(
        self,
        token_gen: Generator[str, None, None],
    ) -> Generator[NDArray[np.int16], None, None]:
        """
        Decode streaming tokens into audio chunks.

        Buffers tokens and yields audio at regular intervals using Orpheus's
        multi-frame encoding (28 tokens, output every 7 tokens).

        Args:
            token_gen: Generator yielding token strings.

        Yields:
            Audio chunks as numpy arrays of PCM samples.
        """
        buffer: list[int] = []
        count = 0

        logger.debug("Starting token decoding")
        for token_text in token_gen:
            token_id = self._turn_token_into_id(token_text, count)
            if token_id is None or token_id <= 0:
                continue

            buffer.append(token_id)
            count += 1

            # Generate audio every 7 tokens after initial 28
            if count % 7 == 0 and count > 27:
                buffer_to_process = buffer[-28:]
                audio_samples = self._convert_buffer(buffer_to_process, count)
                if audio_samples is not None and audio_samples.size > 0:
                    yield audio_samples

    def stream_tts_sync(
        self,
        text: str,
        options: OrpheusTTSOptions | None = None,
    ) -> Generator[tuple[int, NDArray[np.int16]], None, None]:
        """
        Synchronous streaming TTS generation.

        Args:
            text: Text to convert to speech.
            options: Optional TTS configuration.

        Yields:
            Tuples of (sample_rate, audio_chunk).
        """
        opts = options or self.options

        try:
            token_gen = self._generate_tokens_sync(text, opts)
            for audio_chunk in self._token_decoder_sync(token_gen):
                yield opts.sample_rate, audio_chunk
        except Exception as e:
            logger.error(f"Sync streaming error: {e}")
            traceback.print_exc()

    async def stream_tts(
        self,
        text: str,
        options: OrpheusTTSOptions | None = None,
    ) -> AsyncGenerator[tuple[int, NDArray[np.int16]], None]:
        """
        Asynchronous streaming TTS generation.

        This method enables the async for pattern:
            async for sample_rate, chunk in model.stream_tts("Hello"):
                # Process each chunk as it arrives
                await process_audio(chunk, sample_rate)

        Args:
            text: Text to convert to speech.
            options: Optional TTS configuration.

        Yields:
            Tuples of (sample_rate, audio_chunk) as they become available.
        """
        opts = options or self.options
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[int, NDArray[np.int16]] | None] = asyncio.Queue()

        def worker():
            """Background thread worker that feeds the async queue."""
            try:
                for sample_rate, chunk in self.stream_tts_sync(text, opts):
                    # Send chunk to async queue
                    asyncio.run_coroutine_threadsafe(
                        queue.put((sample_rate, chunk)), loop
                    )
            except Exception as e:
                logger.error(f"Worker thread error: {e}")
                traceback.print_exc()
            finally:
                # Signal completion
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        # Start background worker
        threading.Thread(target=worker, daemon=True).start()

        # Yield chunks as they arrive
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    async def tts(
        self,
        text: str,
        options: OrpheusTTSOptions | None = None,
    ) -> bytes:
        """
        Perform complete text-to-speech conversion (async).

        This method waits for the full synthesis to complete and returns
        the complete audio as bytes.

        Args:
            text: Text to convert to speech.
            options: Optional TTS configuration.

        Returns:
            Complete audio as bytes (PCM int16 format).
        """
        opts = options or self.options
        audio_chunks: list[NDArray[np.int16]] = []

        try:
            async for sample_rate, chunk in self.stream_tts(text, opts):
                audio_chunks.append(chunk)
        except Exception as e:
            logger.error(f"TTS error: {e}")
            traceback.print_exc()

        if audio_chunks:
            audio = np.concatenate(audio_chunks)
        else:
            audio = np.zeros(0, dtype=np.int16)

        return audio.tobytes()

    def tts_blocking(
        self,
        text: str,
        options: OrpheusTTSOptions | None = None,
    ) -> tuple[int, NDArray[np.int16]]:
        """
        Synchronous blocking TTS (for non-async contexts).

        Args:
            text: Text to convert to speech.
            options: Optional TTS configuration.

        Returns:
            Tuple of (sample_rate, audio_array).
        """
        opts = options or self.options
        audio_chunks: list[NDArray[np.int16]] = []

        try:
            for sample_rate, chunk in self.stream_tts_sync(text, opts):
                audio_chunks.append(chunk)
        except Exception as e:
            logger.error(f"Blocking TTS error: {e}")
            traceback.print_exc()

        if audio_chunks:
            audio = np.concatenate(audio_chunks)
        else:
            audio = np.zeros(0, dtype=np.int16)

        return opts.sample_rate, audio
