from __future__ import annotations

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    max_text_chars: int = 50000
    export_dir: str = "/tmp/app_edgetts_exports"
    export_ttl: int = 300  # 5 minutes
    export_max: int = 10
    # Delay before speech content starts (ms). Small safety margin now that
    # the A2DP transport is kept alive by bluealsa --keep-alive.
    # Initial tone duration baked into the audio (ms). A faint tone keeps the
    # BT speaker's audio path active so it does not clip the first word.
    playback_delay_ms: int = 500
    # Tone duration inserted between sentences (ms).
    sentence_padding_ms: int = 400
    # Low-level tone used as padding (keeps the speaker from pausing/clipping).
    padding_tone_hz: int = 1000
    padding_tone_db: int = -45
    # Max sentences per segment (balanced 2-3 within a paragraph).
    sentences_per_segment: int = 3
    # Bluetooth speaker MAC - reconnected automatically if it sleeps.
    speaker_mac: str = ""
    # Persisted list of previously connected Bluetooth devices (fast connect).
    bt_history_file: str = "/root/app_edgetts/bt_history.json"
    # Groq LLM for story role analysis (free tier).
    groq_api_key: str = "********************************"
    groq_model: str = "openai/gpt-oss-120b"
    # Where generated story MP3s are stored.
    story_dir: str = "/media/tts_mp3"
    # Where URL-speak text + MP3 archives are stored (listed in the stories tab).
    speak_text_dir: str = "/media/tts_mp3/speak_texts"
    # Only show voices from these language prefixes in the UI.
    allowed_locales: list[str] = ["en", "hi", "te", "kn"]
    # English voices to keep in the UI (all other en-* voices are hidden).
    keep_english_voices: list[str] = [
        "en-US-EmmaMultilingualNeural",
        "en-US-BrianMultilingualNeural",
        "en-US-AvaMultilingualNeural",
        "en-US-AndrewMultilingualNeural",
        "en-GB-SoniaNeural",
        "en-GB-LibbyNeural",
        "en-TZ-ImaniNeural",
        "en-US-BrianNeural",
        "en-US-EmmaNeural",
        "en-US-AndrewNeural",
        "en-US-AvaNeural",
    ]


settings = Settings()
