from __future__ import annotations

import configparser
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_config_ini() -> dict:
    """Load static config from config.ini (project root)."""
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), "config.ini"))

    result: dict = {}

    # Server
    if config.has_section("server"):
        result["host"] = config.get("server", "host", fallback="0.0.0.0")
        result["port"] = config.getint("server", "port", fallback=8000)
        result["log_level"] = config.get("server", "log_level", fallback="INFO")

    # Groq
    if config.has_section("groq"):
        result["groq_api_key"] = config.get("groq", "api_key", fallback="********************************")
        result["groq_model"] = config.get("groq", "model", fallback="openai/gpt-oss-120b")

    # Bluetooth
    if config.has_section("bluetooth"):
        result["speaker_mac"] = config.get("bluetooth", "speaker_mac", fallback="")

    # Storage
    if config.has_section("storage"):
        result["story_dir"] = config.get("storage", "story_dir", fallback="/media/tts_mp3")
        result["speak_text_dir"] = config.get("storage", "speak_text_dir", fallback="/media/tts_mp3/speak_texts")

    # Voices
    if config.has_section("voices"):
        locales = config.get("voices", "allowed_locales", fallback="en, hi, te, kn")
        result["allowed_locales"] = [s.strip() for s in locales.split(",") if s.strip()]
        voices = config.get("voices", "keep_english_voices", fallback="")
        result["keep_english_voices"] = [s.strip() for s in voices.split(",") if s.strip()]

    return result


# Load static config from config.ini; Pydantic Settings will still
# override via APP_ env vars at runtime.
_config_ini = _load_config_ini()


class Settings(BaseSettings):
    # APP_ prefixed env vars override config.ini at runtime (as documented
    # in AGENTS.md); without a prefix, generic vars like PORT/HOST in the
    # shell would silently leak into settings.
    model_config = SettingsConfigDict(env_prefix="APP_")

    host: str = _config_ini.get("host", "0.0.0.0")
    port: int = _config_ini.get("port", 8000)
    # Root log verbosity: INFO hides per-request lines, DEBUG shows them
    # (uvicorn access records + chatty UI polls are logged at DEBUG).
    log_level: str = _config_ini.get("log_level", "INFO")
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
    speaker_mac: str = _config_ini.get("speaker_mac", "")
    # Persisted list of previously connected Bluetooth devices (fast connect).
    bt_history_file: str = "/root/app_edgetts/bt_history.json"
    # Groq LLM for story role analysis (free tier).
    groq_api_key: str = _config_ini.get("groq_api_key", "********************************")
    groq_model: str = _config_ini.get("groq_model", "openai/gpt-oss-120b")
    # Where generated story MP3s are stored.
    story_dir: str = _config_ini.get("story_dir", "/media/tts_mp3")
    # Where URL-speak text + MP3 archives are stored (listed in the stories tab).
    speak_text_dir: str = _config_ini.get("speak_text_dir", "/media/tts_mp3/speak_texts")
    # Only show voices from these language prefixes in the UI.
    allowed_locales: list[str] = _config_ini.get("allowed_locales", ["en", "hi", "te", "kn"])
    # English voices to keep in the UI (all other en-* voices are hidden).
    keep_english_voices: list[str] = _config_ini.get(
        "keep_english_voices",
        [
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
        ],
    )


settings = Settings()
