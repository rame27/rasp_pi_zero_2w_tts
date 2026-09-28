from __future__ import annotations

import asyncio
import edge_tts

from config import settings


_ALL_VOICES_CACHE: list[dict] | None = None


def _is_allowed(locale: str, short_name: str) -> bool:
    """Keep only voices from allowed language prefixes.

    For English, only the curated allowlist is kept (all other en-* voices
    are hidden); other languages keep all their voices.
    """
    if not any(locale.lower().startswith(prefix.lower()) for prefix in settings.allowed_locales):
        return False
    if locale.lower().startswith("en"):
        return short_name in settings.keep_english_voices
    return True


def _sort_key(v: dict) -> tuple:
    """English voices first, in the curated allowlist order; then other
    languages sorted by locale and name."""
    short = v["ShortName"]
    if short in settings.keep_english_voices:
        return (0, settings.keep_english_voices.index(short))
    return (1, v["Locale"], short)


async def load_all_voices() -> list[dict]:
    global _ALL_VOICES_CACHE
    if _ALL_VOICES_CACHE is not None:
        return _ALL_VOICES_CACHE
    voices = await edge_tts.list_voices()
    result = []
    for v in voices:
        if not _is_allowed(v["Locale"], v["ShortName"]):
            continue
        result.append(
            {
                "ShortName": v["ShortName"],
                "Gender": v["Gender"],
                "Locale": v["Locale"],
                "FriendlyName": v["FriendlyName"],
                "VoiceType": v.get("VoiceType"),
            }
        )
    result.sort(key=_sort_key)
    _ALL_VOICES_CACHE = result
    return result


def load_all_voices_sync() -> list[dict]:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(load_all_voices())
