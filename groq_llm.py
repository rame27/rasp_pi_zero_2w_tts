from __future__ import annotations

import json
import logging
import re

from groq import Groq
from pydantic import ValidationError

from config import settings
from models import RoleInfo, SegmentInfo, StoryPlan

log = logging.getLogger(__name__)

MAX_LLM_CHARS = 20000
DEFAULT_VOICE = "en-US-AndrewNeural"


def sanitize_name(name: str) -> str:
    """Lowercase; keep alphanumerics, dash, underscore; collapse separators."""
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()
    return s or "story"


def slice_text(text: str, start_word: int, end_word: int) -> str:
    words = text.split()
    return " ".join(words[start_word:end_word])


def _word_count(text: str) -> int:
    return len(text.split())


def _clamp(idx: int, word_count: int) -> int:
    return max(0, min(idx, word_count))


def _validate_plan(data: dict, voices: set[str], word_count: int) -> StoryPlan:
    """Validate/normalize LLM JSON into a StoryPlan."""
    roles_raw = data.get("roles") or []
    roles: list[RoleInfo] = []
    for r in roles_raw:
        role = str(r.get("role") or "Narrator").strip() or "Narrator"
        voice = str(r.get("voice") or DEFAULT_VOICE).strip()
        if voice not in voices:
            log.warning("Voice %s not in catalog, falling back to %s", voice, DEFAULT_VOICE)
            voice = DEFAULT_VOICE
        roles.append(RoleInfo(role=role, voice=voice))
    if not roles:
        roles = [RoleInfo(role="Narrator", voice=DEFAULT_VOICE)]

    segments_raw = data.get("segments") or []
    segments: list[SegmentInfo] = []
    for s in segments_raw:
        role = str(s.get("role") or "Narrator").strip() or "Narrator"
        start = _clamp(int(s.get("start_word", 0)), word_count)
        end = _clamp(int(s.get("end_word", start)), word_count)
        if end <= start:
            continue
        segments.append(SegmentInfo(role=role, start_word=start, end_word=end))
    segments.sort(key=lambda x: x.start_word)

    narrator_voice = next((r.voice for r in roles if r.role.lower() == "narrator"), DEFAULT_VOICE)
    filled: list[SegmentInfo] = []
    cursor = 0
    for seg in segments:
        if seg.start_word > cursor:
            filled.append(SegmentInfo(role="Narrator", start_word=cursor, end_word=seg.start_word))
        filled.append(seg)
        cursor = max(cursor, seg.end_word)
    if cursor < word_count:
        filled.append(SegmentInfo(role="Narrator", start_word=cursor, end_word=word_count))
    segments = filled

    story_name = sanitize_name(str(data.get("story_name") or "story"))
    return StoryPlan(story_name=story_name, roles=roles, segments=segments)


def _build_prompt(text: str, title: str | None, voices: list[str]) -> str:
    voice_list = ", ".join(voices)
    return f"""You are a story narration planner. Analyze the story text below and plan a multi-voice narration.

Available edge-tts voices (pick ONLY from these): {voice_list}

Return JSON only with this exact shape:
{{
  "story_name": "short clean title for the story",
  "roles": [{{"role": "Narrator", "voice": "<voice>"}}, {{"role": "<character name>", "voice": "<voice>"}}],
  "segments": [{{"role": "<role name>", "start_word": <int>, "end_word": <int>}}]
}}

Rules:
- Always include a Narrator role for non-dialogue text.
- Assign each main character a distinct voice from the list.
- Use character names from the text (including any Dramatis Personae or character list) to assign distinct voices.
- The text may contain a Project Gutenberg license header/footer and a CONTENTS or front-matter section. Ignore those when planning segments — plan segments only for the actual story text.
- segments must cover the ENTIRE story in order, using word indices (0-based, end exclusive) into the text.
- Every word of the story must belong to exactly one segment.
- Use the narrator for narration, characters for their dialogue/actions.
- Split the story into MULTIPLE segments of roughly 150-400 words each. Do NOT merge the whole story into a single Narrator segment.
- Alternate Narrator segments with character dialogue segments so the narration is lively.
- For a typical story, produce at least 3-5 segments.

Story title: {title or "unknown"}

Story text:
{text}"""


def analyze_story(text: str, title: str | None, voices: list[str]) -> StoryPlan:
    """Call Groq, parse and validate the plan. Retries once on failure."""
    client = Groq(api_key=settings.groq_api_key)
    truncated = text[:MAX_LLM_CHARS]
    word_count = _word_count(truncated)
    voice_set = set(voices)
    prompt = _build_prompt(truncated, title, voices)
    log.info("Groq analyze: model=%s chars=%d words=%d voices=%d", settings.groq_model, len(truncated), word_count, len(voices))
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=4000,
            )
            content = resp.choices[0].message.content
            log.info("Groq response (attempt %d): %s", attempt + 1, content[:2000])
            data = json.loads(content)
            plan = _validate_plan(data, voice_set, word_count)
            plan.llm_response = data
            log.info("Groq plan: story=%s roles=%d segments=%d", plan.story_name, len(plan.roles), len(plan.segments))
            return plan
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            last_error = exc
            log.warning("Groq attempt %d failed: %s", attempt + 1, exc)
    raise RuntimeError(f"Groq analysis failed: {last_error}")