from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
import uuid

import edge_tts

from config import settings
from models import RoleInfo, SegmentInfo, StoryInfo, StoryPlan
from services import text_source
from services.audio import _edge_save_with_retry, _ensure_speaker_connected, _write_speak_metadata, is_segment_file
from services.groq_llm import DEFAULT_VOICE, slice_text

log = logging.getLogger(__name__)

_tasks: dict[str, dict] = {}

_combine_lock = threading.Lock()

_play_proc: asyncio.subprocess.Process | None = None


class StoryIncomplete(Exception):
    """Raised by combine_story when synthesis hasn't produced every segment yet.

    combine_story would otherwise happily concatenate whatever segment files
    happen to exist (e.g. after a crash or restart mid-generation), cache the
    result as combined.mp3, and delete the remaining segments as if the story
    were finished. Callers should catch this and play the available segments
    directly instead.
    """


def set_play_proc(proc) -> None:
    global _play_proc
    _play_proc = proc


def clear_play_proc(proc) -> None:
    global _play_proc
    if _play_proc is proc:
        _play_proc = None


def stop_playback() -> None:
    global _play_proc
    if _play_state["name"] is not None:
        combined_play_stopped()
    if _play_proc is not None and _play_proc.returncode is None:
        try:
            _play_proc.kill()
        except ProcessLookupError:
            pass
    _play_proc = None


# --- Combined-playback position tracking (resume support) -------------------
# Remembers where combined.mp3 playback stopped (wall-clock, ~1s accuracy)
# so the next play can offer resume-vs-start. The position is heartbeat-
# persisted every few seconds, so even an abrupt kill leaves the nearest
# stop point behind. Segment-list playback is excluded: unfinished stories
# already resume per segment via playback_done.

RESUME_MIN_MS = 3000
_HEARTBEAT_S = 5.0

_play_state: dict = {"name": None, "offset_ms": 0, "started": 0.0, "stopped": False}
_play_heartbeat_task: asyncio.Task | None = None


def _play_position_ms() -> int:
    """Current combined-playback position: starting offset + elapsed."""
    st = _play_state
    if st["name"] is None:
        return 0
    return st["offset_ms"] + int((time.monotonic() - st["started"]) * 1000)


def _touch_combined_meta(name: str, **fields) -> None:
    """Read-modify-write metadata.json, preserving all other keys."""
    try:
        safe = _safe_name(name)
    except ValueError:
        return
    meta_path = os.path.join(_story_path(safe), "metadata.json")
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    meta.update(fields)
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except OSError:
        pass


def _probe_duration_ms(path: str) -> int | None:
    """Duration of an audio file in ms via ffprobe (None when unknown)."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        seconds = float(proc.stdout.strip())
        if seconds > 0:
            return int(seconds * 1000)
    except (subprocess.SubprocessError, ValueError, OSError):
        pass
    return None


async def _position_heartbeat(name: str) -> None:
    """Persist the playback position every few seconds until cancelled."""
    try:
        while True:
            await asyncio.sleep(_HEARTBEAT_S)
            _touch_combined_meta(name, combined_position_ms=_play_position_ms())
    except asyncio.CancelledError:
        pass


def _stop_heartbeat() -> None:
    global _play_heartbeat_task
    task, _play_heartbeat_task = _play_heartbeat_task, None
    if task is not None and not task.done():
        task.cancel()


def combined_play_started(name: str, offset_ms: int = 0) -> None:
    """Begin tracking a combined.mp3 playback from the given offset."""
    try:
        name = _safe_name(name)
    except ValueError:
        return
    _stop_heartbeat()
    _play_state.update({
        "name": name,
        "offset_ms": max(0, offset_ms),
        "started": time.monotonic(),
        "stopped": False,
    })
    global _play_heartbeat_task
    _play_heartbeat_task = asyncio.create_task(_position_heartbeat(name))


def combined_play_stopped() -> None:
    """Persist the stop point of a combined playback (Stop button / switch)."""
    if _play_state["name"] is None:
        return
    _play_state["stopped"] = True
    _touch_combined_meta(_play_state["name"], combined_position_ms=_play_position_ms())
    _stop_heartbeat()
    _play_state["name"] = None


def combined_play_finished() -> None:
    """Clear the saved position after a playback that ran to the end.

    A playback ended by stop_playback() keeps its persisted position;
    only a natural finish clears it.
    """
    _stop_heartbeat()
    name, stopped = _play_state["name"], _play_state["stopped"]
    _play_state["name"] = None
    if name is not None and not stopped:
        _touch_combined_meta(name, combined_position_ms=0)


def ensure_combined_duration(name: str, combined_path: str) -> None:
    """Probe and store the combined duration once (covers older archives)."""
    try:
        safe = _safe_name(name)
    except ValueError:
        return
    meta_path = os.path.join(_story_path(safe), "metadata.json")
    try:
        with open(meta_path, encoding="utf-8") as f:
            if type(json.load(f).get("combined_duration_ms")) is int:
                return
    except (OSError, json.JSONDecodeError):
        return
    duration_ms = _probe_duration_ms(combined_path)
    if duration_ms is not None:
        _touch_combined_meta(safe, combined_duration_ms=duration_ms)


def clamp_resume_offset(name: str, offset_ms: int) -> int:
    """Clamp a resume offset into [0, duration); past-the-end restarts."""
    if offset_ms <= 0:
        return 0
    try:
        safe = _safe_name(name)
    except ValueError:
        return 0
    try:
        with open(os.path.join(_story_path(safe), "metadata.json"), encoding="utf-8") as f:
            duration_ms = json.load(f).get("combined_duration_ms", 0)
    except (OSError, json.JSONDecodeError):
        duration_ms = 0
    if type(duration_ms) is int and duration_ms > 0 and offset_ms >= duration_ms:
        return 0
    return offset_ms


def cancel_all() -> None:
    """Cancel all running generation tasks and stop any playback."""
    for task in _tasks.values():
        if task["state"] == "running":
            task["cancel"] = True
    stop_playback()


def _safe_name(name: str) -> str:
    if not name or name == "." or name == "..":
        raise ValueError("Invalid story name")
    if os.sep in name or "\\" in name:
        raise ValueError("Invalid story name")
    if "'" in name or "\n" in name or "\r" in name:
        raise ValueError("Invalid story name")
    return name


def speak_name_from_url(url: str) -> str:
    """Derive a filesystem-safe speak name from a URL (basename, extension stripped).

    Returns "" when the URL has no usable basename (caller falls back to a hex id).
    """
    parsed = urllib.parse.urlparse(str(url))
    base = os.path.basename(parsed.path.rstrip("/"))
    if not base:
        return ""
    name = re.sub(r"[^A-Za-z0-9._-]", "-", base).strip(".-")
    root, _ext = os.path.splitext(name)
    name = root or name
    if not name or name in (".", ".."):
        return ""
    return name


def _sanitize_speak_name(raw: str) -> str:
    """Sanitize a title/derived name the same way as URL basenames."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.strip()).strip(".-")
    root, _ext = os.path.splitext(name)
    name = root or name
    if not name or name in (".", ".."):
        return ""
    return name[:80]


def speak_name_from_text(text: str, title: str | None = None) -> str:
    """Derive a filesystem-safe archive name for text input.

    Uses the user-supplied title when present, else the first 4 words
    of the text. Returns "" when nothing usable remains.
    """
    source = (title or "").strip()
    if not source:
        words = (text or "").split()
        source = " ".join(words[:4])
    if not source.strip():
        return ""
    return _sanitize_speak_name(source)


def unique_speak_name(base: str) -> str:
    """Make an archive name unique under speak_text_dir (append -2, -3...)."""
    candidate = base
    suffix = 2
    while os.path.isdir(os.path.join(settings.speak_text_dir, candidate)):
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def speak_name_for_url(url: str, title: str | None = None) -> str:
    """Archive name for a URL speak: user-supplied title wins, else URL basename.

    Returns "" when neither yields a usable name (caller falls back to a hex id).
    """
    if (title or "").strip():
        return _sanitize_speak_name(title)
    return speak_name_from_url(url)


def stored_voice(name: str) -> str | None:
    """Voice used by an existing URL-speak archive (from its metadata.json)."""
    meta_path = os.path.join(settings.speak_text_dir, name, "metadata.json")
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        roles = meta.get("roles") or []
        if roles:
            return roles[0].get("voice")
    except (OSError, json.JSONDecodeError, IndexError):
        pass
    return None


def stored_text(name: str) -> str:
    """Text stored in an existing speak archive (metadata.json "source").

    Falls back to legacy source.txt for archives written before the
    duplicate file was removed. Returns "" when neither exists (caller
    falls back to fetching the URL).
    """
    try:
        with open(os.path.join(settings.speak_text_dir, name, "metadata.json"), encoding="utf-8") as f:
            text = json.load(f).get("source", "")
        if text:
            return text
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    try:
        with open(os.path.join(settings.speak_text_dir, name, "source.txt"), encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def stored_plan(name: str) -> StoryPlan | None:
    """Reuse the plan of an already-processed URL-speak archive (skip the LLM).

    Returns None when the archive has no usable plan (e.g. a plain URL speak
    that was never analyzed as a story).
    """
    meta_path = os.path.join(settings.speak_text_dir, name, "metadata.json")
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        roles = [RoleInfo(role=r["role"], voice=r["voice"]) for r in meta.get("roles", [])]
        segments = [
            SegmentInfo(role=s["role"], start_word=s["start_word"], end_word=s["end_word"])
            for s in meta.get("segments", [])
        ]
        if not roles or not segments:
            return None
        return StoryPlan(
            story_name=meta.get("story_name", name),
            roles=roles,
            segments=segments,
            llm_response=meta.get("llm_response"),
        )
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def create_task() -> str:
    now = time.time()
    stale = [tid for tid, t in _tasks.items() if t["state"] != "running" and now - t.get("_ts", 0) > 3600]
    for tid in stale:
        del _tasks[tid]
    task_id = uuid.uuid4().hex[:12]
    _tasks[task_id] = {"state": "running", "total": 0, "done": 0, "current_role": None, "error": None, "cancel": False, "_ts": now}
    return task_id


def get_task(task_id: str) -> dict | None:
    return _tasks.get(task_id)


def generation_running() -> bool:
    return any(t["state"] == "running" for t in _tasks.values())


def _story_path(name: str) -> str:
    path = os.path.join(settings.story_dir, name)
    if not os.path.isdir(path):
        # URL-speak archives live under speak_text_dir but reuse the story layout.
        alt = os.path.join(settings.speak_text_dir, name)
        if os.path.isdir(alt):
            return alt
    return path


def save_speak_archive(speak_id: str, text: str, voice: str) -> str:
    """Create the archive dir for a speak: store the text in metadata, return the dir path.

    The text lives in metadata.json ("source"); no separate source.txt is
    written. Metadata is written immediately so the archive shows in the
    Stories tab even if synthesis is stopped or interrupted; _run_archived
    fills in the real segment_count once synthesis completes.
    """
    path = os.path.join(settings.speak_text_dir, speak_id)
    os.makedirs(os.path.join(path, "segments"), exist_ok=True)
    _write_speak_metadata(path, speak_id, text, voice)
    return path


def _segments_path(name: str) -> str:
    return os.path.join(_story_path(name), "segments")


def _write_metadata(
    name: str, source: str, plan: StoryPlan, segments: list, complete: bool = False
) -> None:
    """Write metadata.json for a generated story.

    complete is True only when generation ran to completion without being
    cancelled or failing. combine_story refuses to combine (and callers fall
    back to playing the segments directly) while it's False, so a partial set
    of segment files never gets treated as the whole story.
    """
    metadata = {
        "story_name": name,
        "source": source,
        "created": time.time(),
        "roles": [r.model_dump() for r in plan.roles],
        "segments": [s.model_dump() for s in segments],
        "model": settings.groq_model,
        "llm_response": plan.llm_response,
        "complete": complete,
    }
    with open(os.path.join(_story_path(name), "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


async def _synth_segment(text: str, voice: str, out_path: str) -> None:
    await _edge_save_with_retry(text, voice, out_path)


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

CHUNK_MAX_SENTENCES = 3


def _chunk_sentences(text: str, max_sentences: int = CHUNK_MAX_SENTENCES) -> list[str]:
    """Split text into chunks of up to max_sentences sentences."""
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    return [" ".join(sentences[i:i + max_sentences]) for i in range(0, len(sentences), max_sentences)]


async def _play_file(path: str, task: dict) -> None:
    await _ensure_speaker_connected()
    proc = await asyncio.create_subprocess_exec(
        "ffplay", "-nodisp", "-autoexit", "-loglevel", "error", path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    set_play_proc(proc)
    if task.get("cancel"):
        stop_playback()
        return
    await proc.wait()
    clear_play_proc(proc)


async def generate_story(
    url: str | None, text: str | None, story_name: str, plan: StoryPlan, task_id: str,
    pipeline=None,
) -> None:
    task = _tasks[task_id]
    cancelled = False
    name: str | None = None
    segments: list = []
    source = url or "text"
    try:
        # Already-processed URL: replay combined.mp3 or resume the partial
        # segments instead of re-synthesizing from scratch.
        if url:
            url_name = speak_name_from_url(url)
            if url_name:
                url_archive = os.path.join(settings.speak_text_dir, url_name)
                combined_path = os.path.join(url_archive, "combined.mp3")
                if os.path.isfile(combined_path):
                    if pipeline is not None and pipeline.begin():
                        pipeline.start_combined(combined_path, url_name)
                    task["total"] = 1
                    task["done"] = 1
                    task["state"] = "done"
                    log.info("URL %s already processed; replaying combined.mp3", url_name)
                    return
                seg_dir = os.path.join(url_archive, "segments")
                existing = (
                    sorted(f for f in os.listdir(seg_dir) if is_segment_file(f))
                    if os.path.isdir(seg_dir)
                    else []
                )
                if existing:
                    src_text = stored_text(url_name)
                    if src_text:
                        voice = stored_voice(url_name) or DEFAULT_VOICE
                        if pipeline is not None and pipeline.begin():
                            pipeline.start(src_text, voice, url_name, url_archive)
                        task["total"] = 1
                        task["done"] = 1
                        task["state"] = "done"
                        log.info("URL %s already processed; resuming from segment %d", url_name, len(existing) + 1)
                        return
                    # Segments but no stored text (interrupted before metadata
                    # was written): fall through and regenerate from scratch.
        name = _safe_name(story_name)
        if text is None:
            # Raw text (uncurated) so the LLM's word offsets match what it analyzed.
            text, _ = text_source.fetch_text(url, curate=False)
        os.makedirs(_story_path(name), exist_ok=True)
        word_count = len(text.split())
        segments = [s for s in plan.segments if s.end_word <= word_count]
        if not segments:
            task["state"] = "failed"
            task["error"] = "no segments in plan"
            log.warning("Story %s generation failed: no segments in plan", name)
            return
        task["total"] = len(segments)
        seg_dir = _segments_path(name)
        os.makedirs(seg_dir, exist_ok=True)
        # Clean up any previous partial/cached artifacts before generating
        for f in os.listdir(seg_dir):
            try:
                os.remove(os.path.join(seg_dir, f))
            except OSError:
                pass
        combined_path = os.path.join(_story_path(name), "combined.mp3")
        try:
            os.remove(combined_path)
        except OSError:
            pass
        voice_by_role = {r.role: r.voice for r in plan.roles}
        # Build synthesis units: each segment split into 2-3 sentence chunks.
        chunks: list[tuple[int, str, str, str]] = []
        for i, seg in enumerate(segments, start=1):
            seg_text = slice_text(text, seg.start_word, seg.end_word)
            # Apply the curation filters to the slice so boilerplate (license
            # header/footer, CONTENTS, front matter) is ignored in the audio.
            seg_text = text_source.curate_text(seg_text)
            if not seg_text.strip():
                continue
            role_slug = "".join(c for c in seg.role if c.isalnum() or c in "-_") or "role"
            for ci, chunk_text in enumerate(_chunk_sentences(seg_text), start=1):
                out_path = os.path.join(seg_dir, f"{i:03d}_{role_slug}_{ci:02d}.mp3")
                chunks.append((i, seg.role, chunk_text, out_path))
        pending: asyncio.Task | None = None
        for idx, (seg_idx, role, chunk_text, out_path) in enumerate(chunks, start=1):
            if task.get("cancel"):
                cancelled = True
                break
            task["current_role"] = role
            if pending is not None:
                # This chunk's synthesis was prefetched while the previous one played.
                await pending
                pending = None
            else:
                # First chunk: nothing prefetched yet, synthesize inline.
                await _synth_segment(chunk_text, voice_by_role.get(role, DEFAULT_VOICE), out_path)
            # Prefetch the next chunk's synthesis while this one plays.
            if idx < len(chunks):
                nxt_seg, nxt_role, nxt_text, nxt_path = chunks[idx]
                pending = asyncio.create_task(
                    _synth_segment(nxt_text, voice_by_role.get(nxt_role, DEFAULT_VOICE), nxt_path)
                )
            log.info("Chunk %d/%d (segment %d, %s), playing", idx, len(chunks), seg_idx, role)
            await _play_file(out_path, task)
            if task.get("cancel"):
                cancelled = True
                if pending is not None:
                    pending.cancel()
                    try:
                        await pending
                    except asyncio.CancelledError:
                        pass
                break
            # Mark segment progress when its last chunk finishes playing.
            if idx == len(chunks) or chunks[idx][0] != seg_idx:
                task["done"] = seg_idx
        if not cancelled:
            task["done"] = len(segments)
        _write_metadata(name, source, plan, segments, complete=not cancelled)
        if cancelled:
            task["state"] = "cancelled"
            task["error"] = "cancelled by user"
            log.info("Story %s generation cancelled (%d/%d segments)", name, task["done"], len(segments))
        else:
            task["state"] = "done"
            log.info("Story %s generated (%d segments)", name, len(segments))
    except Exception as exc:
        task["state"] = "failed"
        task["error"] = str(exc)
        log.exception("Story generation failed for %s", story_name)
        if name and segments:
            _write_metadata(name, source, plan, segments)


def _record_segment_count(name: str, count: int) -> None:
    """Persist the clip count in metadata.json so list_stories survives segment cleanup."""
    meta_path = os.path.join(_story_path(name), "metadata.json")
    if not os.path.isfile(meta_path):
        return
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["segment_count"] = count
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except (OSError, json.JSONDecodeError):
        pass


def _is_complete(name: str) -> bool:
    """Whether generation for this story/archive ran to completion.

    Metadata missing the "complete" key predates this check and is treated as
    complete (old, already-finished data shouldn't stop being playable); a
    key present and False means generation is still running or was
    interrupted before finishing.
    """
    meta_path = os.path.join(_story_path(name), "metadata.json")
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f).get("complete", True)
    except (OSError, json.JSONDecodeError):
        return True


def combine_story(name: str) -> str:
    """Concatenate segment MP3s into combined.mp3 (cached).

    Raises StoryIncomplete instead of combining when generation hasn't
    finished (interrupted/crashed run, or still in progress) - callers should
    fall back to playing the available segments directly.
    """
    name = _safe_name(name)
    combined = os.path.join(_story_path(name), "combined.mp3")
    if os.path.exists(combined):
        return combined
    with _combine_lock:
        if os.path.exists(combined):
            return combined
        if not _is_complete(name):
            raise StoryIncomplete(f"Story {name} is not fully generated yet")
        seg_dir = _segments_path(name)
        files = sorted(f for f in os.listdir(seg_dir) if is_segment_file(f))
        if not files:
            raise FileNotFoundError(f"No segments for story {name}")
        list_path = os.path.join(_story_path(name), "concat.txt")
        try:
            with open(list_path, "w", encoding="utf-8") as f:
                for fn in files:
                    f.write(f"file '{os.path.join(seg_dir, fn)}'\n")
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", combined],
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            try:
                os.remove(combined)
            except OSError:
                pass
            raise
        finally:
            try:
                os.remove(list_path)
            except OSError:
                pass
        # Record the clip count, then clean up the individual segments.
        # The text already lives in metadata.json ("source"), so the legacy
        # source.txt duplicate and any interrupted-run *.raw.mp3 leftovers go
        # too: a finished archive keeps only combined.mp3 + metadata.json.
        # The combined duration is stored for the resume UI ("12:34 of 45:10").
        _record_segment_count(name, len(files))
        for fn in files:
            try:
                os.remove(os.path.join(seg_dir, fn))
            except OSError:
                pass
        try:
            os.remove(os.path.join(_story_path(name), "source.txt"))
        except OSError:
            pass
        try:
            leftovers = [f for f in os.listdir(seg_dir) if f.endswith(".raw.mp3")]
        except OSError:
            leftovers = []
        for fn in leftovers:
            try:
                os.remove(os.path.join(seg_dir, fn))
            except OSError:
                pass
        duration_ms = _probe_duration_ms(combined)
        if duration_ms is not None:
            _touch_combined_meta(name, combined_duration_ms=duration_ms)
        return combined


def segments_playback_list(name: str) -> str:
    """Write a temp ffmpeg-concat list of the currently available segment files.

    Used to play an incomplete story's segments directly (in order, gapless)
    without caching them into combined.mp3 - the segment files are left in
    place so generation can resume later. Caller is responsible for deleting
    the returned list file once playback finishes.
    """
    name = _safe_name(name)
    seg_dir = _segments_path(name)
    files = sorted(f for f in os.listdir(seg_dir) if is_segment_file(f)) if os.path.isdir(seg_dir) else []
    if not files:
        raise FileNotFoundError(f"No segments for story {name}")
    fd, list_path = tempfile.mkstemp(suffix=".txt", prefix="playback_", dir=_story_path(name))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for fn in files:
            f.write(f"file '{os.path.join(seg_dir, fn)}'\n")
    return list_path


def list_stories() -> list[StoryInfo]:
    stories = []
    # Regular generated stories live in story_dir; URL-speak archives in speak_text_dir.
    for base in (settings.story_dir, settings.speak_text_dir):
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            meta_path = os.path.join(base, name, "metadata.json")
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                seg_dir = _segments_path(name)
                dir_count = len([x for x in os.listdir(seg_dir) if is_segment_file(x)]) if os.path.isdir(seg_dir) else 0
                seg_count = meta.get("segment_count", dir_count)
                resume_ms = meta.get("combined_position_ms", 0)
                duration_ms = meta.get("combined_duration_ms", 0)
                stories.append(StoryInfo(
                    name=name,
                    roles=[r["role"] for r in meta.get("roles", [])],
                    segment_count=seg_count,
                    created=meta.get("created", 0.0),
                    resume_ms=resume_ms if type(resume_ms) is int and resume_ms > 0 else 0,
                    duration_ms=duration_ms if type(duration_ms) is int and duration_ms > 0 else 0,
                ))
            except (OSError, json.JSONDecodeError, KeyError):
                continue
    return stories


def delete_story(name: str) -> None:
    name = _safe_name(name)
    path = _story_path(name)
    if os.path.isdir(path):
        shutil.rmtree(path)