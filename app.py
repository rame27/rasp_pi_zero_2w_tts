from __future__ import annotations

import asyncio
import logging
import os
import secrets
import subprocess
import time
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from models import (
    BTConnectRequest,
    BTConnectResponse,
    BTDevice,
    BTScanResponse,
    BTStatusResponse,
    ErrorResponse,
    SpeakRequest,
    SpeakResponse,
    SpeakerMuteRequest,
    SpeakerState,
    SpeakerVolumeRequest,
    StatusResponse,
    StoryAnalyzeRequest,
    StoryGenerateRequest,
    StoryListResponse,
    StoryPlan,
    StoryTaskStatus,
    VoiceCatalog,
    VoiceInfo,
)
from services import bluetooth, groq_llm, speaker, story_narrator, text_source, voices
from services.audio import NarratorPipeline, _ensure_speaker_connected, is_segment_file

def _resolve_level(name: str) -> int:
    """Map a level name (DEBUG, INFO, ...) to its int; fall back to INFO."""
    level = getattr(logging, str(name).upper(), None)
    return level if isinstance(level, int) else logging.INFO


_LEVEL = _resolve_level(settings.log_level)

logging.basicConfig(
    level=_LEVEL,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler("/tmp/app_edgetts.log", maxBytes=1_000_000, backupCount=2),
    ],
)
# Handler levels gate records that are downgraded after creation (uvicorn
# access): at INFO they drop DEBUG records, at DEBUG they pass them.
for _handler in logging.getLogger().handlers:
    _handler.setLevel(_LEVEL)


class _UvicornAccessToDebug(logging.Filter):
    """Emit uvicorn's per-request access records at DEBUG instead of INFO.

    Noisy lines like '192.168.1.147:63981 - "GET /api/bt/status ..." 200'
    then only appear when log_level=DEBUG; app-level request logging
    (log_requests middleware) still applies its own DEBUG-for-chatty rule.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "uvicorn.access" and record.levelno <= logging.INFO:
            record.levelno = logging.DEBUG
            record.levelname = "DEBUG"
        return True


def _route_uvicorn_logging() -> None:
    """Send uvicorn's loggers through the app's root handlers.

    Uvicorn installs its own handlers/levels via dictConfig; drop them so
    verbosity is governed by a single place (settings.log_level) and access
    records reach both the console and /tmp/app_edgetts.log.
    """
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _UvicornAccessToDebug) for f in access.filters):
        access.addFilter(_UvicornAccessToDebug())
    for name in ("uvicorn.access", "uvicorn.error", "uvicorn"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
        lg.setLevel(_LEVEL)


_route_uvicorn_logging()
log = logging.getLogger(__name__)

_pipeline: NarratorPipeline | None = None


def _get_pipeline() -> NarratorPipeline:
    assert _pipeline is not None, "Pipeline not initialized"
    return _pipeline


async def _warmup_audio() -> None:
    """Play a silent WAV to establish the Bluetooth A2DP transport.

    bluealsa --keep-alive keeps the transport open afterwards, so the first
    speak request starts instantly instead of re-negotiating the stream
    (which used to drop the first ~1s of audio = first words).
    """
    silent_path = os.path.join(os.path.dirname(__file__), "silent.wav")
    if not os.path.exists(silent_path):
        log.warning("Warm-up skipped: silent.wav not found")
        return
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "error",
            silent_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        log.info("Audio warm-up complete (A2DP transport established)")
    except Exception as exc:
        log.warning("Audio warm-up failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    os.makedirs(settings.export_dir, exist_ok=True)
    bluetooth.load_default_speaker()
    _pipeline = NarratorPipeline(export_dir=settings.export_dir)
    asyncio.create_task(_warmup_audio())
    log.info("app_edgetts ready on http://%s:%d", settings.host, settings.port)
    yield
    log.info("Shutting down")


app = FastAPI(
    title="Edge-TTS Web UI",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every HTTP request to the app log file.

    Chatty UI poll endpoints are logged at DEBUG so they stay out of the
    INFO-level log file.
    """
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    client = request.client.host if request.client else "-"
    chatty = (
        request.method == "GET"
        and (
            request.url.path in ("/api/speaker", "/api/bt/status", "/api/status")
            or request.url.path.startswith("/api/story/status")
        )
    )
    log.log(
        logging.DEBUG if chatty else logging.INFO,
        "%s %s -> %d (%.0f ms) from %s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        client,
    )
    return response

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(
                content=f.read(),
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )
    except FileNotFoundError:
        raise HTTPException(500, "index.html not found")


@app.get("/api/voices", response_model=VoiceCatalog)
async def get_voices():
    voice_list = await voices.load_all_voices()
    return VoiceCatalog(voices=[VoiceInfo(**v) for v in voice_list])


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    return StatusResponse(**_get_pipeline().get_status())


@app.post("/api/stop")
async def stop():
    pipe = _get_pipeline()
    pipe.stop()
    story_narrator.cancel_all()
    log.info("Stop requested")
    return {"ok": True, "stopped": True}


@app.post("/api/poweroff")
async def poweroff():
    asyncio.create_task(_system_action("poweroff"))
    return {"ok": True, "action": "poweroff"}


@app.post("/api/restart")
async def restart():
    asyncio.create_task(_system_action("reboot"))
    return {"ok": True, "action": "restart"}


@app.get("/api/bt/status", response_model=BTStatusResponse)
def bt_status():
    return BTStatusResponse(**bluetooth.get_status())


@app.post("/api/bt/scan", response_model=BTScanResponse)
def bt_scan():
    devices = [BTDevice(**d) for d in bluetooth.scan()]
    return BTScanResponse(devices=devices)


@app.post("/api/bt/connect", response_model=BTConnectResponse)
def bt_connect(req: BTConnectRequest):
    if not bluetooth.is_valid_mac(req.mac):
        raise HTTPException(400, f"invalid bluetooth address: {req.mac!r}")
    info = bluetooth.connect(req.mac)
    return BTConnectResponse(
        ok=True,
        connected=info["connected"],
        device=BTDevice(**info),
    )


@app.post("/api/bt/disconnect")
def bt_disconnect(req: BTConnectRequest):
    bluetooth.disconnect(req.mac)
    log.info("BT disconnect requested: %s", req.mac)
    return {"ok": True, "disconnected": True}


def _current_speaker_mac() -> str:
    mac = settings.speaker_mac
    if not mac:
        raise HTTPException(404, "no speaker configured")
    return mac


def _speaker_name(mac: str) -> str:
    for d in bluetooth.list_devices():
        if d["mac"].upper() == mac.upper():
            return d["name"]
    return mac


@app.get("/api/speaker", response_model=SpeakerState)
def speaker_state():
    mac = _current_speaker_mac()
    state = speaker.get_speaker_state(mac)
    return SpeakerState(name=_speaker_name(mac), **state)


@app.post("/api/speaker/volume", response_model=SpeakerVolumeRequest)
def speaker_volume(req: SpeakerVolumeRequest):
    mac = _current_speaker_mac()
    speaker.set_volume(mac, req.volume)
    bluetooth.save_volume(mac, req.volume)
    log.info("Speaker volume set: %s -> %d", mac, req.volume)
    return req


@app.post("/api/speaker/mute")
def speaker_mute(req: SpeakerMuteRequest):
    mac = _current_speaker_mac()
    speaker.set_mute(mac, req.muted)
    log.info("Speaker mute set: %s -> %s", mac, req.muted)
    return {"ok": True, "muted": req.muted}


@app.post("/api/story/analyze", response_model=StoryPlan)
async def story_analyze(req: StoryAnalyzeRequest):
    if req.url:
        # Already-processed URL: reuse the stored plan instead of re-analyzing.
        base = story_narrator.speak_name_from_url(str(req.url))
        if base:
            existing = story_narrator.stored_plan(base)
            if existing:
                log.info("URL %s already processed; reusing stored plan", base)
                return existing
    if req.text:
        text, title = req.text, None
    else:
        try:
            # Curated text (front matter removed) so the LLM's 20K-char budget
            # isn't consumed by Gutenberg boilerplate / preface padding.
            text, title = await asyncio.to_thread(text_source.fetch_text, str(req.url), True)
        except Exception as exc:
            raise HTTPException(502, f"Failed to fetch URL: {exc}")
    catalog = await voices.load_all_voices()
    voice_names = [v["ShortName"] for v in catalog]
    try:
        return await asyncio.to_thread(groq_llm.analyze_story, text, title, voice_names)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))


@app.post("/api/story/generate")
async def story_generate(req: StoryGenerateRequest):
    task_id = story_narrator.create_task()
    asyncio.create_task(story_narrator.generate_story(
        str(req.url) if req.url else None, req.text, req.story_name, req.plan, task_id,
        pipeline=_get_pipeline(),
    ))
    return {"task_id": task_id}


@app.get("/api/story/status/{task_id}", response_model=StoryTaskStatus)
def story_status(task_id: str):
    task = story_narrator.get_task(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    return StoryTaskStatus(task_id=task_id, **task)


@app.get("/api/stories", response_model=StoryListResponse)
def stories_list():
    return StoryListResponse(stories=story_narrator.list_stories())


@app.post("/api/stories/{name}/play")
async def story_play(name: str):
    if story_narrator.generation_running():
        raise HTTPException(409, "Story generation in progress")
    playback_list: str | None = None
    try:
        combined = await asyncio.to_thread(story_narrator.combine_story, name)
        args = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", combined]
    except story_narrator.StoryIncomplete:
        # Generation hasn't produced every segment yet (interrupted/crashed
        # run): play the segments that do exist instead of combining a
        # partial set into combined.mp3 and losing the rest.
        try:
            playback_list = await asyncio.to_thread(story_narrator.segments_playback_list, name)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        args = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", playback_list]
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, "invalid story name")
    await _ensure_speaker_connected()
    story_narrator.stop_playback()  # stop any current playback first
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    story_narrator.set_play_proc(proc)
    asyncio.create_task(_reap_story_play(proc, playback_list))
    return {"ok": True, "playing": name, "partial": playback_list is not None}


async def _reap_story_play(proc, playback_list: str | None = None):
    _, stderr = await proc.communicate()
    if proc.returncode not in (0, -9):
        log.warning("ffplay exited %d: %s", proc.returncode, stderr.decode(errors="replace"))
    story_narrator.clear_play_proc(proc)
    if playback_list is not None:
        try:
            os.remove(playback_list)
        except OSError:
            pass


@app.get("/api/stories/{name}/combined.mp3")
def story_download(name: str):
    try:
        combined = story_narrator.combine_story(name)
    except story_narrator.StoryIncomplete as exc:
        raise HTTPException(409, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, "invalid story name")
    return FileResponse(combined, media_type="audio/mpeg", filename=f"{name}.mp3")


@app.delete("/api/stories/{name}")
def story_delete(name: str):
    story_narrator.stop_playback()
    try:
        story_narrator.delete_story(name)
    except ValueError as exc:
        raise HTTPException(400, "invalid story name")
    return {"ok": True, "deleted": name}


@app.post("/api/speak", response_model=SpeakResponse, responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
async def speak(req: SpeakRequest):
    pipe = _get_pipeline()
    if req.text and req.url:
        raise HTTPException(400, "provide only one of 'text' or 'url'")
    if not pipe.begin():
        raise HTTPException(409, "already speaking - stop first")
    try:
        speak_id = secrets.token_hex(8)
        archive_dir = None
        if req.url:
            # URL speaks are archived under speak_text_dir and listed in the stories tab.
            # Name the archive after the URL basename (e.g. 14838-0) and reuse it:
            # a fully processed URL replays combined.mp3, a partial one resumes
            # from where synthesis was interrupted. Both reuse the stored text,
            # so no URL fetch is needed. Stop never cleans these up; deletion
            # happens via the Stories tab.
            base = story_narrator.speak_name_from_url(req.url)
            if base:
                speak_id = base
            archive_dir = os.path.join(settings.speak_text_dir, speak_id)
            combined = os.path.join(archive_dir, "combined.mp3")
            if os.path.isfile(combined):
                pipe.start_combined(combined, speak_id)
                log.info("URL %s already processed; replaying combined.mp3", speak_id)
                return SpeakResponse(
                    speak_id=speak_id,
                    text_length=len(story_narrator.stored_text(speak_id)),
                    num_segments=1,
                    voice=story_narrator.stored_voice(speak_id) or req.voice,
                )
            seg_dir = os.path.join(archive_dir, "segments")
            existing = (
                sorted(f for f in os.listdir(seg_dir) if is_segment_file(f))
                if os.path.isdir(seg_dir)
                else []
            )
            if existing:
                # Resume: reuse the stored text and voice so segment numbering
                # stays consistent with the segments already on disk.
                text = story_narrator.stored_text(speak_id)
                if not text:
                    # Corrupt archive (no source.txt): fall back to fetching.
                    text = _extract_text(req)
                voice = story_narrator.stored_voice(speak_id) or req.voice
                archive_dir = story_narrator.save_speak_archive(speak_id, text, voice)
                pipe.start(text, voice, speak_id, archive_dir)
                log.info("Resuming %s from segment %d", speak_id, len(existing) + 1)
                return SpeakResponse(
                    speak_id=speak_id,
                    text_length=len(text),
                    num_segments=1,
                    voice=voice,
                )
        # Fresh speak: fetch/curate the text (URL not processed yet, or text input).
        text = _extract_text(req)
        if not text.strip():
            raise HTTPException(400, "no text to speak")
        if req.url:
            archive_dir = story_narrator.save_speak_archive(speak_id, text, req.voice)
        pipe.start(text, req.voice, speak_id, archive_dir)
        log.info("Speak requested: %d chars, voice=%s, id=%s", len(text), req.voice, speak_id)
    except Exception:
        pipe.set_idle()
        raise
    _purge_exports()
    return SpeakResponse(
        speak_id=speak_id,
        text_length=len(text),
        num_segments=1,
        voice=req.voice,
    )


@app.get("/api/download/{speak_id}.mp3")
async def download(speak_id: str):
    if not _is_valid_speak_id(speak_id):
        raise HTTPException(404, "file not found")
    path = _download_path(speak_id)
    # The file may appear a moment later while synthesis is still running:
    # poll briefly instead of returning an immediate 404.
    for _ in range(20):
        if path:
            break
        status = _get_pipeline().get_status()
        if not (status["state"] == "synthesizing" and status["speak_id"] == speak_id):
            break
        await asyncio.sleep(0.5)
        path = _download_path(speak_id)
    if not path:
        raise HTTPException(404, "file not found")
    return FileResponse(path, media_type="audio/mpeg", filename=f"{speak_id}.mp3")


def _download_path(speak_id: str) -> str | None:
    """Existing MP3 for a speak: export, combined archive, or first segment."""
    path = os.path.join(settings.export_dir, f"{speak_id}.mp3")
    if os.path.exists(path):
        return path
    # URL-speak archives keep segments under speak_text_dir; serve the
    # combined file if a replay already produced it, else the first segment.
    archive = os.path.join(settings.speak_text_dir, speak_id)
    combined = os.path.join(archive, "combined.mp3")
    if os.path.exists(combined):
        return combined
    seg_dir = os.path.join(archive, "segments")
    segs = sorted(f for f in os.listdir(seg_dir) if is_segment_file(f)) if os.path.isdir(seg_dir) else []
    return os.path.join(seg_dir, segs[0]) if segs else None


def _extract_text(req: SpeakRequest) -> str:
    if req.text and req.url:
        raise HTTPException(400, "provide only one of 'text' or 'url'")
    if not req.text and not req.url:
        raise HTTPException(400, "provide 'text' or 'url'")
    if req.url:
        text, _title = text_source.fetch_text(str(req.url))
    else:
        text = text_source.curate_text(req.text or "")
    if len(text) > settings.max_text_chars:
        text = text[:settings.max_text_chars]
    return text


def _is_valid_speak_id(speak_id: str) -> bool:
    import re

    # Hex ids (text speaks) or URL-basename-derived names (URL speaks).
    return bool(re.fullmatch(r"[0-9a-f]{16}|[A-Za-z0-9][A-Za-z0-9._-]*", speak_id))


async def _system_action(action: str) -> None:
    """Run a system action after a short delay so the HTTP response flushes first."""
    await asyncio.sleep(2.0)
    try:
        subprocess.Popen(["systemctl", action])
        log.info("System action '%s' scheduled", action)
    except Exception as exc:
        log.error("System action '%s' failed: %s", action, exc)


_purged: set[str] = set()


def _purge_exports():
    try:
        files = [
            os.path.join(settings.export_dir, f)
            for f in os.listdir(settings.export_dir)
            if f.endswith(".mp3")
        ]
    except OSError:
        return
    now = time.time()
    keep = {p for p in files if now - os.path.getmtime(p) <= settings.export_ttl}
    if len(keep) > settings.export_max:
        keep = set(sorted(keep, key=os.path.getmtime)[-settings.export_max:])
    for fn in os.listdir(settings.export_dir):
        if not fn.endswith(".mp3"):
            continue
        path = os.path.join(settings.export_dir, fn)
        if path not in keep:
            try:
                os.remove(path)
                _purged.add(fn[:-4])
            except OSError:
                pass


if __name__ == "__main__":
    # log_config=None: keep the app's logging wiring (_route_uvicorn_logging)
    # instead of uvicorn reinstalling its own handlers/levels.
    uvicorn.run(app, host=settings.host, port=settings.port, log_config=None)
