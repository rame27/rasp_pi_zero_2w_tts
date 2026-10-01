# AGENTS.md - Raspberry Pi Zero 2W TTS Project

## Project Overview
FastAPI-based text-to-speech web application running on Raspberry Pi Zero 2W (headless DietPi OS). Provides:
- Text-to-speech via edge-tts (Microsoft Edge voices)
- Bluetooth speaker management (scan, connect, volume, mute)
- Story generation with role-based narration using Groq LLM
- URL text extraction and curation (Project Gutenberg, etc.)
- Audio playback with anti-clipping tone padding

## Quick Start (on the Pi)

```bash
# SSH into the device
sshpass -p 'killbill' ssh root@192.168.1.241

# Activate virtual environment and run
source /root/piper/piper_env/bin/activate
./run.sh
# OR directly:
/root/piper/piper_env/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

The app serves on `http://<pi-ip>:8000` with a web UI at `/` and REST API at `/api/*`.

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Main FastAPI app, lifespan, all API endpoints |
| `config.py` | Pydantic Settings (env-configurable) |
| `config.ini` | Static configuration (API keys, host/port, etc.) |
| `models.py` | Pydantic request/response models |
| `run.sh` | Startup script (activates venv, runs uvicorn) |
| `requirements.txt` | Python dependencies |
| `services/audio.py` | Core TTS pipeline (synthesis, chunking, ffmpeg concat, playback) |
| `services/story_narrator.py` | Story generation, segment management, combine/playback |
| `services/bluetooth.py` | Bluetooth device scan/connect/volume/mute |
| `services/groq_llm.py` | Story analysis via Groq API |
| `services/text_source.py` | URL fetching, text curation (Gutenberg boilerplate removal) |
| `services/voices.py` | Edge-TTS voice catalog loading/filtering |
| `services/speaker.py` | BlueALSA volume/mute control |

## Configuration

Static configuration lives in `config.ini` (project root). Runtime overrides via `APP_` prefixed environment variables.

**config.ini sections:**
- `[server]` - `host`, `port`, `log_level` (INFO hides per-request lines, DEBUG shows them)
- `[groq]` - `api_key`, `model`
- `[bluetooth]` - `speaker_mac`
- `[storage]` - `story_dir`, `speak_text_dir`
- `[voices]` - `allowed_locales`, `keep_english_voices`

**Other settings** (in `config.py` with sensible defaults):
- `max_text_chars` - Text truncation limit (50000)
- `export_dir` - Temp exports (`/tmp/app_edgetts_exports`)
- `export_ttl` - Export TTL (300s / 5min)
- `export_max` - Max exports (10)
- `playback_delay_ms` - Delay before speech (500ms)
- `sentence_padding_ms` - Tone between sentences (400ms)
- `padding_tone_hz` - Tone frequency (1000Hz)
- `padding_tone_db` - Tone level (-45dB)
- `sentences_per_segment` - Sentences per segment (3)
- `bt_history_file` - Bluetooth history (`/root/app_edgetts/bt_history.json`)

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Web UI (static/index.html) |
| GET | `/api/voices` | Filtered voice catalog |
| GET | `/api/status` | Pipeline state (idle/synthesizing/playing) |
| POST | `/api/stop` | Stop current synthesis/playback |
| POST | `/api/poweroff` / `/api/restart` | System actions |
| GET | `/api/bt/status` | Bluetooth status + history |
| POST | `/api/bt/scan` | Scan for devices |
| POST | `/api/bt/connect` | Connect to MAC |
| POST | `/api/bt/disconnect` | Disconnect MAC |
| GET | `/api/speaker` | Speaker state (volume, mute, battery) |
| POST | `/api/speaker/volume` | Set volume (0-127) |
| POST | `/api/speaker/mute` | Set mute |
| POST | `/api/story/analyze` | Analyze text/URL → role/voice plan (Groq) |
| POST | `/api/story/generate` | Generate story from plan (async task) |
| GET | `/api/story/status/{task_id}` | Task progress |
| GET | `/api/stories` | List generated stories + URL-speak archives |
| POST | `/api/stories/{name}/play` | Play story (combined or segments) |
| GET | `/api/stories/{name}/combined.mp3` | Download combined MP3 |
| DELETE | `/api/stories/{name}` | Delete story |
| POST | `/api/speak` | Synthesize + play text or URL (ephemeral or archived) |
| GET | `/api/download/{speak_id}.mp3` | Download generated MP3 |

## Development Notes

### Architecture Quirks
- **Access logs at DEBUG**: uvicorn's per-request lines (`"GET /api/bt/status ..." 200 OK`) and chatty UI polls are logged at DEBUG, not INFO. Set `log_level = DEBUG` in `config.ini` `[server]` (or `APP_LOG_LEVEL=DEBUG`) to see them. Uvicorn's loggers are rewired into the app's root handlers in `app.py` (`_route_uvicorn_logging`), which runs at import — after uvicorn's own config — so both `python app.py` (uses `log_config=None`) and `python -m uvicorn` pick it up.
- **Audio warm-up**: On startup, plays `silent.wav` via `ffplay` to establish A2DP transport (prevents first-word clipping)
- **Anti-clipping tones**: Every segment/chunk prepended with low-level 1000Hz tone (`padding_tone_db=-45`) to keep Bluetooth audio path active
- **Sentence grouping**: Paragraphs are hard boundaries; within paragraphs, sentences balanced into 2-3 per segment
- **URL-speak archives**: Stored under `speak_text_dir` with metadata.json; replayable, resumable, listed in Stories tab
- **Story generation**: Pipeline synthesis + playback run concurrently (producer/consumer pattern)

### External Dependencies (must exist on Pi)
- `ffmpeg` / `ffplay` - Audio concat and playback
- `bluetoothctl` / `bluealsa` - Bluetooth management (with `--keep-alive`)
- `pactl` / `amixer` - Volume/mute via BlueALSA
- Python packages: `fastapi`, `uvicorn`, `edge-tts`, `pysbd`, `httpx`, `beautifulsoup4`, `lxml`, `pydantic`, `pydantic-settings`

### Testing / Verification
```bash
# Check service status
systemctl status bluealsa  # Bluetooth audio
systemctl status bluetooth

# Test TTS manually
source /root/piper/piper_env/bin/activate
python -c "import edge_tts; import asyncio; asyncio.run(edge_tts.Communicate('test', 'en-US-AndrewNeural').save('/tmp/test.mp3'))"

# View logs
tail -f /tmp/app_edgetts.log
```

### Common Issues
- **First word clipped**: Ensure `bluealsa --keep-alive` is running; warm-up tone in `lifespan` handles this
- **Speaker disconnects**: `speaker_mac` in config enables auto-reconnect in `_ensure_speaker_connected()`
- **Groq rate limits**: Free tier; `groq_model` defaults to `openai/gpt-oss-120b`
- **Large text**: `max_text_chars=50000` truncates input; stories split into 3-sentence chunks

## Git Workflow
- Main branch: `main`
- Two commits only: initial + cleanup
- No CI/CD configured; deploy by running `run.sh` on Pi

## Security Notes
- API keys in `config.ini` (Groq, Xiaozhi) - **do not commit real keys**
- SSH password in `rasp_zero2W.md` - change in production
- No authentication on API endpoints (local network only)
