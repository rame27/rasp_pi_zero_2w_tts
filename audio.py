from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
from typing import Callable, Literal

import edge_tts
import pysbd

from config import settings

log = logging.getLogger(__name__)

# Sentence-boundary disambiguator: handles abbreviations (Mr., Dr., St.),
# decimals, ellipses and initials instead of splitting on every period.
_segmenter = pysbd.Segmenter(language="en", clean=False)

# Coordinating conjunctions that can start a list-continuation fragment
# ("and Peter." after a comma-separated list).
_FRAGMENT_CONJUNCTIONS = {"and", "or", "but", "nor", "so", "yet"}

# Sentence-final punctuation: a fragment ending in one of these is a complete
# utterance ("Yes.", "No.", "Stop!") and must NOT be joined.
_SENTENCE_END_RE = re.compile(r"[.!?]\s*$")


async def _ensure_speaker_connected() -> bool:
    """Check the Bluetooth speaker is connected; reconnect if it slept."""
    mac = settings.speaker_mac
    try:
        proc = await asyncio.create_subprocess_exec(
            "bluetoothctl", "info", mac,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        if b"Connected: yes" in out:
            return True
        log.info("Speaker not connected, reconnecting %s", mac)
        proc = await asyncio.create_subprocess_exec(
            "bluetoothctl", "connect", mac,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        ok = b"Connection successful" in out
        log.info("Reconnect %s: %s", mac, "ok" if ok else "failed")
        return ok
    except Exception as exc:
        log.warning("Speaker check failed: %s", exc)
        return False


def _is_fragment(sentence: str) -> bool:
    """True if the sentence is a short list fragment to join into the previous one.

    pysbd splits on line breaks, so Gutenberg list formatting like
    "were--\\n\\n Flopsy,\\n Mopsy,\\nand Peter." becomes separate one-word
    "sentences". A fragment is <= 2 words that either lacks sentence-final
    punctuation ("Flopsy,") or starts with a coordinating conjunction
    ("and Peter."). Complete utterances ("Yes.", "No.", "Stop!") are kept.
    """
    words = sentence.split()
    if len(words) > 2:
        return False
    if _SENTENCE_END_RE.search(sentence):
        return words[0].lower() in _FRAGMENT_CONJUNCTIONS
    return True


def _join_fragments(sentences: list[str]) -> list[str]:
    """Merge short list fragments into the preceding sentence.

    Example: ["...their names", "were--", "Flopsy,", "and Peter."]
    -> ["...their names were-- Flopsy, and Peter."]
    """
    merged: list[str] = []
    for sentence in sentences:
        if merged and _is_fragment(sentence):
            merged[-1] = f"{merged[-1]} {sentence}"
        else:
            merged.append(sentence)
    return merged


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, respecting abbreviations (Mr., Dr., ...).

    Short list fragments produced by line breaks ("Flopsy," / "and Peter.")
    are joined into the preceding sentence so the narrator doesn't pause
    between every list item.
    """
    sentences = [s.strip() for s in _segmenter.segment(text) if s.strip()]
    return _join_fragments(sentences)


def _chunk_sizes(n: int, max_size: int) -> list[int]:
    """Split n items into balanced chunks of 2..max_size (1 if n == 1).

    Uses the fewest chunks with each <= max_size, then distributes evenly so
    no chunk is left with a single leftover sentence: 4 -> [2, 2], 5 -> [3, 2],
    6 -> [3, 3], 7 -> [3, 2, 2], 8 -> [3, 3, 2].
    """
    if n <= max_size:
        return [n]
    k = (n + max_size - 1) // max_size  # fewest chunks with each <= max_size
    base = n // k
    rem = n % k
    return [base + 1] * rem + [base] * (k - rem)


def _group_sentences(text: str, max_per_group: int | None = None) -> list[str]:
    """Group sentences into balanced segments of 2-3, within paragraphs.

    Paragraphs (blank-line separated) are hard boundaries: sentences are never
    grouped across them. Within a paragraph, sentences are distributed into
    balanced chunks of 2-3 (4 sentences -> 2+2, 5 -> 3+2) so the narrator reads
    a natural run of text without single-sentence leftovers. A lone sentence
    stays alone. Defaults to settings.sentences_per_segment.

    Word-wrap removal is done upstream in preprocessing (curate_text), so the
    text arriving here already has wrapped lines joined into paragraphs.
    """
    if max_per_group is None:
        max_per_group = settings.sentences_per_segment
    groups: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        sentences = _split_sentences(paragraph)
        i = 0
        for size in _chunk_sizes(len(sentences), max_per_group):
            groups.append(" ".join(sentences[i : i + size]))
            i += size
    return groups


async def _synthesize_sentences(
    sentences: list[str],
    voice: str,
    output_path: str,
    progress_cb: Callable[[int], None] | None = None,
) -> None:
    """Synthesize each text chunk and concatenate with a faint tone between them.

    The Bluetooth speaker clips the first word of each chunk after a silence
    gap (its buffer drains / it pauses during silence). A low-level tone keeps
    the audio path continuously active so no chunk start is clipped.

    progress_cb is invoked with the count of completed chunks after each one,
    so callers can report live progress during long syntheses.
    """
    initial_ms = settings.playback_delay_ms
    padding_ms = settings.sentence_padding_ms
    tone_freq = settings.padding_tone_hz
    tone_db = settings.padding_tone_db

    part_paths: list[str] = []
    try:
        for i, sentence in enumerate(sentences):
            part_path = f"{output_path}.part{i}.mp3"
            part_paths.append(part_path)  # register before await so cancel cleans it
            communicate = edge_tts.Communicate(sentence, voice)
            await communicate.save(part_path)
            if progress_cb:
                progress_cb(i + 1)

        cmd = ["ffmpeg", "-y"]
        for p in part_paths:
            cmd += ["-i", p]

        filters = [
            f"[{i}:a]aresample=24000,aformat=sample_fmts=s16:channel_layouts=mono[a{i}]"
            for i in range(len(part_paths))
        ]

        def tone_src(dur_ms: int, label: str) -> str:
            # ffmpeg's sine source is at ~-18dBFS, so attenuate to reach the
            # configured target level (tone_db is the desired final level).
            atten = tone_db + 18
            return (
                f"sine=frequency={tone_freq}:duration={dur_ms / 1000.0}"
                f":sample_rate=24000,volume={atten}dB,"
                f"aformat=sample_fmts=s16:channel_layouts=mono[{label}]"
            )

        concat_inputs: list[str] = []
        n = 0
        # Initial tone: wakes the speaker / fills its buffer before speech.
        filters.append(tone_src(initial_ms, "t0"))
        concat_inputs.append("[t0]")
        n += 1
        for i in range(len(part_paths)):
            concat_inputs.append(f"[a{i}]")
            n += 1
            if i < len(part_paths) - 1:
                filters.append(tone_src(padding_ms, f"t{i + 1}"))
                concat_inputs.append(f"[t{i + 1}]")
                n += 1
        filters.append(f"{''.join(concat_inputs)}concat=n={n}:v=0:a=1[out]")

        cmd += [
            "-filter_complex", ";".join(filters),
            "-map", "[out]",
            "-c:a", "libmp3lame", "-q:a", "4",
            output_path,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {stderr.decode(errors='replace')}")
    finally:
        for p in part_paths:
            try:
                os.remove(p)
            except OSError:
                pass


async def _synth_segment_with_tone(text: str, voice: str, out_path: str, tone_ms: int) -> None:
    """Synthesize one sentence and prepend a low-level tone (anti-clipping).

    Each segment file is [tone][sentence] so sequential playback and the later
    ffmpeg concat keep the audio path continuously active between sentences.
    """
    tmp_path = f"{out_path}.raw.mp3"
    comm = edge_tts.Communicate(text, voice)
    try:
        await comm.save(tmp_path)
    except asyncio.CancelledError:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    tone_freq = settings.padding_tone_hz
    tone_db = settings.padding_tone_db
    atten = tone_db + 18
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i",
        f"sine=frequency={tone_freq}:duration={tone_ms / 1000.0}:sample_rate=24000,"
        f"volume={atten}dB,aformat=sample_fmts=s16:channel_layouts=mono",
        "-i", tmp_path,
        "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]",
        "-map", "[out]", "-c:a", "libmp3lame", "-q:a", "4",
        out_path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await proc.communicate()
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg tone prepend failed: {stderr.decode(errors='replace')}")
    try:
        os.remove(tmp_path)
    except OSError:
        pass


def _write_speak_metadata(
    archive_dir: str, speak_id: str, text: str, voice: str, segment_count: int | None = None
) -> None:
    """Write metadata.json for a URL-speak archive (story-compatible layout).

    segment_count is omitted when None so list_stories falls back to counting
    the actual segment files (used for the initial metadata written at archive
    creation, before synthesis has produced any segments).

    When the archive already has metadata (e.g. a resumed or replayed URL
    speak), the original created timestamp is preserved so the story keeps its
    first-seen position in the Stories tab.
    """
    created = time.time()
    meta_path = os.path.join(archive_dir, "metadata.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                created = json.load(f).get("created", created)
        except (OSError, json.JSONDecodeError):
            pass
    metadata = {
        "story_name": speak_id,
        "source": text,
        "created": created,
        "roles": [{"role": "Narrator", "voice": voice}],
        "segments": [{"role": "Narrator", "start_word": 0, "end_word": len(text.split()), "voice": voice}],
    }
    if segment_count is not None:
        metadata["segment_count"] = segment_count
    with open(os.path.join(archive_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# Segment files are 001.mp3 (URL speaks) or 001_Narrator_01.mp3 (generated
# stories). Temp files like 004.mp3.raw.mp3 (interrupted tone-prepend) must not
# be counted as segments.
_SEGMENT_FILE_RE = re.compile(r"^\d{3}(_.+)?\.mp3$")


def is_segment_file(fn: str) -> bool:
    """True for segment files (001.mp3 / 001_Narrator_01.mp3), excluding temp files."""
    return bool(_SEGMENT_FILE_RE.match(fn))


def archive_speak(output_path: str, speak_id: str, text: str, voice: str, archive_dir: str) -> None:
    """Copy a synthesized MP3 into the speak archive and write story metadata."""
    os.makedirs(os.path.join(archive_dir, "segments"), exist_ok=True)
    shutil.copy(output_path, os.path.join(archive_dir, "segments", "001_Narrator.mp3"))
    metadata = {
        "story_name": speak_id,
        "source": text,
        "created": time.time(),
        "roles": [{"role": "Narrator", "voice": voice}],
        "segments": [{"role": "Narrator", "start_word": 0, "end_word": len(text.split()), "voice": voice}],
    }
    with open(os.path.join(archive_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


class NarratorPipeline:
    """Synthesize text with edge-tts and play the MP3 on the Pi's audio output."""

    def __init__(self, export_dir: str = settings.export_dir):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
        self.state: Literal["idle", "synthesizing", "playing"] = "idle"
        self.speak_id: str | None = None
        self.segments_total = 0
        self.segments_done = 0
        self.start_time = 0.0
        self._task: asyncio.Task | None = None
        self._stop_requested = False
        self._play_proc: asyncio.subprocess.Process | None = None

    def get_status(self) -> dict:
        elapsed = int((time.time() - self.start_time) * 1000) if self.start_time > 0 else 0
        return {
            "state": self.state,
            "speak_id": self.speak_id,
            "segments_total": self.segments_total,
            "segments_done": self.segments_done,
            "elapsed_ms": elapsed,
        }

    def begin(self) -> bool:
        if self.state != "idle":
            return False
        self._stop_requested = False
        return True

    def start(self, text: str, voice: str, speak_id: str, archive_dir: str | None = None) -> None:
        self.speak_id = speak_id
        self.segments_total = len(_group_sentences(text))
        self.segments_done = 0
        self.start_time = time.time()
        self.state = "synthesizing"
        self._task = asyncio.create_task(self._run(text, voice, speak_id, archive_dir))

    def start_combined(self, combined_path: str, speak_id: str) -> None:
        """Play an already-combined MP3 (a URL speak that was fully processed).

        No synthesis happens: the existing combined.mp3 is played directly.
        Used when the same URL is spoken again after a previous run completed.
        """
        self.speak_id = speak_id
        self.segments_total = 1
        self.segments_done = 0
        self.start_time = time.time()
        self.state = "playing"
        self._task = asyncio.create_task(self._run_combined(combined_path, speak_id))

    async def _run(self, text: str, voice: str, speak_id: str, archive_dir: str | None = None) -> None:
        output_path = os.path.join(self.export_dir, f"{speak_id}.mp3")
        try:
            # 1) Synthesize per segment (balanced 2-3 sentences, within paragraphs)
            sentences = _group_sentences(text)
            self.state = "synthesizing"
            if archive_dir:
                await self._run_archived(text, sentences, voice, speak_id, archive_dir)
            else:
                await self._run_plain(sentences, voice, speak_id, output_path)
        except asyncio.CancelledError:
            log.info("Playback cancelled for %s", speak_id)
            raise
        except Exception as exc:
            log.error("Pipeline error for %s: %s", speak_id, exc)
        finally:
            # Text speaks are ephemeral: remove the generated MP3 once done or
            # stopped (URL speaks keep their archive under speak_text_dir).
            if archive_dir is None and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    log.info("Removed ephemeral export %s", output_path)
                except OSError:
                    pass
            self.set_idle()

    async def _run_archived(
        self, text: str, sentences: list[str], voice: str, speak_id: str, archive_dir: str
    ) -> None:
        """URL speak: synthesize each segment into the archive's segments dir.

        Synthesis and playback run concurrently: each segment is played as soon
        as it is ready, while the remaining segments are still being
        synthesized in the background. The individual MP3s live in
        speak_texts/{id}/segments/ so the Stories tab can replay them; a replay
        combines them into combined.mp3 and deletes the individual files.

        If the archive already contains segment files (an earlier run was
        interrupted), synthesis resumes from the first missing segment and the
        existing ones are played as-is.
        """
        seg_dir = os.path.join(archive_dir, "segments")
        os.makedirs(seg_dir, exist_ok=True)
        existing = sorted(f for f in os.listdir(seg_dir) if is_segment_file(f))
        if existing:
            log.info("Resuming %s from segment %d", speak_id, len(existing) + 1)
        self.segments_done = len(existing)
        queue: asyncio.Queue[str] = asyncio.Queue()
        synth_done = asyncio.Event()

        async def producer() -> None:
            try:
                for i, sentence in enumerate(sentences, start=1):
                    if self._stop_requested:
                        break
                    seg_file = os.path.join(seg_dir, f"{i:03d}.mp3")
                    if os.path.exists(seg_file):
                        # Already synthesized during an earlier run: play as-is.
                        await queue.put(seg_file)
                        self.segments_done = i
                        continue
                    tone_ms = settings.playback_delay_ms if i == 1 else settings.sentence_padding_ms
                    await _synth_segment_with_tone(sentence, voice, seg_file, tone_ms)
                    await queue.put(seg_file)
                    self.segments_done = i
                if not self._stop_requested:
                    _write_speak_metadata(archive_dir, speak_id, text, voice, len(sentences))
                    log.info("Synthesized %s (%d segments)", speak_id, len(sentences))
            finally:
                synth_done.set()

        async def consumer() -> None:
            await _ensure_speaker_connected()
            while not self._stop_requested:
                try:
                    seg_file = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if synth_done.is_set():
                        break
                    continue
                self.state = "playing"
                await self._play_file(seg_file)
            log.info("Playback finished for %s", speak_id)

        producer_task = asyncio.create_task(producer())
        consumer_task = asyncio.create_task(consumer())
        try:
            await asyncio.gather(producer_task, consumer_task)
        finally:
            for t in (producer_task, consumer_task):
                if not t.done():
                    t.cancel()
            await asyncio.gather(producer_task, consumer_task, return_exceptions=True)

    async def _run_combined(self, combined_path: str, speak_id: str) -> None:
        """Play an existing combined.mp3 (fully processed URL speak)."""
        try:
            await _ensure_speaker_connected()
            self.state = "playing"
            await self._play_file(combined_path)
            log.info("Playback finished for %s (combined)", speak_id)
        except asyncio.CancelledError:
            log.info("Playback cancelled for %s (combined)", speak_id)
            raise
        except Exception as exc:
            log.error("Pipeline error for %s (combined): %s", speak_id, exc)
        finally:
            self.set_idle()

    async def _run_plain(self, sentences: list[str], voice: str, speak_id: str, output_path: str) -> None:
        """Plain-text speak: synthesize into a single combined MP3, then play it."""
        def _on_progress(done: int) -> None:
            self.segments_done = done

        await _synthesize_sentences(sentences, voice, output_path, progress_cb=_on_progress)
        self.segments_done = len(sentences)
        log.info(
            "Synthesized %s (%d bytes, %d sentences)",
            speak_id, os.path.getsize(output_path), len(sentences),
        )
        if self._stop_requested:
            self.set_idle()
            return

        # 2) Play through the Pi's audio output.
        #    Ensure the Bluetooth speaker is awake/connected first.
        await _ensure_speaker_connected()
        #    The wake-up tone is baked into the audio file already.
        self.state = "playing"
        await self._play_file(output_path)
        log.info("Playback finished for %s", speak_id)

    async def _play_file(self, path: str) -> None:
        """Play one MP3 via ffplay, tracking the process so stop() can kill it."""
        proc = await asyncio.create_subprocess_exec(
            "ffplay", "-nodisp", "-autoexit", "-loglevel", "error", path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._play_proc = proc
        await proc.wait()
        self._play_proc = None

    def stop(self) -> None:
        self._stop_requested = True
        if self._play_proc is not None and self._play_proc.returncode is None:
            try:
                self._play_proc.kill()
            except ProcessLookupError:
                pass
        self._play_proc = None
        if self._task and not self._task.done():
            self._task.cancel()
        self.set_idle()

    def set_idle(self) -> None:
        self.state = "idle"
        self.speak_id = None
        self.segments_total = 0
        self.segments_done = 0
        self.start_time = 0.0
        self._task = None
        self._stop_requested = False
