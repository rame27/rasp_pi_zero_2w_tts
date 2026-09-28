import asyncio
import json
import os

from services.audio import _synthesize_sentences, _synth_segment_with_tone, _write_speak_metadata, archive_speak


def test_archive_speak_copies_mp3_and_writes_metadata(tmp_path):
    src = tmp_path / "src.mp3"
    src.write_bytes(b"ID3fake")
    archive_dir = tmp_path / "archive"
    archive_speak(str(src), "abc123", "Hello world.", "en-US-AndrewNeural", str(archive_dir))
    assert (archive_dir / "segments" / "001_Narrator.mp3").read_bytes() == b"ID3fake"
    meta = json.loads((archive_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["story_name"] == "abc123"
    assert meta["source"] == "Hello world."
    assert meta["roles"] == [{"role": "Narrator", "voice": "en-US-AndrewNeural"}]
    assert meta["segments"][0]["role"] == "Narrator"
    assert meta["segments"][0]["end_word"] == 2


def test_synthesize_sentences_reports_progress(monkeypatch, tmp_path):
    """segments_done must advance per sentence so the UI shows live progress."""

    class FakeCommunicate:
        def __init__(self, sentence, voice):
            pass

        async def save(self, path):
            with open(path, "wb") as f:
                f.write(b"fake")

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def fake_exec(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr("services.audio.edge_tts.Communicate", FakeCommunicate)
    monkeypatch.setattr("services.audio.asyncio.create_subprocess_exec", fake_exec)

    progress = []
    out = str(tmp_path / "out.mp3")
    asyncio.run(
        _synthesize_sentences(
            ["One.", "Two.", "Three."], "v", out, progress_cb=lambda n: progress.append(n)
        )
    )
    assert progress == [1, 2, 3]

def test_synth_segment_with_tone_prepends_tone(monkeypatch, tmp_path):
    """Each segment is [tone][sentence] and the raw temp file is cleaned up."""

    class FakeCommunicate:
        def __init__(self, sentence, voice):
            pass

        async def save(self, path):
            with open(path, "wb") as f:
                f.write(b"raw")

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def fake_exec(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr("services.audio.edge_tts.Communicate", FakeCommunicate)
    monkeypatch.setattr("services.audio.asyncio.create_subprocess_exec", fake_exec)

    out = str(tmp_path / "seg.mp3")
    asyncio.run(_synth_segment_with_tone("Hello.", "v", out, 400))
    assert os.path.exists(out)
    assert not os.path.exists(out + ".raw.mp3")


def test_write_speak_metadata(tmp_path):
    _write_speak_metadata(str(tmp_path), "abc123", "Hello world.", "en-US-AndrewNeural", 2)
    meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert meta["story_name"] == "abc123"
    assert meta["segment_count"] == 2
    assert meta["roles"] == [{"role": "Narrator", "voice": "en-US-AndrewNeural"}]
    assert meta["segments"][0]["end_word"] == 2
