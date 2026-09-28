from __future__ import annotations

import logging
import re
import subprocess

log = logging.getLogger(__name__)


def _pcm_path(mac: str) -> str:
    """Return the BlueALSA A2DP sink PCM path for a device MAC."""
    dev = mac.replace(":", "_")
    return f"/org/bluealsa/hci0/dev_{dev}/a2dpsrc/sink"


def _bluez_path(mac: str) -> str:
    """Return the BlueZ D-Bus object path for a device MAC."""
    dev = mac.replace(":", "_")
    return f"/org/bluez/hci0/dev_{dev}"


def _run(cmd: list[str], timeout: int = 15) -> str:
    """Run a command synchronously and return stdout."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.stdout or ""
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.warning("Command %s failed: %s", cmd[0], exc)
        return ""


def get_volume(mac: str) -> int | None:
    """Return the speaker volume (0-127) or None if unavailable."""
    out = _run(["bluealsa-cli", "volume", _pcm_path(mac)])
    match = re.search(r"Volume:\s*(\d+)", out)
    return int(match.group(1)) if match else None


def set_volume(mac: str, volume: int) -> None:
    """Set the speaker volume, clamped to 0-127."""
    volume = max(0, min(127, int(volume)))
    _run(["bluealsa-cli", "volume", _pcm_path(mac), str(volume)])


def get_mute(mac: str) -> bool | None:
    """Return True if muted, False if unmuted, None if unavailable."""
    out = _run(["bluealsa-cli", "mute", _pcm_path(mac)])
    match = re.search(r"Muted:\s*(true|false)", out)
    return match.group(1) == "true" if match else None


def set_mute(mac: str, muted: bool) -> None:
    """Set the speaker mute switch."""
    _run(["bluealsa-cli", "mute", _pcm_path(mac), "on" if muted else "off"])


def get_battery(mac: str) -> int | None:
    """Return the speaker battery percentage or None if not reported."""
    out = _run(
        [
            "busctl",
            "--system",
            "get-property",
            "org.bluez",
            _bluez_path(mac),
            "org.bluez.Battery1",
            "Percentage",
        ]
    )
    match = re.search(r"(\d+)\s*$", out.strip())
    return int(match.group(1)) if match else None


def get_speaker_state(mac: str) -> dict:
    """Return combined speaker state for the UI."""
    return {
        "mac": mac,
        "volume": get_volume(mac),
        "muted": get_mute(mac),
        "battery": get_battery(mac),
    }