from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from typing import Any

from config import settings
from services import speaker

log = logging.getLogger(__name__)

_HISTORY_MAX = 20
ASOUND_CONF = "/etc/asound.conf"


def _asound_conf_content(mac: str) -> str:
    """Return asound.conf content routing default ALSA output to a bluealsa device."""
    return (
        "pcm.!default {\n"
        "    type bluealsa\n"
        f'    device "{mac}"\n'
        '    profile "a2dp"\n'
        "}\n"
        "\n"
        "ctl.!default {\n"
        "    type bluealsa\n"
        f'    device "{mac}"\n'
        '    profile "a2dp"\n'
        "}\n"
    )


def set_system_speaker(mac: str, path: str = ASOUND_CONF) -> None:
    """Atomically rewrite asound.conf so the device becomes the system default output."""
    if os.path.exists(path):
        shutil.copy2(path, f"{path}.bak")
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(_asound_conf_content(mac))
    os.replace(tmp, path)


def get_system_speaker(path: str = ASOUND_CONF) -> str | None:
    """Return the MAC of the current system default speaker, or None."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None
    match = re.search(r'device\s+"([0-9A-Fa-f:]{17})"', text)
    return match.group(1).upper() if match else None


def _run(cmd: list[str], timeout: int = 30) -> str:
    """Run a bluetoothctl command synchronously and return stdout."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.stdout or ""
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.warning("Command %s failed: %s", cmd[0], exc)
        return ""


def _device_info(mac: str) -> dict[str, Any]:
    """Return {mac, name, paired, connected} for a device."""
    info: dict[str, Any] = {"mac": mac, "name": mac, "paired": False, "connected": False}
    out = _run(["bluetoothctl", "info", mac])
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Name:"):
            info["name"] = line.split(":", 1)[1].strip() or mac
        elif line.startswith("Paired:"):
            info["paired"] = line.split(":", 1)[1].strip() == "yes"
        elif line.startswith("Connected:"):
            info["connected"] = line.split(":", 1)[1].strip() == "yes"
    return info


def list_devices() -> list[dict[str, Any]]:
    """List all known Bluetooth devices with their state."""
    out = _run(["bluetoothctl", "devices"])
    macs = [line.split()[1] for line in out.splitlines() if line.startswith("Device ")]
    return [_device_info(mac) for mac in macs]


def scan(timeout: int = 12) -> list[dict[str, Any]]:
    """Scan for Bluetooth devices (blocking for ~timeout seconds)."""
    _run(["bluetoothctl", "power", "on"])
    _run(["bluetoothctl", "--timeout", str(timeout), "scan", "on"], timeout=timeout + 10)
    return list_devices()


def connect(mac: str) -> dict[str, Any]:
    """Pair (if needed), trust, and connect to a device.

    On success the device becomes the app's default speaker and is saved to
    the fast-connect history.
    """
    info = _device_info(mac)
    if not info["paired"]:
        _run(["bluetoothctl", "pair", mac], timeout=60)
        info = _device_info(mac)
    _run(["bluetoothctl", "trust", mac])
    _run(["bluetoothctl", "connect", mac], timeout=60)
    info = _device_info(mac)
    if info["connected"]:
        settings.speaker_mac = mac
        set_system_speaker(mac)
        _save_history(mac, info["name"])
        actual = speaker.get_volume(mac)
        if actual is not None:
            # Speaker is the source of truth: sync saved volume to reality.
            save_volume(mac, actual)
        else:
            # Speaker doesn't report volume: restore the last known value.
            saved_volume = _get_history_volume(mac)
            if saved_volume is not None:
                speaker.set_volume(mac, saved_volume)
        log.info("Connected BT device %s (%s) - set as system speaker", info["name"], mac)
    return info


def disconnect(mac: str) -> None:
    _run(["bluetoothctl", "disconnect", mac])


def get_status() -> dict[str, Any]:
    """Return current connection status, system default speaker, and history."""
    devices = list_devices()
    connected = next((d for d in devices if d["connected"]), None)
    default_mac = get_system_speaker()
    system_default = next((d for d in devices if d["mac"].upper() == default_mac), None)
    if system_default is None and default_mac:
        system_default = {"mac": default_mac, "name": default_mac, "paired": False, "connected": False}
    return {
        "connected": connected is not None,
        "device": connected,
        "system_default": system_default,
        "history": _load_history(),
    }


def load_default_speaker() -> None:
    """Set the default speaker to the most recently connected device.

    Also reconciles /etc/asound.conf so the system default output matches,
    and syncs the saved volume with the speaker's actual volume so the UI
    reflects reality after an app restart.
    """
    history = _load_history()
    if history:
        settings.speaker_mac = history[0]["mac"]
        if get_system_speaker() != settings.speaker_mac:
            set_system_speaker(settings.speaker_mac)
        actual = speaker.get_volume(settings.speaker_mac)
        if actual is not None:
            save_volume(settings.speaker_mac, actual)
        log.info("Default speaker set to %s (%s)", history[0]["name"], history[0]["mac"])


def _history_path() -> str:
    return settings.bt_history_file


def _load_history() -> list[dict[str, Any]]:
    try:
        with open(_history_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("devices", [])
    except (OSError, json.JSONDecodeError):
        return []


def _save_history(mac: str, name: str) -> None:
    history = _load_history()
    existing = next((d for d in history if d["mac"] == mac), {})
    history = [d for d in history if d["mac"] != mac]
    entry = {"mac": mac, "name": name, "last_connected": int(time.time())}
    if "volume" in existing:
        entry["volume"] = existing["volume"]
    history.insert(0, entry)
    history = history[:_HISTORY_MAX]
    try:
        with open(_history_path(), "w", encoding="utf-8") as f:
            json.dump({"devices": history}, f, indent=2)
    except OSError as exc:
        log.warning("Failed to save BT history: %s", exc)


def _get_history_volume(mac: str) -> int | None:
    """Return the saved volume for a device, or None."""
    for d in _load_history():
        if d["mac"] == mac and "volume" in d:
            return d["volume"]
    return None


def save_volume(mac: str, volume: int) -> None:
    """Persist the volume for a device so it is re-applied on connect."""
    history = _load_history()
    for d in history:
        if d["mac"] == mac:
            d["volume"] = volume
            break
    else:
        history.insert(0, {"mac": mac, "name": mac, "last_connected": int(time.time()), "volume": volume})
    try:
        with open(_history_path(), "w", encoding="utf-8") as f:
            json.dump({"devices": history}, f, indent=2)
    except OSError as exc:
        log.warning("Failed to save BT history: %s", exc)