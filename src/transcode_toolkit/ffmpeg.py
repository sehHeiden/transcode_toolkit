from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .config import AudioPreset


class MediaInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    bitrate: int
    duration: float

    @classmethod
    def from_path(cls, path: Path) -> MediaInfo:
        raw = probe_media(path)
        fmt = raw.get("format", {})
        return cls(
            bitrate=int(fmt.get("bit_rate", 0)),
            duration=float(fmt.get("duration", 0)),
        )


def validate_duration(path: Path, expected: float, tolerance: float = 0.5) -> bool:
    if expected <= 0:
        return True
    try:
        return abs(MediaInfo.from_path(path).duration - expected) <= tolerance
    except Exception:
        return False


def cleanup(path: Path) -> None:
    if path.exists():
        path.unlink()


@lru_cache(maxsize=1024)
def probe(path: str, _mtime: float) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def probe_media(path: Path) -> dict:
    return probe(str(path), path.stat().st_mtime)


def run_ffmpeg(cmd: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)


@lru_cache(maxsize=1)
def available_encoders() -> set[str]:
    result = subprocess.run(["ffmpeg", "-encoders", "-v", "quiet"], capture_output=True, text=True)
    return {
        line.split()[1] for line in result.stdout.splitlines() if len(line.split()) > 1 and line.lstrip()[0] in "VAS"
    }


def measure_snr(path: Path) -> float:
    cmd = ["ffmpeg", "-i", str(path), "-af", "astats=metadata=1", "-f", "null", "-", "-v", "info"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    rms_lines = [ln for ln in result.stderr.splitlines() if "RMS" in ln and "dB" in ln]
    if not rms_lines:
        return 96.0
    return _parse_rms_db(rms_lines[-1])


def measure_vmaf(original: Path, encoded: Path) -> float:
    cmd = [
        "ffmpeg",
        "-i",
        str(original),
        "-i",
        str(encoded),
        "-lavfi",
        "[0:v][1:v]libvmaf",
        "-f",
        "null",
        "-",
        "-v",
        "info",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return _parse_vmaf_score(result.stderr)


def has_vmaf_support() -> bool:
    result = subprocess.run(["ffmpeg", "-filters", "-v", "quiet"], capture_output=True, text=True)
    return "libvmaf" in result.stdout


def build_audio_cmd(source: Path, output: Path, bitrate: int, preset: AudioPreset) -> list[str]:
    cmd = ["ffmpeg", "-i", str(source), "-c:a", "libopus", "-b:a", str(bitrate)]
    if preset.application:
        cmd.extend(["-application", preset.application])
    if preset.channels:
        cmd.extend(["-ac", str(preset.channels)])
    if preset.cutoff:
        cmd.extend(["-cutoff", str(preset.cutoff)])
    cmd.extend(["-y", str(output)])
    return cmd


def build_video_cmd(source: Path, output: Path, *, codec: str, crf: int, speed: str) -> list[str]:
    cmd = ["ffmpeg", "-i", str(source), "-c:v", codec, "-crf", str(crf)]
    cmd.extend(["-preset", speed, "-c:a", "copy", "-y", str(output)])
    return cmd


def _parse_rms_db(line: str) -> float:
    for part in line.split():
        try:
            return float(part)
        except ValueError:
            continue
    return 96.0


def _parse_vmaf_score(stderr: str) -> float:
    lines = [ln for ln in stderr.splitlines() if "VMAF score" in ln]
    return float(lines[-1].split(":")[-1].strip()) if lines else 0.0
