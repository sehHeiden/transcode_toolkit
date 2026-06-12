from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _generate(tmp_path: Path, name: str, args: list[str]) -> Path:
    out = tmp_path / name
    subprocess.run(args, capture_output=True, check=True)
    return out


@pytest.fixture
def speech_wav(tmp_path: Path) -> Path:
    out = tmp_path / "speech.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "flite=text='Hello world. This is a test of speech synthesis for audio transcoding.'",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(out),
        ],
        capture_output=True,
        check=True,
    )
    return out


@pytest.fixture
def speech_mp3(tmp_path: Path) -> Path:
    out = tmp_path / "speech.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "flite=text='Hello world. This is a test of speech synthesis for audio transcoding.'",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            "-y",
            str(out),
        ],
        capture_output=True,
        check=True,
    )
    return out


@pytest.fixture
def speech_mp3_low(tmp_path: Path) -> Path:
    out = tmp_path / "speech_low.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "flite=text='Hello world. This is a test of speech synthesis for audio transcoding.'",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "64k",
            "-y",
            str(out),
        ],
        capture_output=True,
        check=True,
    )
    return out


@pytest.fixture
def already_opus(tmp_path: Path) -> Path:
    out = tmp_path / "existing.opus"
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:a",
            "libopus",
            "-b:a",
            "128k",
            "-y",
            str(out),
        ],
        capture_output=True,
        check=True,
    )
    return out


@pytest.fixture
def noisy_video(tmp_path: Path) -> Path:
    out = tmp_path / "noisy.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=25:duration=2",
            "-vf",
            "noise=alls=20:allf=t",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-y",
            str(out),
        ],
        capture_output=True,
        check=True,
    )
    return out


@pytest.fixture
def clean_video(tmp_path: Path) -> Path:
    out = tmp_path / "clean.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=25:duration=2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-y",
            str(out),
        ],
        capture_output=True,
        check=True,
    )
    return out


@pytest.fixture
def compressed_video(tmp_path: Path) -> Path:
    out = tmp_path / "compressed.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=25:duration=2",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "28",
            "-y",
            str(out),
        ],
        capture_output=True,
        check=True,
    )
    return out


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    out = tmp_path / "tiny.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=5:duration=1",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-y",
            str(out),
        ],
        capture_output=True,
        check=True,
    )
    return out


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text("""\
audio:
  size_keep_ratio: 0.95
  extensions: [".flac", ".mp3", ".wav", ".aac", ".m4a", ".ogg", ".opus", ".wma"]
  presets:
    music:
      bitrate: "128k"
      application: "audio"
      min_snr_db: 70.0
      snr_bitrate_scale: true
    audiobook:
      bitrate: "64k"
      application: "voip"
      channels: 1
      min_snr_db: 50.0
      snr_bitrate_scale: true
    low:
      bitrate: "96k"
      application: "audio"
      min_snr_db: 60.0
      snr_bitrate_scale: true

video:
  min_savings_percent: 10
  size_keep_ratio: 0.95
  extensions: [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm"]

global:
  workers: null
  create_backups: true
""")
    return config
