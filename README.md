# Transcode Toolkit

[![CI/CD](https://github.com/sehHeiden/transcode_toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/sehHeiden/transcode_toolkit/actions)
[![Python](https://img.shields.io/badge/Python-3.13+-3776ab.svg?logo=python&logoColor=white)](https://python.org)
[![License: GPL v2+](https://img.shields.io/badge/License-GPL%20v2%2B-blue.svg)](LICENSE)

Unified toolkit for estimating space savings and batch transcoding **video** (H.265/AV1/NVENC) and **audio** (Opus) libraries.

## Quick Start

```bash
git clone https://github.com/sehHeiden/transcode_toolkit.git
cd transcode_toolkit
uv sync

uv run transcode-toolkit --help
```

## Usage

```bash
# Audio
uv run transcode-toolkit audio estimate .
uv run transcode-toolkit audio transcode . --preset music
uv run transcode-toolkit audio estimate . --preset audiobook --csv results.csv

# Video
uv run transcode-toolkit video estimate .
uv run transcode-toolkit video transcode . --crf 28 --speed fast
uv run transcode-toolkit video transcode . --gpu

# TUI
uv run transcode-toolkit tui

# Utilities
uv run transcode-toolkit utils duplicates .
uv run transcode-toolkit utils info
```

## Audio Presets

| Preset | Bitrate | Application | Channels | Use Case |
|--------|---------|-------------|----------|----------|
| music | 128k | audio | stereo | General music |
| high | 192k | audio | stereo | High quality music |
| low | 96k | audio | stereo | Space-constrained music |
| audiobook | 64k | voip | mono | Speech/audiobooks |
| audiobook_stereo | 96k | voip | stereo | Stereo speech |

Bitrate is automatically scaled down based on source SNR when `snr_bitrate_scale` is enabled.

## Project Structure

```
src/transcode_toolkit/
├── __init__.py          # Package version
├── __main__.py          # Entry point
├── cli.py               # Typer CLI (audio, video, utils, tui commands)
├── config.py            # Pydantic config models (ToolkitConfig, AudioPreset, ...)
├── ffmpeg.py            # FFmpeg wrapper, MediaInfo, validate_duration
├── chain.py             # Chain pipeline with Chainable Protocol
├── audio.py             # Audio transcode/estimate logic
├── video.py             # Video transcode/estimate logic
├── types.py             # ProcessingResult, ProcessingStatus
└── tui/                 # Textual TUI
    └── __init__.py
```

## Configuration

Edit `config.yaml` at the repo root. Key settings:

- **audio.presets** - Bitrate, application, SNR thresholds per preset
- **audio.size_keep_ratio** - Skip if output >= this ratio of source (default 0.95)
- **video.min_savings_percent** - Skip if savings below this % (default 10)
- **global.create_backups** - Create `.bak` files before replacing (default true)
- **global.workers** - Thread pool size (default 2)

## Development

```bash
uv run ruff check src/ tests/   # Lint
uv run mypy src/                 # Type check
uv run pytest tests/ -v          # Test (requires FFmpeg)
```

## Docker

```bash
docker compose run transcode audio estimate /data
docker compose run transcode video transcode /data --gpu
```

## Requirements

- Python 3.13+
- FFmpeg (with ffprobe) in PATH
- Optional: NVIDIA drivers for GPU-accelerated video transcoding
