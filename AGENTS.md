# Transcode Toolkit - Agent Instructions

## Project Overview

Unified CLI/TUI toolkit for video/audio transcoding and size estimation. Uses FFmpeg for media processing.

## Essential Commands

### Development Setup
```bash
uv sync
```

### Code Quality (CI order)
```bash
uv run ruff check src/ tests/   # Lint
uv run mypy src/                 # Type check
uv run ty check src/             # Alt type checker
uv run pytest tests/ -v          # Test
```

### Single Test
```bash
uv run pytest tests/test_file.py::test_name -v
```

### Run CLI
```bash
uv run transcode-toolkit --help
uv run transcode-toolkit audio estimate .
uv run transcode-toolkit video transcode . --gpu
uv run transcode-toolkit tui
```

## Architecture

- **src-layout**: `src/transcode_toolkit/` is the package root
- **Entry point**: `__main__.py` → `cli.py` (Typer app with subcommands)
- **CLI**: Typer with subcommand groups (audio, video, utils, tui)
- **Config**: `config.yaml` at repo root, loaded via Pydantic (`config.py`)
- **Modules**:
  - `ffmpeg.py` - FFmpeg/ffprobe wrapper, `MediaInfo` model, `validate_duration`, `cleanup`, encoder detection
  - `config.py` - `ToolkitConfig`, `AudioPreset`, `VideoConfig`, `GlobalConfig` (all frozen Pydantic models)
  - `chain.py` - `Chain` pipeline for discover → filter → transcode
  - `audio.py` - Audio transcode/estimate (Opus output, SNR-based bitrate scaling)
  - `video.py` - Video transcode/estimate (x265/AV1/NVENC, VMAF quality metrics)
  - `tui/` - Textual-based TUI
  - `types.py` - `ProcessingResult`, `ProcessingStatus`

## Key Conventions

- **Ruff**: `select = ["ALL"]`, ignores in `pyproject.toml`
- **Mypy**: targets `src/` only, no `ignore_missing_imports`
- **Tests**: `pytest` with `pytest-cov`; FFmpeg-generated fixtures via `conftest.py`; no mocking
- **Dependencies**: Runtime in `[project]`, dev in `[dependency-groups] dev`
- **Python**: Requires 3.13+ (`requires-python = ">=3.13"`)

## Design Principles

- **YAGNI**: No dead code. Removed `Chain.sample/estimate`, `with_workers`, `estimate_audio_directory`, `hypothesis`, unused config fields
- **DRY**: `MediaInfo` centralizes probe data, `validate_duration` and `cleanup` shared in `ffmpeg.py`, `estimate_audio` takes no unused `config` param
- **Duration validation**: Output duration checked against input (0.5s tolerance) before replacing originals
- **Error handling**: Failed files appear as `ProcessingStatus.ERROR` in results (not silently dropped)

## Gotchas

- FFmpeg must be in PATH for runtime functionality
- Config loaded from `config.yaml` at repo root (not in package)
- Tests generate real media fixtures via FFmpeg (`conftest.py`); no mocking
- `audio.py`: Output gets `.opus` extension, original deleted after success
- `video.py`: Audio stream copied (`-c:a copy`), not re-encoded
- `probe()` is `lru_cache`d on `(path, mtime)` - stale cache possible if file changes during session
