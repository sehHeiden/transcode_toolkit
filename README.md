# Transcode Toolkit

Unified helper scripts for estimating space savings, analysing quality and batch transcoding **video** (H.265/HEVC) and **audio** (Opus) libraries.

## ✨ New Features

- **🎯 Smart Audio Estimation**: Get intelligent preset recommendations
- **🔧 Dependency Checking**: Automatic FFmpeg installation verification
- **⚙️ Unified Presets**: Estimation and transcoding use identical settings
- **📊 Preset Comparison**: Compare all audio presets side-by-side

---

## Folder structure

```
transcode_toolkit/
├── media_toolkit.py          # single CLI entry‑point
│
├── video/
│   ├── __init__.py
│   ├── transcode.py          # batch H.265 encoding (CPU or NVENC)
│   ├── estimate.py           # size‑saving estimator
│   └── blur_quality.py       # (placeholder) blur/VMAF analysis
│
├── audio/
│   ├── __init__.py
│   ├── estimate.py           # Opus size estimator
│   └── opus_quality.py       # (placeholder) speech/music detection
│
├── tests/                    # pytest tests
│   └── test_transcode.py
│
└── pyproject.toml            # uv / PEP 621 metadata
```

---

## Quick start (uv)

```bash
# 1 – create virtual env & lock file
aud init                 # generates .venv + pyproject.toml + uv.lock

# 2 – install runtime dependencies
uv sync                  # identical to 'uv pip install -r'

# 3 – run tests
uv run pytest -q

# 4 – explore CLI
uv run python media_toolkit.py -h
```

### 🆕 New: Smart Audio Estimation

```bash
# 1. Check if FFmpeg is installed
uv run python check_deps.py

# 2. Get intelligent recommendations for all presets
uv run python media_toolkit.py audio estimate /path/to/audio

# 3. Save detailed comparison report
uv run python media_toolkit.py audio estimate /path/to/audio --json audio_analysis.json

# 4. Convert with recommended preset
uv run python media_toolkit.py audio transcode /path/to/audio --preset audiobook
```

**Sample Output:**
```
=== PRESET COMPARISON ===
Preset               Current  Estimated    Saving       %
------------------------------------------------------------
audiobook ★         120.5 MB    24.1 MB    96.4 MB   80.0%
low                 120.5 MB    72.3 MB    48.2 MB   40.0%
music               120.5 MB    96.4 MB    24.1 MB   20.0%

★ RECOMMENDED: audiobook
  → Bitrate: 32k
  → Application: voip
  → Frequency cutoff: 12000 Hz
  → Channels: 1
```

### Example workflows

```bash
# Estimate potential video savings (HEVC, conservative bitrate table)
uv run python media_toolkit.py video estimate  D:\Videos  --csv reports/video.csv

# Batch‑transcode with NVIDIA NVENC, keep originals as .bak
uv run python media_toolkit.py video transcode D:\Videos --gpu --workers 3 -v

# Compare all audio presets and get smart recommendations
uv run python media_toolkit.py audio estimate  E:\Audiobooks --compare

# Legacy: Analyze specific preset only
uv run python media_toolkit.py audio estimate  E:\Audiobooks --mode audiobook --csv reports/audio.csv
```

---

## Building a standalone Windows EXE (Nuitka)

```powershell
uv add nuitka zstandard ninja          # one‑time, dev‑only

# Include ffmpeg.exe, ffprobe.exe, nvEncodeAPI64.dll in 'bin/'
mkdir bin
# → copy binaries into bin/

uv run python -m nuitka media_toolkit.py `
    --standalone --onefile `
    --include-data-dir=bin=bin `
    --output-dir=dist
```

Result: `dist\media_toolkit.exe` – portable, GPU‑enabled.

---

## Roadmap

* Implement real **blur/VMAF** analysis in `video/blur_quality.py`.
* Add **speech/music classifier** in `audio/opus_quality.py`.
* Write GitHub Actions workflow to build Nuitka artefacts on every tag.
