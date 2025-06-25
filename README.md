# Transcode Toolkit

Unified toolkit for estimating space savings and batch transcoding **video** (H.265/HEVC) and **audio** (Opus) libraries with a modern, modular architecture.

## ✨ Features

- **🎯 Smart Audio Estimation**: Get intelligent preset recommendations with SNR-based bitrate scaling
- **⚙️ Modular Architecture**: Clean src-layout structure following Python best practices
- **🛠️ CLI Interface**: Comprehensive command-line interface with subcommands
- **📊 Preset Comparison**: Compare all audio presets side-by-side
- **🔧 Built-in Utilities**: Duplicate file detection, backup management, system info
- **🚀 Easy Usage**: Simple Makefile commands for common operations

---

## Project Structure

```
transcode_toolkit/
├── src/transcode_toolkit/     # Main package (src-layout)
│   ├── __init__.py
│   ├── cli/                   # Command-line interface
│   │   ├── main.py           # Main CLI entry point
│   │   └── commands/         # Command handlers
│   ├── config/               # Configuration management
│   │   ├── settings.py       # Config classes and loading
│   │   └── constants.py      # System constants
│   ├── core/                 # Core functionality
│   │   ├── base.py          # Base classes
│   │   ├── config.py        # Config manager
│   │   ├── ffmpeg.py        # FFmpeg wrapper
│   │   ├── file_manager.py  # File operations
│   │   └── duplicate_finder.py # Duplicate detection
│   ├── processors/          # Media processors
│   │   ├── audio_processor.py
│   │   └── video_processor.py
│   ├── audio/              # Audio utilities
│   │   └── estimate.py     # Size estimation
│   ├── video/              # Video utilities
│   │   ├── transcode.py    # H.265 encoding
│   │   ├── estimate.py     # Size estimation
│   │   └── blur_quality.py # Quality analysis
│   └── tui/                # Future: Text UI
├── tests/                  # Test files
├── config.yaml            # Configuration file
├── Makefile              # Convenient command shortcuts
└── pyproject.toml        # Project metadata
```

---

## Quick Start

### Installation

```bash
# Clone and set up environment
git clone <repository-url>
cd transcode_toolkit

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Basic Usage

```bash
# Show help
python -m src.transcode_toolkit.cli.main --help

# Check system information
python -m src.transcode_toolkit.cli.main utils info

# Run tests
pytest tests/ -v
```

## 🎵 Audio Commands

### Estimate Size Savings

```bash
# Get intelligent recommendations for all presets
python -m src.transcode_toolkit.cli.main audio estimate "C:\Music"

# Works with paths containing special characters
python -m src.transcode_toolkit.cli.main audio estimate "F:\Audio\Hörbücher"

# Save estimation results to CSV
python -m src.transcode_toolkit.cli.main audio estimate "C:\Music" --csv results.csv

# Analyze specific preset only
python -m src.transcode_toolkit.cli.main audio estimate "C:\Music" --preset audiobook --no-compare
```

### Transcode Audio Files

```bash
# Transcode with specific preset (always recursive)
python -m src.transcode_toolkit.cli.main audio transcode "C:\Music" --preset audiobook --recursive

# Works with paths containing special characters
python -m src.transcode_toolkit.cli.main audio transcode "F:\Audio\Hörbücher" --preset audiobook --recursive

# Different audio presets:
python -m src.transcode_toolkit.cli.main audio transcode "C:\Music" --preset music --recursive
python -m src.transcode_toolkit.cli.main audio transcode "C:\Music" --preset high --recursive
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

## 🎬 Video Commands

```bash
# Estimate video size savings
python -m src.transcode_toolkit.cli.main video estimate "D:\Videos"

# Transcode with GPU acceleration
python -m src.transcode_toolkit.cli.main video transcode "D:\Videos" --gpu --crf 22

# Transcode with CPU (default)
python -m src.transcode_toolkit.cli.main video transcode "D:\Videos" --crf 24
```

## 🔧 Utility Commands

```bash
# Find duplicate files
python -m src.transcode_toolkit.cli.main utils duplicates "C:\Music" --extensions .mp3 .flac

# Clean up backup files
python -m src.transcode_toolkit.cli.main utils cleanup "C:\Music" --force

# Show configuration and system info
python -m src.transcode_toolkit.cli.main utils info
```

## 📋 Available Audio Presets

- **music**: High quality for music (128k, audio application)
- **audiobook**: Optimized for speech (32k, voip application, mono)
- **low**: Lower bitrate option (64k)
- **high**: Higher quality option (192k)

Presets automatically scale bitrate based on source material quality (SNR) when enabled.

---

## 🧪 Development

```bash
# Run type checking
make type-check
# or
python -m mypy .

# Run tests
pytest tests/ -v

# Run linting
ruff check src/ tests/

# Run all checks
ty check .  # Alternative type checker
```

---

## 📋 Requirements

- Python 3.13+
- FFmpeg (with ffprobe) in PATH
- Optional: NVIDIA drivers for GPU-accelerated video transcoding

## 🗂️ Configuration

Edit `config.yaml` to customize:
- Audio presets (bitrates, applications, SNR thresholds)
- Video settings
- Global options (workers, backup strategy)

## 📈 Roadmap

- **TUI Interface**: Terminal-based user interface in `src/transcode_toolkit/tui/`
- **Enhanced Video Analysis**: Blur/VMAF quality metrics
- **Audio Classification**: Automatic speech/music detection
- **Batch Processing**: Queue management for large operations
- **Progress Monitoring**: Enhanced progress tracking and estimation
