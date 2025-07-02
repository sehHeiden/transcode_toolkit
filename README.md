# Transcode Toolkit

[![Python](https://img.shields.io/badge/Python-3.13+-3776ab.svg?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Package Manager](https://img.shields.io/badge/Package%20Manager-uv-ff6b35.svg?logo=python&logoColor=white)](https://github.com/astral-sh/uv)
[![Code Style](https://img.shields.io/badge/Code%20Style-Ruff-000000.svg?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![Type Checking](https://img.shields.io/badge/Type%20Checking-mypy-1f5582.svg)](https://mypy.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Tests-pytest-0a9edc.svg?logo=pytest&logoColor=white)](https://pytest.org/)

[![FFmpeg](https://img.shields.io/badge/FFmpeg-Required-007808.svg?logo=ffmpeg&logoColor=white)](https://ffmpeg.org/)
[![Video Codecs](https://img.shields.io/badge/Video-H.265%20%7C%20AV1%20%7C%20H.264-blue.svg)](https://en.wikipedia.org/wiki/High_Efficiency_Video_Coding)
[![Audio Codecs](https://img.shields.io/badge/Audio-Opus%20%7C%20AAC-orange.svg)](https://opus-codec.org/)
[![GPU Support](https://img.shields.io/badge/GPU-NVENC%20%7C%20Intel%20QSV%20%7C%20AMD%20AMF-76B900.svg)](https://developer.nvidia.com/ffmpeg)
[![Platforms](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/sehHeiden/transcode_toolkit)
[![Build Status](https://img.shields.io/badge/Build-Passing-success.svg)](https://github.com/sehHeiden/transcode_toolkit)
[![Coverage](https://img.shields.io/badge/Coverage-25%25-red.svg)](https://github.com/sehHeiden/transcode_toolkit)

Unified toolkit for estimating space savings and batch transcoding **video** (H.265/HEVC/AV1) and **audio** (Opus) libraries with a modern, modular architecture.

> 🚀 **Quick Start**: [`git clone`](#installation) → [`uv sync`](#installation) → [`python -m src.transcode_toolkit.cli.main --help`](#basic-usage)

---

## 🚀 Quick Actions

<div align="center">

| Action | Command | Description |
|--------|---------|-------------|
| 📊 **Estimate Audio** | `python -m src.transcode_toolkit.cli.main audio estimate .` | Get space savings estimates |
| 🎵 **Transcode Audio** | `python -m src.transcode_toolkit.cli.main audio transcode . --preset audiobook` | Convert audio files |
| 🎬 **Estimate Video** | `python -m src.transcode_toolkit.cli.main video estimate .` | Analyze video compression |
| 🎥 **Transcode Video** | `python -m src.transcode_toolkit.cli.main video transcode . --gpu` | GPU-accelerated video |
| 🔍 **Find Duplicates** | `python -m src.transcode_toolkit.cli.main utils duplicates .` | Detect duplicate files |
| ℹ️ **System Info** | `python -m src.transcode_toolkit.cli.main utils info` | Check configuration |

</div>

## ✨ Features at a Glance

<div align="center">

| 🎯 **Smart Analysis** | ⚡ **Performance** | 🛠️ **Flexibility** |
|:---------------------|:-------------------|:--------------------|
| SNR-based bitrate scaling | GPU acceleration | Modular architecture |
| Intelligent presets | Batch processing | Comprehensive CLI |
| Quality estimation | Multi-threaded | Easy configuration |

</div>

### Core Capabilities

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

## 🤝 Contributing

<div align="center">

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/sehHeiden/transcode_toolkit/pulls)
[![Good First Issue](https://img.shields.io/badge/Good%20First%20Issue-Available-blue.svg)](https://github.com/sehHeiden/transcode_toolkit/labels/good%20first%20issue)
[![Help Wanted](https://img.shields.io/badge/Help%20Wanted-Open-orange.svg)](https://github.com/sehHeiden/transcode_toolkit/labels/help%20wanted)

</div>

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Install** dependencies: `uv sync`
4. **Write** tests for your changes
5. **Format** code: `ruff format .`
6. **Run** tests: `pytest tests/`
7. **Commit** changes: `git commit -m 'Add amazing feature'`
8. **Push** to branch: `git push origin feature/amazing-feature`
9. **Open** a Pull Request

### Development Setup

```bash
# Clone and setup
git clone https://github.com/sehHeiden/transcode_toolkit.git
cd transcode_toolkit
uv sync

# Run the full test suite
pytest tests/ -v --cov=src

# Check code quality
ruff check . && ruff format . && mypy .
```

---

<div align="center">

**Made with ❤️ for the media transcoding community**

[![GitHub stars](https://img.shields.io/github/stars/sehHeiden/transcode_toolkit?style=social)](https://github.com/sehHeiden/transcode_toolkit/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/sehHeiden/transcode_toolkit?style=social)](https://github.com/sehHeiden/transcode_toolkit/network/members)
[![GitHub issues](https://img.shields.io/github/issues/sehHeiden/transcode_toolkit)](https://github.com/sehHeiden/transcode_toolkit/issues)

[Report Bug](https://github.com/sehHeiden/transcode_toolkit/issues) • [Request Feature](https://github.com/sehHeiden/transcode_toolkit/issues) • [Documentation](https://github.com/sehHeiden/transcode_toolkit/wiki)

</div>
