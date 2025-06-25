"""Transcode Toolkit - Advanced media transcoding and processing utilities."""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Your Name"
__description__ = "Advanced media transcoding and processing utilities"

# Public API exports
from .config import MediaToolkitConfig, get_config
from .core import (
    BackupStrategy,
    ConfigManager,
    DuplicateFinder,
    FFmpegError,
    FFmpegProbe,
    FFmpegProcessor,
    FileManager,
    MediaProcessor,
    ProcessingError,
    ProcessingResult,
    ProcessingStatus,
    with_config_overrides,
)
from .processors import AudioProcessor, VideoProcessor

__all__ = [
    # Configuration
    "MediaToolkitConfig",
    "get_config",
    "ConfigManager",
    "with_config_overrides",
    # Core functionality
    "MediaProcessor",
    "FFmpegProcessor",
    "FFmpegProbe",
    "FileManager",
    "DuplicateFinder",
    # Processors
    "AudioProcessor",
    "VideoProcessor",
    # Enums and data classes
    "ProcessingStatus",
    "ProcessingResult",
    "BackupStrategy",
    # Exceptions
    "ProcessingError",
    "FFmpegError",
]
