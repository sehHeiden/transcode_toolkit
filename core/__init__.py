"""Core abstractions and utilities for the media toolkit."""

from .base import MediaProcessor, ProcessingError, ProcessingResult, ProcessingStatus
from .config import ConfigManager, ProcessingOptions, with_config_overrides
from .ffmpeg import FFmpegError, FFmpegProbe, FFmpegProcessor
from .file_manager import BackupStrategy, FileManager

__all__ = [
    "BackupStrategy",
    "ConfigManager",
    "FFmpegError",
    "FFmpegProbe",
    "FFmpegProcessor",
    "FileManager",
    "MediaProcessor",
    "ProcessingError",
    "ProcessingOptions",
    "ProcessingResult",
    "ProcessingStatus",
    "with_config_overrides",
]
