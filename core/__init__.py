"""Core abstractions and utilities for the media toolkit."""

from .base import MediaProcessor, ProcessingResult, ProcessingError, ProcessingStatus
from .ffmpeg import FFmpegProbe, FFmpegProcessor, FFmpegError
from .config import ConfigManager, ProcessingOptions, with_config_overrides
from .file_manager import FileManager, BackupStrategy

__all__ = [
    "MediaProcessor",
    "ProcessingResult",
    "ProcessingError",
    "ProcessingStatus",
    "FFmpegProbe",
    "FFmpegProcessor",
    "FFmpegError",
    "ConfigManager",
    "ProcessingOptions",
    "with_config_overrides",
    "FileManager",
    "BackupStrategy",
]
