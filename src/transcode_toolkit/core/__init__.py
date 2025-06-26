"""Core abstractions and utilities for the media toolkit."""

from .base import MediaProcessor, ProcessingError, ProcessingResult, ProcessingStatus
from .config import ConfigManager, ProcessingOptions, with_config_overrides
from .duplicate_finder import DuplicateFinder, FileInfo
from .ffmpeg import FFmpegError, FFmpegProbe, FFmpegProcessor
from .file_manager import BackupStrategy, FileManager
from .video_analysis import QualityDecision, VideoComplexity

__all__ = [
    "BackupStrategy",
    "ConfigManager",
    "DuplicateFinder",
    "FFmpegError",
    "FFmpegProbe",
    "FFmpegProcessor",
    "FileInfo",
    "FileManager",
    "MediaProcessor",
    "ProcessingError",
    "ProcessingOptions",
    "ProcessingResult",
    "ProcessingStatus",
    "QualityDecision",
    "VideoComplexity",
    "with_config_overrides",
]
