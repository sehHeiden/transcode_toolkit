"""Core abstractions and utilities for the media toolkit."""

from .base import MediaProcessor, ProcessingResult, ProcessingError
from .ffmpeg import FFmpegProbe, FFmpegProcessor
from .config import ConfigManager, ProcessingOptions
from .file_manager import FileManager, BackupStrategy

__all__ = [
    "MediaProcessor",
    "ProcessingResult", 
    "ProcessingError",
    "FFmpegProbe",
    "FFmpegProcessor",
    "ConfigManager",
    "ProcessingOptions",
    "FileManager",
    "BackupStrategy",
]
