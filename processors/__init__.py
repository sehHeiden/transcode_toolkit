"""Media processors using the new core architecture."""

from .audio_processor import AudioProcessor
from .video_processor import VideoProcessor

__all__ = [
    "AudioProcessor",
    "VideoProcessor",
]
