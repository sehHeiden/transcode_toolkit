"""Media processors using the new core architecture."""

from .audio_processor import AudioProcessor, AudioEstimator
from .video_processor import VideoProcessor, VideoEstimator

__all__ = [
    "AudioProcessor",
    "AudioEstimator", 
    "VideoProcessor",
    "VideoEstimator",
]
