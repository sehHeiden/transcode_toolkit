"""
System constants that should never change.

These are technical/system limits, not user preferences.
User-configurable values should go in config.yaml instead.
"""

# System/Protocol limits (never change)
FFMPEG_DEFAULT_TIMEOUT = 300  # 5 minutes - system timeout
FFPROBE_TIMEOUT = 30  # 30 seconds - system timeout
MAX_WORKERS_CAP = 8  # System performance limit
VERBOSE_LOGGING_THRESHOLD = 2  # Standard logging level

# Technical audio specifications (rarely change)
LOSSLESS_SNR = 96.0  # 16-bit theoretical maximum
DSD_SNR = 120.0  # DSD technical specification
DEFAULT_CONSERVATIVE_SNR = 60.0  # Safe fallback

# Technical minimum bitrates (codec specifications)
VOICE_MIN_BITRATE = 32000  # Technical minimum for intelligible speech
MUSIC_MIN_BITRATE = 64000  # Technical minimum for acceptable music quality

# Opus reprocessing tolerance (technical threshold)
OPUS_REPROCESS_TOLERANCE = 1.2  # 20% tolerance for Opus→Opus conversion
