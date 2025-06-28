"""
System constants that should never change.

These are technical/system limits, not user preferences.
User-configurable values should go in config.yaml instead.
"""

# System/Protocol limits (keep only used constants)
VERBOSE_LOGGING_THRESHOLD = 2  # Standard logging level

# File processing technical limits (keep only used constants)
MIN_FILES_FOR_DUPLICATE_DETECTION = 2  # Minimum files needed for duplicate detection
AUDIO_SMALL_FOLDER_THRESHOLD = 3  # Small folder threshold for audio analysis
AUDIO_MEDIUM_FOLDER_THRESHOLD = 20  # Medium folder threshold for audio analysis
ERROR_MESSAGE_TRUNCATE_LENGTH = 100  # Maximum length for error message display
FFPROBE_ERROR_PARTS_COUNT = 3  # Expected parts count in FFprobe error parsing
FUTURE_RESULT_TUPLE_LENGTH = 3  # Expected tuple length in async results
MAX_DISPLAYED_GROUPS = 10  # Maximum duplicate groups to display

# Video quality technical thresholds (keep only used constants)
GRAIN_HIGH_THRESHOLD = 0.7  # High grain content threshold
