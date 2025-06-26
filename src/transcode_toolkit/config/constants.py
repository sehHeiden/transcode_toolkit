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

# System performance thresholds (technical limits)
CPU_CRITICAL_THRESHOLD = 90  # Critical CPU usage requiring immediate cooling
CPU_HIGH_THRESHOLD = 80  # High CPU usage requiring brief cooling
CPU_MODERATE_THRESHOLD = 70  # Moderate CPU usage for audio processing
CPU_VIDEO_HIGH_THRESHOLD = 85  # High CPU threshold for video processing
CPU_VIDEO_MODERATE_THRESHOLD = 60  # Moderate CPU threshold for video processing

# Temperature thresholds (technical limits in Celsius)
TEMP_CRITICAL_THRESHOLD = 80  # Critical temperature requiring extended cooling
TEMP_HIGH_THRESHOLD = 70  # High temperature requiring moderate cooling
TEMP_MODERATE_THRESHOLD = 60  # Moderate temperature threshold
TEMP_VIDEO_HIGH_THRESHOLD = 65  # High temperature threshold for video
TEMP_VIDEO_MODERATE_THRESHOLD = 55  # Moderate temperature threshold for video

# Memory usage thresholds (technical limits)
MEMORY_HIGH_THRESHOLD = 80  # High memory usage indicating thermal stress
MEMORY_VIDEO_HIGH_THRESHOLD = 75  # High memory threshold for video processing

# File processing technical limits
MIN_FILES_FOR_DUPLICATE_DETECTION = 2  # Minimum files needed for duplicate detection
AUDIO_SMALL_FOLDER_THRESHOLD = 3  # Small folder threshold for audio analysis
AUDIO_MEDIUM_FOLDER_THRESHOLD = 20  # Medium folder threshold for audio analysis
ERROR_MESSAGE_TRUNCATE_LENGTH = 100  # Maximum length for error message display
FFPROBE_ERROR_PARTS_COUNT = 3  # Expected parts count in FFprobe error parsing
FUTURE_RESULT_TUPLE_LENGTH = 3  # Expected tuple length in async results
MAX_DISPLAYED_GROUPS = 10  # Maximum duplicate groups to display
PARALLEL_PROCESSING_THRESHOLD = 20  # File count threshold for parallel processing
SMALL_FILE_SET_THRESHOLD = 10  # Threshold for small vs large file sets

# Video processing technical constants
RESOLUTION_4K = 2160  # 4K resolution height
RESOLUTION_1440P = 1440  # 1440p resolution height
RESOLUTION_1080P = 1080  # 1080p resolution height
RESOLUTION_720P = 720  # 720p resolution height
RESOLUTION_480P = 480  # 480p processing resolution limit
BGR_CHANNEL_COUNT = 3  # Number of channels in BGR color format

# Video quality technical thresholds
COMPLEXITY_HIGH_THRESHOLD = 0.8  # High complexity content threshold
COMPLEXITY_MEDIUM_THRESHOLD = 0.6  # Medium complexity content threshold
COMPLEXITY_LOW_THRESHOLD = 0.3  # Low complexity content threshold
GRAIN_HIGH_THRESHOLD = 0.7  # High grain content threshold

# SSIM quality technical thresholds
SSIM_EXCELLENT_THRESHOLD = 0.95  # Excellent SSIM quality
SSIM_GOOD_THRESHOLD = 0.92  # Good SSIM quality
SSIM_ACCEPTABLE_THRESHOLD = 0.90  # Acceptable SSIM quality
SSIM_4K_ADJUSTMENT = -0.02  # SSIM adjustment for 4K content
SSIM_HD_ADJUSTMENT = 0.01  # SSIM adjustment for HD content
SSIM_GRAIN_ADJUSTMENT = -0.03  # SSIM adjustment for grainy content

# CRF encoding technical thresholds
CRF_HIGH_QUALITY_THRESHOLD = 20  # High quality CRF threshold
CRF_LOW_QUALITY_THRESHOLD = 26  # Low quality CRF threshold

# File size estimation technical factors
COMPLEXITY_HIGH_SIZE_FACTOR = 1.2  # Size factor for high complexity content
COMPLEXITY_LOW_SIZE_FACTOR = 0.8  # Size factor for low complexity content
CRF_HIGH_QUALITY_SIZE_FACTOR = 1.3  # Size factor for high quality CRF
CRF_LOW_QUALITY_SIZE_FACTOR = 0.7  # Size factor for low quality CRF
