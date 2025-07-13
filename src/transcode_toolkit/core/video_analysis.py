"""Shared video analysis functions for estimation and transcoding."""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from .ffmpeg import FFmpegProbe, FFmpegProcessor

LOG = logging.getLogger(__name__)

# Constants for magic values
COLOR_CHANNELS = 3
ANALYSIS_RESOLUTION = 360
RESERVED_SMALL_FILE_COUNT = 3
RESERVED_MEDIUM_FILE_COUNT = 10
FOUR_K_HEIGHT = 2160
TWO_K_HEIGHT = 1440
FULL_HD_HEIGHT = 1080
HD_HEIGHT = 720
SHORT_VIDEO_DURATION = 120
MEDIUM_VIDEO_DURATION = 600
HIGH_COMPLEXITY_THRESHOLD = 0.8
MEDIUM_COMPLEXITY_THRESHOLD = 0.6
LOW_COMPLEXITY_THRESHOLD = 0.3
HIGH_GRAIN_THRESHOLD = 0.7
SSIM_THRESHOLD_VERY_HIGH = 0.95
SSIM_THRESHOLD_HIGH = 0.92
SSIM_THRESHOLD_MEDIUM = 0.90

# Simple module-level caches
_file_cache: dict[Path, dict[str, Any]] = {}
_folder_quality_cache: dict[Path, float] = {}


@dataclass
class QualityDecision:
    """Result of effective CRF calculation."""

    effective_crf: int
    effective_preset: str
    limitation_reason: str | None
    predicted_ssim: float


@dataclass
class VideoComplexity:
    """Video content complexity analysis."""

    motion_score: float  # 0.0 to 1.0
    detail_score: float  # 0.0 to 1.0
    grain_score: float  # 0.0 to 1.0
    overall_complexity: float  # 0.0 to 1.0


@dataclass
class CRFCalculationParams:
    """Parameters for CRF calculation to reduce function arguments."""

    file_path: Path
    video_info: dict[str, Any]
    complexity: VideoComplexity
    folder_quality: float | None = None
    force_preset: str | None = None
    force_crf: int | None = None


@dataclass
class QuickTestParams:
    """Parameters for quick test encoding to reduce function arguments."""

    file_path: Path
    test_crf: int
    test_duration: int
    gpu: bool = False
    codec: str = "libx265"
    speed_preset: str = "medium"


def analyze_file(file_path: Path, *, use_cache: bool = True) -> dict[str, Any]:
    """Get video analysis for a single file with caching."""
    if use_cache and file_path in _file_cache:
        return _file_cache[file_path]

    try:
        video_info = FFmpegProbe.get_video_info(file_path)
        complexity = estimate_complexity(file_path, video_info)
        estimated_ssim = estimate_ssim_threshold(video_info, complexity)

        analysis = {
            "duration": video_info["duration"],
            "size": video_info["size"],
            "codec": video_info["codec"] or "unknown",
            "width": video_info.get("width"),
            "height": video_info.get("height"),
            "bitrate": int(video_info["bitrate"]) if video_info["bitrate"] else None,
            "fps": video_info.get("fps"),
            "complexity": complexity,
            "estimated_ssim_threshold": estimated_ssim,
        }

        if use_cache:
            _file_cache[file_path] = analysis
    except (OSError, RuntimeError, ValueError) as e:
        LOG.warning("Failed to analyze %s: %s", file_path, e)
        # Return minimal analysis with defaults
        analysis = {
            "duration": 0.0,
            "size": file_path.stat().st_size if file_path.exists() else 0,
            "codec": "unknown",
            "width": 1920,
            "height": 1080,
            "bitrate": None,
            "fps": 25.0,
            "complexity": VideoComplexity(0.5, 0.5, 0.5, 0.5),
            "estimated_ssim_threshold": 0.92,  # Conservative default
        }
        if use_cache:
            _file_cache[file_path] = analysis

    return analysis


def estimate_complexity(file_path: Path, video_info: dict[str, Any]) -> VideoComplexity:
    """
    OPTIMIZED: Estimate video complexity using ultra-fast 3-frame sampling.

    Reduced from 5 frames to 3 frames for 40% speed improvement.
    """
    try:
        duration = video_info.get("duration", 0)
        if duration <= 0:
            return VideoComplexity(0.5, 0.5, 0.5, 0.5)

        # OPTIMIZED: Extract only 3 strategic frames (was 5)
        sample_times = [
            duration * 0.2,  # 20% mark
            duration * 0.5,  # Middle
            duration * 0.8,  # 80% mark
        ]

        frames = _extract_sample_frames(file_path, sample_times)
        if not frames:
            return VideoComplexity(0.5, 0.5, 0.5, 0.5)

        # Analyze frames for complexity metrics
        motion_scores = []
        detail_scores = []
        grain_scores = []

        for i, frame in enumerate(frames):
            if frame is None:
                continue

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == COLOR_CHANNELS else frame

            # Detail analysis using edge detection
            edges = cv2.Canny(gray, 50, 150)
            detail_score = np.mean(edges) / 255.0

            # Grain analysis using high-frequency content
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            grain_score = min(1.0, float(np.var(laplacian)) / 10000.0)

            # Motion analysis (compare with previous frame if available)
            motion_score = 0.5  # Default for first frame
            if i > 0 and frames[i - 1] is not None:
                prev_frame = frames[i - 1]
                if prev_frame is not None:  # Additional check for mypy
                    prev_gray = (
                        cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                        if len(prev_frame.shape) == COLOR_CHANNELS
                        else prev_frame
                    )
                    diff = cv2.absdiff(gray, prev_gray)
                    motion_score = min(1.0, float(np.mean(diff)) / 255.0 * 5.0)  # Scale up motion sensitivity

            detail_scores.append(detail_score)
            grain_scores.append(grain_score)
            motion_scores.append(motion_score)

        # Calculate average scores
        avg_motion = float(np.mean(motion_scores)) if motion_scores else 0.5
        avg_detail = float(np.mean(detail_scores)) if detail_scores else 0.5
        avg_grain = float(np.mean(grain_scores)) if grain_scores else 0.5

        # Overall complexity is weighted average
        overall = float(avg_motion * 0.4 + avg_detail * 0.4 + avg_grain * 0.2)

        return VideoComplexity(
            motion_score=avg_motion, detail_score=avg_detail, grain_score=avg_grain, overall_complexity=overall
        )

    except (OSError, ValueError, RuntimeError) as e:
        LOG.warning("Failed to estimate complexity for %s: %s", file_path, e)
        return VideoComplexity(0.5, 0.5, 0.5, 0.5)


def _extract_sample_frames(file_path: Path, sample_times: list[float]) -> list[np.ndarray | None]:
    """OPTIMIZED: Extract frames with faster OpenCV operations and smaller scale."""
    frames: list[np.ndarray | None] = []

    # Open once and reuse the capture
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        return [None] * len(sample_times)

    try:
        # Set buffer size to 1 for faster seeking
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        for sample_time in sample_times:
            try:
                # Seek to specific time
                cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
                ret, frame = cap.read()

                if ret and frame is not None:
                    # OPTIMIZED: Resize to even smaller resolution (max 360p for faster analysis)
                    height, width = frame.shape[:2]
                    if height > ANALYSIS_RESOLUTION:
                        scale = ANALYSIS_RESOLUTION / height
                        new_width = int(width * scale)
                        frame = cv2.resize(frame, (new_width, ANALYSIS_RESOLUTION), interpolation=cv2.INTER_LINEAR)

                    frames.append(frame)
                else:
                    frames.append(None)

            except (OSError, ValueError, RuntimeError) as e:
                LOG.debug("Failed to extract frame at %ss from %s: %s", sample_time, file_path, e)
                frames.append(None)

    finally:
        cap.release()

    return frames


def estimate_ssim_threshold(video_info: dict[str, Any], complexity: VideoComplexity) -> float:
    """Estimate appropriate SSIM threshold based on content complexity and quality."""
    try:
        # Base SSIM threshold
        base_threshold = 0.92

        # Adjust based on complexity
        if complexity.overall_complexity > HIGH_COMPLEXITY_THRESHOLD:
            # Very complex content (action scenes) - more permissive
            base_threshold = 0.88
        elif complexity.overall_complexity > MEDIUM_COMPLEXITY_THRESHOLD:
            # Medium complexity - slightly more permissive
            base_threshold = 0.90
        elif complexity.overall_complexity < LOW_COMPLEXITY_THRESHOLD:
            # Simple content (talking heads) - can be more strict
            base_threshold = 0.95

        # Adjust based on resolution
        height = video_info.get("height", 1080)
        if height >= FOUR_K_HEIGHT:  # 4K
            base_threshold -= 0.02  # Slightly more permissive for 4K
        elif height <= HD_HEIGHT:  # HD or lower
            base_threshold += 0.01  # Slightly more strict for lower res

        # Adjust based on grain content
        if complexity.grain_score > HIGH_GRAIN_THRESHOLD:
            base_threshold -= 0.03  # More permissive for grainy content

        # Clamp to reasonable range
        return max(0.85, min(0.98, base_threshold))

    except (OSError, ValueError, RuntimeError):
        return 0.92  # Safe default


def analyze_folder_quality(folder: Path, video_files: list[Path], sample_percentage: float = 0.2) -> float:
    """Analyze folder quality using conservative sampling."""
    if folder in _folder_quality_cache:
        return _folder_quality_cache[folder]

    total_files = len(video_files)
    if total_files == 0:
        return 0.92

    # Conservative sampling strategy similar to audio
    if total_files <= RESERVED_SMALL_FILE_COUNT:
        samples = video_files[:]  # All files
    elif total_files <= RESERVED_MEDIUM_FILE_COUNT:
        samples = video_files[1:-1]  # Skip first/last
    else:
        # Large folder: sample percentage, minimum 3 files
        sample_count = max(3, int(total_files * sample_percentage))
        skip_count = max(1, total_files // 10)  # Skip 10% from edges

        safe_start = skip_count
        safe_end = total_files - skip_count
        safe_range = safe_end - safe_start

        if safe_range <= 0:
            samples = [video_files[total_files // 2]]
        else:
            samples = []
            for i in range(sample_count):
                idx = safe_start + (i * safe_range) // sample_count
                samples.append(video_files[idx])

    # Analyze samples
    complexity_scores = []
    ssim_thresholds = []

    for file_path in samples:
        try:
            analysis = analyze_file(file_path, use_cache=True)
            complexity_scores.append(analysis["complexity"].overall_complexity)
            ssim_thresholds.append(analysis["estimated_ssim_threshold"])
        except (OSError, ValueError, RuntimeError):
            continue

    # Use most conservative (lowest) SSIM threshold from samples
    folder_threshold = min(ssim_thresholds) if ssim_thresholds else 0.92

    LOG.debug(
        "Folder %s SSIM threshold: %.3f (%d samples from %d files)",
        folder.name,
        folder_threshold,
        len(ssim_thresholds),
        total_files,
    )

    _folder_quality_cache[folder] = folder_threshold
    return folder_threshold


def calculate_optimal_crf(  # noqa: C901, PLR0912, PLR0913, PLR0915
    file_path: Path,
    video_info: dict[str, Any],
    complexity: VideoComplexity,
    folder_quality: float | None = None,
    force_preset: str | None = None,
    force_crf: int | None = None,
) -> QualityDecision:
    """
    Calculate optimal CRF value based on content analysis and target SSIM.

    Uses lookup tables and content analysis for fast parameter selection.

    Args:
        file_path: Path to the video file being analyzed
        video_info: Video metadata from FFprobe
        complexity: Video complexity analysis result
        folder_quality: Optional folder-wide quality threshold
        force_preset: Override automatic preset selection
        force_crf: Override automatic CRF calculation

    """
    try:
        # Use folder quality if provided, otherwise calculate from file
        if folder_quality is not None:
            effective_target = folder_quality
        else:
            effective_target = estimate_ssim_threshold(video_info, complexity)

        # Check if source is already heavily compressed
        source_bitrate = video_info.get("bitrate", 0) or 0
        source_bitrate = int(source_bitrate) if source_bitrate else 0

        # Define expected bitrates by resolution for well-compressed content
        height = video_info.get("height", 1080)
        expected_bitrates = {
            480: 1_500_000,  # 1.5 Mbps
            720: 3_000_000,  # 3 Mbps
            1080: 6_000_000,  # 6 Mbps
            1440: 12_000_000,  # 12 Mbps
            2160: 25_000_000,  # 25 Mbps
        }

        # Find expected bitrate for this resolution
        expected_bitrate = 25_000_000  # Default for very high res
        for res, bitrate in sorted(expected_bitrates.items()):
            if height <= res:
                expected_bitrate = bitrate
                break

        # Adjust CRF based on source compression level
        is_heavily_compressed = source_bitrate > 0 and source_bitrate < expected_bitrate * 0.7

        # CRF lookup table based on complexity and target SSIM
        # Preset configs now handle GPU-specific CRF values
        crf_adjustment = 0

        if height >= FOUR_K_HEIGHT:  # 4K
            if is_heavily_compressed:
                crf_lookup = {
                    (0.0, 0.3): (32 + crf_adjustment, 28 + crf_adjustment, 26 + crf_adjustment, 22 + crf_adjustment),
                    (0.3, 0.6): (30 + crf_adjustment, 26 + crf_adjustment, 24 + crf_adjustment, 20 + crf_adjustment),
                    (0.6, 1.0): (28 + crf_adjustment, 24 + crf_adjustment, 22 + crf_adjustment, 18 + crf_adjustment),
                }
            else:
                crf_lookup = {
                    (0.0, 0.3): (28 + crf_adjustment, 24 + crf_adjustment, 22 + crf_adjustment, 18 + crf_adjustment),
                    (0.3, 0.6): (26 + crf_adjustment, 22 + crf_adjustment, 20 + crf_adjustment, 16 + crf_adjustment),
                    (0.6, 1.0): (24 + crf_adjustment, 20 + crf_adjustment, 18 + crf_adjustment, 14 + crf_adjustment),
                }
            preset = "slow"  # 4K benefits from slower preset
        elif height >= TWO_K_HEIGHT:  # 1440p
            if is_heavily_compressed:
                crf_lookup = {
                    (0.0, 0.3): (30 + crf_adjustment, 26 + crf_adjustment, 24 + crf_adjustment, 20 + crf_adjustment),
                    (0.3, 0.6): (28 + crf_adjustment, 24 + crf_adjustment, 22 + crf_adjustment, 18 + crf_adjustment),
                    (0.6, 1.0): (26 + crf_adjustment, 22 + crf_adjustment, 20 + crf_adjustment, 16 + crf_adjustment),
                }
            else:
                crf_lookup = {
                    (0.0, 0.3): (26 + crf_adjustment, 22 + crf_adjustment, 20 + crf_adjustment, 16 + crf_adjustment),
                    (0.3, 0.6): (24 + crf_adjustment, 20 + crf_adjustment, 18 + crf_adjustment, 14 + crf_adjustment),
                    (0.6, 1.0): (22 + crf_adjustment, 18 + crf_adjustment, 16 + crf_adjustment, 12 + crf_adjustment),
                }
            preset = "medium"
        elif height >= FULL_HD_HEIGHT:  # 1080p
            if is_heavily_compressed:
                crf_lookup = {
                    (0.0, 0.3): (29 + crf_adjustment, 25 + crf_adjustment, 23 + crf_adjustment, 19 + crf_adjustment),
                    (0.3, 0.6): (27 + crf_adjustment, 23 + crf_adjustment, 21 + crf_adjustment, 17 + crf_adjustment),
                    (0.6, 1.0): (25 + crf_adjustment, 21 + crf_adjustment, 19 + crf_adjustment, 15 + crf_adjustment),
                }
            else:
                crf_lookup = {
                    (0.0, 0.3): (25 + crf_adjustment, 21 + crf_adjustment, 19 + crf_adjustment, 15 + crf_adjustment),
                    (0.3, 0.6): (23 + crf_adjustment, 19 + crf_adjustment, 17 + crf_adjustment, 13 + crf_adjustment),
                    (0.6, 1.0): (21 + crf_adjustment, 17 + crf_adjustment, 15 + crf_adjustment, 11 + crf_adjustment),
                }
            preset = "medium"
        else:  # 720p and below
            if is_heavily_compressed:
                crf_lookup = {
                    (0.0, 0.3): (28 + crf_adjustment, 24 + crf_adjustment, 22 + crf_adjustment, 18 + crf_adjustment),
                    (0.3, 0.6): (26 + crf_adjustment, 22 + crf_adjustment, 20 + crf_adjustment, 16 + crf_adjustment),
                    (0.6, 1.0): (24 + crf_adjustment, 20 + crf_adjustment, 18 + crf_adjustment, 14 + crf_adjustment),
                }
            else:
                crf_lookup = {
                    (0.0, 0.3): (24 + crf_adjustment, 20 + crf_adjustment, 18 + crf_adjustment, 14 + crf_adjustment),
                    (0.3, 0.6): (22 + crf_adjustment, 18 + crf_adjustment, 16 + crf_adjustment, 12 + crf_adjustment),
                    (0.6, 1.0): (20 + crf_adjustment, 16 + crf_adjustment, 14 + crf_adjustment, 10 + crf_adjustment),
                }
            preset = "fast"  # Lower resolution can use faster preset

        # Check if this is a GPU preset by looking at force_preset
        is_gpu_preset = force_preset and any(
            marker in force_preset.lower() for marker in ["gpu", "nvenc", "amf", "qsv"]
        )

        # Use forced CRF if provided, otherwise calculate
        if force_crf is not None:
            selected_crf = force_crf
        elif is_gpu_preset:
            # For GPU presets, be more conservative with CRF adjustments
            # Start with a higher baseline since GPU encoders are less efficient
            selected_crf = 28  # Conservative GPU default

            # Only make small adjustments for complexity
            complexity_val = complexity.overall_complexity
            if complexity_val > HIGH_COMPLEXITY_THRESHOLD:  # Very high complexity
                selected_crf = max(24, selected_crf - 2)
            elif complexity_val > MEDIUM_COMPLEXITY_THRESHOLD:  # Medium-high complexity
                selected_crf = max(26, selected_crf - 1)
            elif complexity_val < LOW_COMPLEXITY_THRESHOLD:  # Low complexity
                selected_crf = min(32, selected_crf + 2)

            # Minimal adjustment for grain (GPU encoders don't handle grain as well)
            if complexity.grain_score > HIGH_GRAIN_THRESHOLD:
                selected_crf = max(22, selected_crf - 1)
        else:
            # Standard CPU encoder CRF calculation
            complexity_val = complexity.overall_complexity
            selected_crf = 23  # Safe default

            for (min_c, max_c), crfs in crf_lookup.items():
                if min_c <= complexity_val < max_c:
                    # Select CRF based on target SSIM
                    if effective_target >= SSIM_THRESHOLD_VERY_HIGH:
                        selected_crf = crfs[3]
                    elif effective_target >= SSIM_THRESHOLD_HIGH:
                        selected_crf = crfs[2]
                    elif effective_target >= SSIM_THRESHOLD_MEDIUM:
                        selected_crf = crfs[1]
                    else:
                        selected_crf = crfs[0]
                    break

            # Adjust for grain content
            if complexity.grain_score > HIGH_GRAIN_THRESHOLD:
                selected_crf = max(10, selected_crf - 2)  # Lower CRF for grainy content

        # Use forced preset if provided, otherwise use calculated preset
        if force_preset is not None:
            preset = force_preset

        # Determine limitation reason
        limitation_reason = None
        if complexity.overall_complexity > HIGH_COMPLEXITY_THRESHOLD:
            limitation_reason = "High complexity content requires lower CRF"
        elif complexity.grain_score > HIGH_GRAIN_THRESHOLD:
            limitation_reason = "Grainy content requires lower CRF"
        elif height >= FOUR_K_HEIGHT:
            limitation_reason = "4K resolution requires optimized settings"

        return QualityDecision(
            effective_crf=selected_crf,
            effective_preset=preset,
            limitation_reason=limitation_reason,
            predicted_ssim=effective_target,
        )

    except (OSError, ValueError, RuntimeError) as e:
        LOG.warning("Failed to calculate optimal CRF for %s: %s", file_path, e)
        return QualityDecision(
            effective_crf=23,
            effective_preset="medium",
            limitation_reason="Fallback due to analysis error",
            predicted_ssim=0.92,
        )


def validate_quality_fast(original_path: Path, encoded_path: Path, sample_count: int = 5) -> float:
    """
    Fast SSIM validation using strategic frame sampling.

    Args:
        original_path: Original video file
        encoded_path: Encoded video file
        sample_count: Number of frames to sample for comparison

    Returns:
        Average SSIM score

    """
    try:
        # Get video duration
        video_info = FFmpegProbe.get_video_info(original_path)
        duration = video_info.get("duration", 0)

        if duration <= 0:
            return 0.5  # Unknown quality

        # Generate sample times
        sample_times = []
        for i in range(sample_count):
            time_point = duration * (0.1 + 0.8 * i / (sample_count - 1))  # Sample from 10% to 90%
            sample_times.append(time_point)

        # Extract frames from both videos
        original_frames = _extract_sample_frames(original_path, sample_times)
        encoded_frames = _extract_sample_frames(encoded_path, sample_times)

        ssim_scores = []

        for orig, enc in zip(original_frames, encoded_frames, strict=False):
            if orig is None or enc is None:
                continue

            # Ensure same size
            enc_resized = cv2.resize(enc, (orig.shape[1], orig.shape[0])) if orig.shape != enc.shape else enc

            # Convert to grayscale and calculate real SSIM
            orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY) if len(orig.shape) == COLOR_CHANNELS else orig
            enc_gray = (
                cv2.cvtColor(enc_resized, cv2.COLOR_BGR2GRAY)
                if len(enc_resized.shape) == COLOR_CHANNELS
                else enc_resized
            )

            # Normalize to 0-1
            orig_normalized = orig_gray.astype(np.float64) / 255.0
            enc_normalized = enc_gray.astype(np.float64) / 255.0

            # Calculate SSIM with scikit-image
            ssim_score = ssim(orig_normalized, enc_normalized, data_range=1.0)

            ssim_scores.append(ssim_score)

        return float(np.mean(ssim_scores)) if ssim_scores else 0.5

    except (OSError, ValueError, RuntimeError) as e:
        LOG.warning("Failed to validate quality between %s and %s: %s", original_path, encoded_path, e)
        return 0.5


def quick_test_encode(  # noqa: PLR0913, PLR0915, C901, PLR0912
    file_path: Path,
    test_crf: int,
    test_duration: int = 30,
    *,
    gpu: bool = False,
    codec: str = "libx265",
    speed_preset: str = "fast",
) -> tuple[Path | None, float, dict[str, float]]:
    """
    Unified test encode that measures both SSIM quality and encoding speed.

    Args:
        file_path: Source video file
        test_crf: CRF value to test
        test_duration: Duration in seconds to test
        gpu: Whether to use GPU acceleration
        codec: Encoder codec to use
        speed_preset: Encoding speed preset

    Returns:
        Tuple of (temp_file_path, measured_ssim, speed_metrics)
        speed_metrics contains: {"fps": encoding_fps, "processing_time_min": estimated_full_file_time}

    """
    try:
        video_info = FFmpegProbe.get_video_info(file_path)
        duration = video_info.get("duration", 0)

        # OPTIMIZED: Adaptive test duration for speed while maintaining accuracy
        if duration < test_duration:
            test_duration = max(5, int(duration * 0.2))  # Use 20% of video or 5s minimum
        # Further optimize test duration based on video length
        elif duration <= SHORT_VIDEO_DURATION:  # Short videos (<=2 min)
            test_duration = min(15, int(duration * 0.3))  # Up to 30% or 15s
        elif duration <= MEDIUM_VIDEO_DURATION:  # Medium videos (<=10 min)
            test_duration = min(20, int(duration * 0.1))  # Up to 10% or 20s
        else:  # Long videos
            test_duration = min(25, int(duration * 0.05))  # Up to 5% or 25s

        # Extract test segment from middle of video
        start_time = max(0, (duration - test_duration) / 2)

        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_segment:
            segment_path = Path(temp_segment.name)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_encoded:
            encoded_path = Path(temp_encoded.name)

        try:
            # Calculate dynamic timeout based on codec and preset
            # AV1 with slower preset can take much longer
            base_timeout = 60
            if codec == "libaom-av1":
                if speed_preset in ["veryslow", "slower"]:
                    base_timeout = 600  # 10 minutes for very slow AV1
                elif speed_preset in ["slow", "medium"]:
                    base_timeout = 300  # 5 minutes for moderate AV1
                else:
                    base_timeout = 180  # 3 minutes for faster AV1
            elif codec in ["libx265", "libhevc"]:
                # 5 min for slow HEVC, 2 min for normal
                base_timeout = 300 if speed_preset in ["veryslow", "slower", "slow"] else 120
            elif codec in ["libvpx-vp9"]:
                base_timeout = 240  # 4 minutes for VP9
            else:
                base_timeout = 120  # 2 minutes for other codecs

            # Scale timeout based on test duration
            timeout_multiplier = max(1.0, test_duration / 10.0)  # Scale for longer test segments
            dynamic_timeout = int(base_timeout * timeout_multiplier)

            # Extract test segment
            ffmpeg = FFmpegProcessor(timeout=dynamic_timeout)

            # For some codecs (like WMV3), copying to MP4 container fails
            # so we detect incompatible combinations and re-encode if needed
            video_info = FFmpegProbe.get_video_info(file_path)
            source_codec = video_info.get("codec", "").lower()
            needs_reencoding = source_codec in ["wmv3", "wmv2", "wmv1", "vc1"]

            # Check audio compatibility with MP4 container
            try:
                audio_info = FFmpegProbe.get_audio_info(file_path)
                audio_codec = audio_info.get("codec", "").lower()
                # WMV audio codecs that don't work with MP4
                audio_incompatible = audio_codec in ["wmav1", "wmav2", "wma", "wmapro"]
            except (OSError, ValueError, RuntimeError):
                audio_incompatible = True  # Assume incompatible if we can't detect

            if needs_reencoding or audio_incompatible:
                # Re-encode with compatible codecs
                segment_cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(start_time),
                    "-i",
                    str(file_path),
                    "-t",
                    str(test_duration),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",  # High quality for accurate testing
                    "-c:a",
                    "aac" if audio_incompatible else "copy",  # Re-encode audio if incompatible
                    "-b:a",
                    "128k" if audio_incompatible else "",  # Bitrate for re-encoded audio
                    str(segment_path),
                ]
                # Remove empty strings from command
                segment_cmd = [arg for arg in segment_cmd if arg]
            else:
                # Use stream copy for compatible codecs
                segment_cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(start_time),
                    "-i",
                    str(file_path),
                    "-t",
                    str(test_duration),
                    "-c",
                    "copy",
                    str(segment_path),
                ]

            ffmpeg.run_command(segment_cmd, file_path)

            # Encode test segment
            encode_cmd = ffmpeg.build_video_command(
                input_file=segment_path,
                output_file=encoded_path,
                codec=codec,  # Use actual codec from preset
                crf=test_crf,
                preset=speed_preset,  # Use actual speed preset
                gpu=gpu,
            )

            start_time = time.time()
            ffmpeg.run_command(encode_cmd, segment_path)
            encode_time = time.time() - start_time

            # Measure encoding speed
            source_fps = video_info.get("fps", 25.0)
            encoding_fps = (test_duration * source_fps) / encode_time if encode_time > 0 else 0

            # Estimate full file processing time
            full_duration = video_info.get("duration", 0)
            estimated_full_time_min = (full_duration / 60) / (encoding_fps / source_fps) if encoding_fps > 0 else 999

            # Quick SSIM validation using intelligent sampling
            ssim_score = validate_quality_fast(segment_path, encoded_path, sample_count=3)

            # Prepare speed metrics
            speed_metrics = {
                "fps": encoding_fps,
                "processing_time_min": estimated_full_time_min,
                "test_encode_time_sec": encode_time,
                "source_fps": source_fps,
            }

            LOG.debug("Unified test: %.1fs, %.1ffps, SSIM: %.3f", encode_time, encoding_fps, ssim_score)

            # Clean up segment file
            segment_path.unlink(missing_ok=True)

        except Exception:
            # Clean up on error
            segment_path.unlink(missing_ok=True)
            encoded_path.unlink(missing_ok=True)
            raise
        else:
            return encoded_path, ssim_score, speed_metrics

    except (OSError, ValueError, RuntimeError) as e:
        LOG.warning("Quick test encode failed for %s: %s", file_path, e)
        return None, 0.5, {"fps": 25.0, "processing_time_min": 10.0, "test_encode_time_sec": 0, "source_fps": 25.0}
