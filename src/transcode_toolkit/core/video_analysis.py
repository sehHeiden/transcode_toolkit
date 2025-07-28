"""Shared video analysis functions for estimation and transcoding."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from .ffmpeg import FFmpegError, FFmpegProbe, FFmpegProcessor

LOG = logging.getLogger(__name__)

# Constants for magic values
COLOR_CHANNELS = 3
ANALYSIS_RESOLUTION = 360
RESERVED_SMALL_FILE_COUNT = 3
RESERVED_MEDIUM_FILE_COUNT = 10
SHORT_VIDEO_DURATION = 120
MEDIUM_VIDEO_DURATION = 600

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


def analyze_file(file_path: Path, *, use_cache: bool = True) -> dict[str, Any]:
    """Get video analysis for a single file with caching."""
    if use_cache and file_path in _file_cache:
        return _file_cache[file_path]

    try:
        video_info = FFmpegProbe.get_video_info(file_path)
        complexity = estimate_complexity(file_path, video_info)

        analysis = {
            "duration": video_info["duration"],
            "size": video_info["size"],
            "codec": video_info["codec"] or "unknown",
            "width": video_info.get("width"),
            "height": video_info.get("height"),
            "bitrate": int(video_info["bitrate"]) if video_info["bitrate"] else None,
            "fps": video_info.get("fps"),
            "complexity": complexity,
        }

        if use_cache:
            _file_cache[file_path] = analysis
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired, FFmpegError) as e:
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

    for file_path in samples:
        try:
            analysis = analyze_file(file_path, use_cache=True)
            complexity_scores.append(analysis["complexity"].overall_complexity)
        except (OSError, ValueError, RuntimeError):
            continue

    # Use conservative SSIM threshold based on average complexity
    avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0.5
    # Higher complexity = lower SSIM threshold (more conservative)
    folder_threshold = 0.95 - (avg_complexity * 0.05)  # Range: 0.90-0.95

    LOG.debug(
        "Folder %s SSIM threshold: %.3f (avg complexity: %.3f, %d samples from %d files)",
        folder.name,
        folder_threshold,
        avg_complexity,
        len(complexity_scores),
        total_files,
    )

    _folder_quality_cache[folder] = folder_threshold
    return folder_threshold


def validate_quality_fast(original_path: Path, encoded_path: Path, sample_count: int = 5) -> float:
    """
    Fast SSIM validation using 5-second intervals throughout the video.

    Args:
        original_path: Original video file
        encoded_path: Encoded video file
        sample_count: Number of 5-second intervals to sample

    Returns:
        Average SSIM score

    """
    try:
        # Get video duration
        video_info = FFmpegProbe.get_video_info(original_path)
        duration = video_info.get("duration", 0)

        if duration <= 0:
            return 0.5  # Unknown quality

        # Generate sample times at 5-second intervals
        sample_times = []
        interval = 5.0  # 5 seconds

        # Calculate how many 5-second intervals we can fit
        max_intervals = int(duration // interval)
        if max_intervals == 0:
            # Video is shorter than 5 seconds, sample at the middle
            sample_times = [duration / 2]
        else:
            # Sample evenly distributed 5-second intervals
            actual_count = min(sample_count, max_intervals)
            for i in range(actual_count):
                # Distribute intervals evenly across video duration
                position = (i * duration) / actual_count
                # Round to nearest 5-second mark
                position = round(position / interval) * interval
                sample_times.append(position)

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


def quick_test_encode(  # noqa: PLR0913, PLR0915
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

    Uses optimized fast timeouts for better user experience.

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
            # Ultra-fast timeouts - maximum 10 seconds for any codec/preset combination
            max_timeout = 10  # Hard limit of 10 seconds

            # For very slow codecs, still keep the 10-second limit
            dynamic_timeout = max_timeout

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
