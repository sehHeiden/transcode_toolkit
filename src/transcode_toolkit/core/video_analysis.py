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

from .ffmpeg import FFmpegProbe, FFmpegProcessor

LOG = logging.getLogger(__name__)

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


def analyze_file(file_path: Path, use_cache: bool = True) -> dict[str, Any]:
    """Get video analysis for a single file with caching."""
    if use_cache and file_path in _file_cache:
        return _file_cache[file_path]

    try:
        video_info = FFmpegProbe.get_video_info(file_path)
        complexity = estimate_complexity(file_path, video_info)
        estimated_ssim = estimate_ssim_threshold(file_path, video_info, complexity)

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

        return analysis

    except Exception as e:
        LOG.warning(f"Failed to analyze {file_path}: {e}")
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
    Estimate video complexity using fast sampling method.

    Uses strategic frame sampling to avoid processing entire video.
    """
    try:
        duration = video_info.get("duration", 0)
        if duration <= 0:
            return VideoComplexity(0.5, 0.5, 0.5, 0.5)

        # Extract 5 strategic frames for analysis
        sample_times = [
            duration * 0.1,  # 10% mark
            duration * 0.3,  # 30% mark
            duration * 0.5,  # Middle
            duration * 0.7,  # 70% mark
            duration * 0.9,  # 90% mark
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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            # Detail analysis using edge detection
            edges = cv2.Canny(gray, 50, 150)
            detail_score = np.mean(edges) / 255.0

            # Grain analysis using high-frequency content
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            grain_score = min(1.0, float(np.var(laplacian)) / 10000.0)

            # Motion analysis (compare with previous frame if available)
            motion_score = 0.5  # Default for first frame
            if i > 0 and frames[i - 1] is not None:
                prev_gray = (
                    cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY) if len(frames[i - 1].shape) == 3 else frames[i - 1]
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

    except Exception as e:
        LOG.warning(f"Failed to estimate complexity for {file_path}: {e}")
        return VideoComplexity(0.5, 0.5, 0.5, 0.5)


def _extract_sample_frames(file_path: Path, sample_times: list[float]) -> list[np.ndarray | None]:
    """Extract specific frames from video at given timestamps."""
    frames: list[np.ndarray | None] = []

    for sample_time in sample_times:
        try:
            # Use OpenCV to extract frame at specific time
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                frames.append(None)
                continue

            # Seek to specific time
            cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
            ret, frame = cap.read()

            if ret and frame is not None:
                # Resize frame for faster processing (max 480p)
                height, width = frame.shape[:2]
                if height > 480:
                    scale = 480 / height
                    new_width = int(width * scale)
                    frame = cv2.resize(frame, (new_width, 480))

                frames.append(frame)
            else:
                frames.append(None)

            cap.release()

        except Exception as e:
            LOG.debug(f"Failed to extract frame at {sample_time}s from {file_path}: {e}")
            frames.append(None)

    return frames


def estimate_ssim_threshold(file_path: Path, video_info: dict[str, Any], complexity: VideoComplexity) -> float:
    """Estimate appropriate SSIM threshold based on content complexity and quality."""
    try:
        # Base SSIM threshold
        base_threshold = 0.92

        # Adjust based on complexity
        if complexity.overall_complexity > 0.8:
            # Very complex content (action scenes) - more permissive
            base_threshold = 0.88
        elif complexity.overall_complexity > 0.6:
            # Medium complexity - slightly more permissive
            base_threshold = 0.90
        elif complexity.overall_complexity < 0.3:
            # Simple content (talking heads) - can be more strict
            base_threshold = 0.95

        # Adjust based on resolution
        height = video_info.get("height", 1080)
        if height >= 2160:  # 4K
            base_threshold -= 0.02  # Slightly more permissive for 4K
        elif height <= 720:  # HD or lower
            base_threshold += 0.01  # Slightly more strict for lower res

        # Adjust based on grain content
        if complexity.grain_score > 0.7:
            base_threshold -= 0.03  # More permissive for grainy content

        # Clamp to reasonable range
        return max(0.85, min(0.98, base_threshold))

    except Exception:
        return 0.92  # Safe default


def analyze_folder_quality(folder: Path, video_files: list[Path], sample_percentage: float = 0.2) -> float:
    """Analyze folder quality using conservative sampling."""
    if folder in _folder_quality_cache:
        return _folder_quality_cache[folder]

    total_files = len(video_files)
    if total_files == 0:
        return 0.92

    # Conservative sampling strategy similar to audio
    if total_files <= 3:
        samples = video_files[:]  # All files
    elif total_files <= 10:
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
        except Exception:
            continue

    # Use most conservative (lowest) SSIM threshold from samples
    folder_threshold = min(ssim_thresholds) if ssim_thresholds else 0.92

    LOG.debug(
        f"Folder {folder.name} SSIM threshold: {folder_threshold:.3f} ({len(ssim_thresholds)} samples from {total_files} files)"
    )

    _folder_quality_cache[folder] = folder_threshold
    return folder_threshold


def calculate_optimal_crf(
    file_path: Path,
    video_info: dict[str, Any],
    complexity: VideoComplexity,
    target_ssim: float = 0.92,
    folder_quality: float | None = None,
) -> QualityDecision:
    """
    Calculate optimal CRF value based on content analysis and target SSIM.

    Uses lookup tables and content analysis for fast parameter selection.
    """
    try:
        # Use folder quality if provided, otherwise calculate from file
        if folder_quality is not None:
            effective_target = folder_quality
        else:
            effective_target = estimate_ssim_threshold(file_path, video_info, complexity)

        # CRF lookup table based on complexity and target SSIM
        # Format: (complexity_range, resolution_category): (ssim_0.85, ssim_0.90, ssim_0.92, ssim_0.95)
        height = video_info.get("height", 1080)

        if height >= 2160:  # 4K
            crf_lookup = {
                (0.0, 0.3): (28, 24, 22, 18),  # Simple content
                (0.3, 0.6): (26, 22, 20, 16),  # Medium content
                (0.6, 1.0): (24, 20, 18, 14),  # Complex content
            }
            preset = "slow"  # 4K benefits from slower preset
        elif height >= 1440:  # 1440p
            crf_lookup = {(0.0, 0.3): (26, 22, 20, 16), (0.3, 0.6): (24, 20, 18, 14), (0.6, 1.0): (22, 18, 16, 12)}
            preset = "medium"
        elif height >= 1080:  # 1080p
            crf_lookup = {(0.0, 0.3): (25, 21, 19, 15), (0.3, 0.6): (23, 19, 17, 13), (0.6, 1.0): (21, 17, 15, 11)}
            preset = "medium"
        else:  # 720p and below
            crf_lookup = {(0.0, 0.3): (24, 20, 18, 14), (0.3, 0.6): (22, 18, 16, 12), (0.6, 1.0): (20, 16, 14, 10)}
            preset = "fast"  # Lower resolution can use faster preset

        # Find appropriate CRF based on complexity
        complexity_val = complexity.overall_complexity
        selected_crf = 23  # Safe default

        for (min_c, max_c), crfs in crf_lookup.items():
            if min_c <= complexity_val < max_c:
                # Select CRF based on target SSIM
                if effective_target >= 0.95:
                    selected_crf = crfs[3]
                elif effective_target >= 0.92:
                    selected_crf = crfs[2]
                elif effective_target >= 0.90:
                    selected_crf = crfs[1]
                else:
                    selected_crf = crfs[0]
                break

        # Adjust for grain content
        if complexity.grain_score > 0.7:
            selected_crf = max(10, selected_crf - 2)  # Lower CRF for grainy content

        # Determine limitation reason
        limitation_reason = None
        if complexity.overall_complexity > 0.8:
            limitation_reason = "High complexity content requires lower CRF"
        elif complexity.grain_score > 0.7:
            limitation_reason = "Grainy content requires lower CRF"
        elif height >= 2160:
            limitation_reason = "4K resolution requires optimized settings"

        return QualityDecision(
            effective_crf=selected_crf,
            effective_preset=preset,
            limitation_reason=limitation_reason,
            predicted_ssim=effective_target,
        )

    except Exception as e:
        LOG.warning(f"Failed to calculate optimal CRF for {file_path}: {e}")
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
            if orig.shape != enc.shape:
                enc = cv2.resize(enc, (orig.shape[1], orig.shape[0]))

            # Convert to grayscale and calculate SSIM
            orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY) if len(orig.shape) == 3 else orig
            enc_gray = cv2.cvtColor(enc, cv2.COLOR_BGR2GRAY) if len(enc.shape) == 3 else enc

            # Calculate SSIM using OpenCV (faster than scikit-image)
            ssim_score = cv2.matchTemplate(orig_gray, enc_gray, cv2.TM_CCOEFF_NORMED)[0][0]
            # Convert to SSIM-like scale (0-1)
            ssim_score = (ssim_score + 1) / 2
            ssim_scores.append(ssim_score)

        return float(np.mean(ssim_scores)) if ssim_scores else 0.5

    except Exception as e:
        LOG.warning(f"Failed to validate quality between {original_path} and {encoded_path}: {e}")
        return 0.5


def quick_test_encode(
    file_path: Path, test_crf: int, test_duration: int = 30, gpu: bool = False
) -> tuple[Path | None, float]:
    """
    Quick test encode of a video segment for quality validation.

    Args:
        file_path: Source video file
        test_crf: CRF value to test
        test_duration: Duration in seconds to test
        gpu: Whether to use GPU acceleration

    Returns:
        Tuple of (temp_file_path, estimated_ssim)

    """
    try:
        video_info = FFmpegProbe.get_video_info(file_path)
        duration = video_info.get("duration", 0)

        if duration < test_duration:
            test_duration = max(10, int(duration * 0.3))  # Use 30% of video or 10s minimum

        # Extract test segment from middle of video
        start_time = max(0, (duration - test_duration) / 2)

        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_segment:
            segment_path = Path(temp_segment.name)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_encoded:
            encoded_path = Path(temp_encoded.name)

        try:
            # Extract test segment
            ffmpeg = FFmpegProcessor(timeout=60)

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
                codec="hevc_nvenc" if gpu else "libx265",
                crf=test_crf,
                preset="fast",  # Use fast preset for test
                gpu=gpu,
            )

            start_time = time.time()
            ffmpeg.run_command(encode_cmd, segment_path)
            encode_time = time.time() - start_time

            # Quick SSIM validation
            ssim_score = validate_quality_fast(segment_path, encoded_path, sample_count=3)

            LOG.debug(f"Test encode completed in {encode_time:.1f}s, SSIM: {ssim_score:.3f}")

            # Clean up segment file
            segment_path.unlink(missing_ok=True)

            return encoded_path, ssim_score

        except Exception:
            # Clean up on error
            segment_path.unlink(missing_ok=True)
            encoded_path.unlink(missing_ok=True)
            raise

    except Exception as e:
        LOG.warning(f"Quick test encode failed for {file_path}: {e}")
        return None, 0.5
