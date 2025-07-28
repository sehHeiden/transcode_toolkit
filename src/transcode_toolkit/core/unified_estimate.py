"""Unified estimation - detailed per-file analysis with savings, SSIM, and speed metrics."""

import csv
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from tqdm import tqdm as tqdm_progress
else:
    try:
        from tqdm import tqdm as tqdm_progress
    except ImportError:
        tqdm_progress = Any  # type: ignore[misc,assignment]

from tqdm import tqdm

from . import ConfigManager, FFmpegError, FFmpegProbe, VideoComplexity
from .video_analysis import analyze_file

# Weighted scoring configuration
DEFAULT_WEIGHTS = {
    "quality": 0.4,  # 40% weight for quality (SSIM)
    "speed": 0.3,  # 30% weight for processing speed (inverse of time)
    "savings": 0.3,  # 30% weight for file size savings
}


@dataclass
class PresetScore:
    """Represents a preset with its weighted score components."""

    preset_name: str
    quality_score: float  # 0.0 to 1.0 (higher is better)
    speed_score: float  # 0.0 to 1.0 (higher is better)
    savings_score: float  # 0.0 to 1.0 (higher is better)
    weighted_score: float  # Overall weighted score

    # Original values for reference
    predicted_ssim: float
    processing_time_min: float
    savings_percent: float


class EstimationResult(NamedTuple):
    """Result of preset comparison analysis."""

    preset: str
    current_size: int
    estimated_size: int
    saving: int
    saving_percent: float
    predicted_ssim: float


def ffmpeg_cmd(input_path: Path, output_path: Path, *, crf: int = 23, gpu: bool = False) -> list[str]:
    """
    Generate FFmpeg command for video transcoding.

    Args:
        input_path: Source video file
        output_path: Output video file
        crf: Constant Rate Factor (quality setting, lower = better quality)
        gpu: Whether to use GPU acceleration (NVENC)

    Returns:
        FFmpeg command as list of strings

    """
    cmd = ["ffmpeg", "-i", str(input_path)]

    if gpu:
        # Use NVIDIA hardware encoding
        cmd.extend(
            [
                "-c:v",
                "hevc_nvenc",
                "-preset",
                "fast",
                "-crf",
                str(crf),
            ]
        )
    else:
        # Use software encoding
        cmd.extend(
            [
                "-c:v",
                "libx265",
                "-preset",
                "medium",
                "-crf",
                str(crf),
            ]
        )

    # Copy audio without re-encoding
    cmd.extend(["-c:a", "copy"])

    # Add output file
    cmd.append(str(output_path))

    return cmd


# Estimation constants
MIN_SAVINGS_PERCENT = 15
SHORT_VIDEO_DURATION = 60
MEDIUM_VIDEO_DURATION = 300
HIGH_BITRATE_MONO = 96000
TARGET_MONO_BITRATE = 64000
HIGH_BITRATE_STEREO = 192000
TARGET_STEREO_BITRATE = 192000
MEDIUM_BITRATE = 128000
TARGET_MEDIUM_BITRATE = 128000
LOW_BITRATE = 96000
TARGET_LOW_BITRATE = 96000

# Video complexity thresholds
HIGH_COMPLEXITY_THRESHOLD = 0.7
MEDIUM_COMPLEXITY_THRESHOLD = 0.4

# Audio bitrate thresholds
HIGH_BITRATE_THRESHOLD = 256000
MEDIUM_BITRATE_THRESHOLD = 128000

# Codec efficiency gap thresholds
EFFICIENCY_GAP_MODERATE = 1
EFFICIENCY_GAP_GOOD = 2
EFFICIENCY_GAP_SIGNIFICANT = 3
EFFICIENCY_GAP_MASSIVE = 4


class FileAnalysis(NamedTuple):
    """Detailed analysis result for a single file."""

    file_path: Path
    file_type: str  # "video" or "audio"
    current_size_mb: float
    best_preset: str
    estimated_size_mb: float
    savings_mb: float
    savings_percent: float
    predicted_ssim: float | None  # Only for video
    estimated_speed_fps: float | None  # Only for video
    processing_time_min: float | None  # Only for video
    alternatives: list[dict[str, Any]]  # Alternative presets with their metrics


def get_media_files(directory: Path) -> tuple[list[Path], list[Path]]:
    """Get video and audio files from directory."""
    config_manager = ConfigManager()
    video_extensions = config_manager.config.video.extensions
    audio_extensions = config_manager.config.audio.extensions

    all_files = list(directory.rglob("*"))
    video_files = [f for f in all_files if f.suffix.lower() in video_extensions]
    audio_files = [f for f in all_files if f.suffix.lower() in audio_extensions]

    return video_files, audio_files


def get_video_files_with_audio(video_files: list[Path]) -> list[Path]:
    """Find video files that have audio tracks."""
    video_files_with_audio = []
    for video_file in video_files:
        try:
            audio_info = FFmpegProbe.get_audio_info(video_file)
            if audio_info and audio_info.get("duration", 0) > 0:
                video_files_with_audio.append(video_file)
        except (OSError, ValueError, RuntimeError) as e:
            # Assume video files have audio tracks if we can't determine otherwise
            # This is safer than failing completely
            LOG.debug("Could not determine audio track for %s: %s. Assuming it has audio.", video_file, e)
            video_files_with_audio.append(video_file)
    return video_files_with_audio


def create_progress_bars(total_video: int, total_audio: int, *, verbose_mode: bool) -> dict[str, Any]:
    """Create progress bars for file processing."""
    if verbose_mode:
        return {"video": None, "audio": None}

    video_bar = (
        tqdm(total=total_video, desc="📹 Video analysis", unit="file", position=0, leave=True)
        if total_video > 0
        else None
    )
    audio_bar = (
        tqdm(total=total_audio, desc="🔊 Audio analysis", unit="file", position=1, leave=True)
        if total_audio > 0
        else None
    )

    return {"video": video_bar, "audio": audio_bar}


def cleanup_progress_bars(progress_bars: dict[str, Any]) -> None:
    """Close progress bars if they exist."""
    for bar in progress_bars.values():
        if bar:
            bar.close()


def _calculate_reduction_factor(
    complexity: VideoComplexity | None, crf: int, source_codec: str | None = None, target_codec: str | None = None
) -> float:
    """
    Calculate size reduction factor based on video complexity, CRF, and codec transition.

    Args:
        complexity: Video complexity analysis results
        crf: Constant Rate Factor value
        source_codec: Original video codec (e.g., 'h264', 'hevc')
        target_codec: Target video codec (e.g., 'libx265', 'libaom-av1')

    Returns:
        Reduction factor between 0.1 and 0.95 (10% to 95% of original size)

    """
    # Base reduction based on complexity
    if complexity and complexity.overall_complexity > HIGH_COMPLEXITY_THRESHOLD:
        # High complexity - less compression possible
        base_reduction = 0.85
    elif complexity and complexity.overall_complexity > MEDIUM_COMPLEXITY_THRESHOLD:
        # Medium complexity
        base_reduction = 0.7
    else:
        # Low complexity - more compression possible
        base_reduction = 0.6

    # Adjust for CRF - each point difference from 24 changes reduction by 2%
    crf_factor = (crf - 24) * 0.02
    complexity_reduction = max(0.2, min(0.9, base_reduction + crf_factor))

    # Apply codec-specific adjustment
    codec_factor = _get_codec_reduction_factor(source_codec, target_codec)

    # Combine complexity and codec factors
    # The codec factor represents additional savings from better compression
    final_reduction = complexity_reduction * codec_factor

    return max(0.1, min(0.95, final_reduction))  # Clamp between 10% and 95%


def _get_codec_reduction_factor(source_codec: str | None, target_codec: str | None) -> float:  # noqa: PLR0911
    """
    Calculate additional reduction factor based on codec transition.

    Args:
        source_codec: Source video codec (e.g., 'h264', 'hevc', 'mpeg2')
        target_codec: Target video codec (e.g., 'libx265', 'libaom-av1')

    Returns:
        Codec reduction factor (0.1 to 1.0)

    """
    if not source_codec or not target_codec:
        return 1.0  # No codec-specific adjustment

    # Normalize codec names
    source_codec = source_codec.lower()
    target_codec = target_codec.lower()

    # Map codec names to efficiency categories
    codec_efficiency = {
        # Modern efficient codecs
        "hevc": 4,
        "h265": 4,
        "av1": 5,
        "vp9": 3,
        # Standard codecs
        "h264": 3,
        "avc": 3,
        "vp8": 2,
        # Older/less efficient codecs
        "mpeg4": 2,
        "xvid": 2,
        "divx": 2,
        "mpeg2": 1,
        "mpeg1": 1,
        "wmv3": 1,
        "wmv2": 1,
        "wmv1": 1,
        "vc1": 1,
        # Uncompressed/lossless
        "rawvideo": 0,
        "huffyuv": 0,
        "ffv1": 0,
    }

    # Map target codec names to efficiency
    target_efficiency_map = {
        "libx265": 4,
        "hevc_nvenc": 4,
        "hevc_amf": 4,
        "hevc_qsv": 4,
        "libaom-av1": 5,
        "libsvtav1": 5,
        "librav1e": 5,
        "libvpx-vp9": 3,
        "libx264": 3,
        "h264_nvenc": 3,
        "h264_amf": 3,
        "h264_qsv": 3,
        "libvvenc": 6,  # VVC is even more efficient
    }

    source_efficiency = codec_efficiency.get(source_codec, 2)  # Default to medium efficiency
    target_efficiency = target_efficiency_map.get(target_codec, 3)  # Default to H.264 level

    # Calculate codec factor based on efficiency gap
    efficiency_gap = target_efficiency - source_efficiency

    # Return codec factor based on efficiency gap using constants
    if efficiency_gap <= 0:
        return 1.0  # No additional reduction
    if efficiency_gap == EFFICIENCY_GAP_MODERATE:
        return 0.85  # 15% additional reduction
    if efficiency_gap == EFFICIENCY_GAP_GOOD:
        return 0.65  # 35% additional reduction
    if efficiency_gap == EFFICIENCY_GAP_SIGNIFICANT:
        return 0.45  # 55% additional reduction
    if efficiency_gap >= EFFICIENCY_GAP_MASSIVE:
        return 0.25  # 75% additional reduction
    return 1.0  # Fallback


def _calculate_ssim_for_preset(
    preset_config: object, file_path: Path, complexity: VideoComplexity | None = None
) -> float:
    """
    Calculate expected SSIM for a given preset configuration using actual 5-second test encode.

    Args:
        preset_config: Preset configuration object with CRF value
        file_path: Path to the video file for testing
        complexity: Video complexity information (optional)

    Returns:
        Measured SSIM value from actual test encode between 0.0 and 1.0

    """
    from .ffmpeg import FFmpegError
    from .video_analysis import quick_test_encode

    try:
        # Use the existing quick_test_encode function to get actual SSIM measurement
        crf = getattr(preset_config, "crf", 24)
        codec = getattr(preset_config, "codec", "libx265")
        speed_preset = getattr(preset_config, "preset", "medium")

        # Determine if this is a GPU preset
        gpu = any(marker in codec.lower() for marker in ["nvenc", "amf", "qsv"])

        # Perform 5-second test encode to get actual SSIM (with fast timeouts)
        encoded_path, measured_ssim, speed_metrics = quick_test_encode(
            file_path=file_path,
            test_crf=crf,
            test_duration=5,  # Always use 5 seconds as requested
            gpu=gpu,
            codec=codec,
            speed_preset=speed_preset,
        )

        # Clean up the temporary encoded file
        if encoded_path and encoded_path.exists():
            encoded_path.unlink(missing_ok=True)

        # Return the measured SSIM, ensuring it's within reasonable bounds
        return max(0.70, min(0.99, measured_ssim))

    except FFmpegError as e:
        # Check if this is a timeout error
        if isinstance(e.__cause__, subprocess.TimeoutExpired):
            LOG.warning(
                "Preset '%s' with codec '%s' and speed '%s' timed out during test encode - skipping this preset",
                getattr(preset_config, "name", "unknown"),
                codec,
                speed_preset,
            )
            # Return NaN to indicate this preset should be excluded
            return float("nan")
        # Other FFmpeg errors, log and use fallback
        LOG.warning("FFmpeg error during test encode for %s: %s", file_path, e)
        # Fall through to fallback calculation
    except (OSError, ValueError, RuntimeError) as e:
        LOG.warning("Failed to perform test encode for SSIM measurement on %s: %s", file_path, e)
        # Fall through to fallback calculation

    # Fallback to formula-based calculation only if test encode fails
    crf = getattr(preset_config, "crf", 24)
    base_ssim = max(0.80, min(0.99, 0.98 - (crf - 18) * 0.01))

    # Apply complexity adjustment if available
    if complexity:
        complexity_factor = 1.0 - (complexity.overall_complexity * 0.05)
        base_ssim *= complexity_factor

    return max(0.70, min(0.99, base_ssim))


def _calculate_processing_time(current_mb: float, preset_config: object) -> float:
    """
    Calculate processing time based on codec complexity and preset.

    Args:
        current_mb: Current file size in MB
        preset_config: Preset configuration object

    Returns:
        Estimated processing time in minutes

    """
    # Base processing speed (MB per minute)
    base_speed = 10.0  # Conservative estimate for CPU H.265

    # Codec speed multipliers (relative to H.265)
    codec_multipliers = {
        "libx265": 1.0,  # Baseline
        "hevc_nvenc": 0.2,  # GPU is much faster
        "hevc_amf": 0.25,  # AMD GPU
        "hevc_qsv": 0.3,  # Intel GPU
        "libaom-av1": 3.0,  # AV1 is slower
        "libsvtav1": 2.0,  # SVT-AV1 is faster than libaom
        "librav1e": 4.0,  # Rav1e is slowest
        "libvvenc": 5.0,  # VVC is very slow
    }

    # Speed preset multipliers
    speed_multipliers = {
        "slower": 2.0,
        "slow": 1.5,
        "medium": 1.0,
        "fast": 0.7,
        "faster": 0.5,
    }

    # Get codec from preset config
    codec = getattr(preset_config, "codec", "libx265")
    speed_preset = getattr(preset_config, "preset", "medium")

    # Calculate multiplier
    codec_mult = codec_multipliers.get(codec, 1.0)
    speed_mult = speed_multipliers.get(speed_preset, 1.0)

    # Adjust base speed
    adjusted_speed = base_speed / (codec_mult * speed_mult)

    # Calculate time
    return current_mb / adjusted_speed


def process_video_files(
    video_files: list[Path], progress_bar: tqdm_progress | None, *, verbose_mode: bool
) -> list[FileAnalysis]:
    """Process video files and return analysis results with SSIM estimation."""
    analyses = []

    for video_file in video_files:
        try:
            # Update progress bar with current file being analyzed
            if progress_bar:
                progress_bar.set_description(f"📹 Analyzing: {video_file.name[:30]}...")

            # Get detailed video analysis including complexity and SSIM estimation
            video_analysis = analyze_file(video_file, use_cache=True)
            current_size_mb = video_file.stat().st_size / (1024 * 1024)

            # Get complexity and estimated SSIM
            complexity = video_analysis.get("complexity")
            # Use dynamic SSIM calculation as fallback
            config_manager = ConfigManager()
            default_preset = config_manager.config.get_video_preset("default")
            estimated_ssim = _calculate_ssim_for_preset(default_preset, video_file, complexity)

            # Get source codec information
            source_codec = None
            try:
                video_info = FFmpegProbe.get_video_info(video_file)
                source_codec = video_info.get("codec")
            except (OSError, ValueError, RuntimeError) as e:
                LOG.debug("Failed to get video codec for %s: %s", video_file, e)

            # Get default video preset for standard estimation (reuse existing config_manager)
            default_preset = config_manager.config.get_video_preset("default")

            # Calculate savings using unified method with codec information
            reduction_factor = _calculate_reduction_factor(
                complexity, default_preset.crf, source_codec, default_preset.codec
            )
            estimated_size_mb = current_size_mb * reduction_factor
            savings_mb = current_size_mb - estimated_size_mb
            savings_percent = (savings_mb / current_size_mb * 100) if current_size_mb > 0 else 0

            # Calculate processing time using proper method
            processing_time = _calculate_processing_time(current_size_mb, default_preset)

            # Log analysis details in verbose mode
            if verbose_mode:
                LOG.info(
                    "Analyzed %s: complexity=%.2f, SSIM=%.3f, estimated savings=%.1f%%",
                    video_file.name,
                    complexity.overall_complexity if complexity else 0.5,
                    estimated_ssim,
                    savings_percent,
                )

            analysis = FileAnalysis(
                file_path=video_file,
                file_type="video",
                current_size_mb=current_size_mb,
                best_preset="default",
                estimated_size_mb=estimated_size_mb,
                savings_mb=savings_mb,
                savings_percent=savings_percent,
                predicted_ssim=estimated_ssim,
                estimated_speed_fps=30.0,
                processing_time_min=processing_time,
                alternatives=[],
            )
            analyses.append(analysis)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, FFmpegError) as e:
            LOG.warning("Failed to analyze %s: %s", video_file, e)
        finally:
            if progress_bar:
                progress_bar.update(1)

    return analyses


LOG = logging.getLogger(__name__)


def analyze_directory(directory: Path) -> tuple[list[FileAnalysis], dict[str, str | None]]:
    """Analyze all files in directory with detailed per-file metrics."""
    verbose_mode = LOG.isEnabledFor(logging.INFO)

    if verbose_mode:
        LOG.info("🔍 Analyzing %s using detailed transcoding tests...", directory)

    # Get file lists
    video_files, audio_files = get_media_files(directory)

    if verbose_mode:
        LOG.info("Found %d video files, %d audio files", len(video_files), len(audio_files))

    # Create progress tracking
    progress_bars = create_progress_bars(len(video_files), len(audio_files), verbose_mode=verbose_mode)

    # Process all files
    analyses = _process_all_files(video_files, audio_files, progress_bars, verbose_mode=verbose_mode)

    # Cleanup progress bars
    cleanup_progress_bars(progress_bars)

    # Compare video presets if we have video files
    if video_files:
        video_results = compare_video_presets(video_files)
        optimal_video_preset = recommend_video_preset(video_results)
        print_video_comparison(video_results, optimal_video_preset)
    else:
        optimal_video_preset = None

    # Determine optimal presets
    optimal_presets = _determine_optimal_presets(analyses)
    if optimal_video_preset:
        optimal_presets["video_preset"] = optimal_video_preset

    return analyses, optimal_presets


def _process_all_files(
    video_files: list[Path],
    audio_files: list[Path],
    progress_bars: dict[str, Any],
    *,
    verbose_mode: bool,
) -> list[FileAnalysis]:
    """Process all media files and return analyses."""
    analyses = []

    # Process video files
    analyses.extend(process_video_files(video_files, progress_bars["video"], verbose_mode=verbose_mode))

    # Process audio files (simplified)
    for audio_file in audio_files:
        try:
            analysis = _analyze_audio_file(audio_file)
            analyses.append(analysis)
        except (OSError, ValueError, RuntimeError):
            pass
        finally:
            if progress_bars["audio"]:
                progress_bars["audio"].update(1)

    return analyses


def _determine_optimal_presets(analyses: list[FileAnalysis]) -> dict[str, str | None]:
    """Determine the best presets for video and audio."""
    video_analyses = [a for a in analyses if a.file_type == "video"]
    audio_analyses = [a for a in analyses if a.file_type == "audio"]

    best_video_preset = _find_best_preset(video_analyses) if video_analyses else None
    best_audio_preset = _find_best_preset(audio_analyses) if audio_analyses else None

    return {"video_preset": best_video_preset, "audio_preset": best_audio_preset}


def _analyze_video_file(video_file: Path, *, verbose: bool = False) -> FileAnalysis:
    """Analyze a single video file with unified frame-based analysis."""
    _ = verbose  # Unused but part of interface
    current_size_mb = video_file.stat().st_size / (1024 * 1024)

    # Use unified frame-based analysis
    try:
        video_analysis = analyze_file(video_file, use_cache=True)
        complexity = video_analysis.get("complexity")

        # Use dynamic SSIM calculation as fallback
        config_manager = ConfigManager()
        default_preset = config_manager.config.get_video_preset("default")
        estimated_ssim = _calculate_ssim_for_preset(default_preset, video_file, complexity)

        # Get source codec for better estimation
        source_codec = None
        try:
            video_info = FFmpegProbe.get_video_info(video_file)
            source_codec = video_info.get("codec")
        except (OSError, ValueError, RuntimeError):
            pass

        # Calculate with unified method (reuse existing config_manager)
        reduction_factor = _calculate_reduction_factor(
            complexity, default_preset.crf, source_codec, default_preset.codec
        )
        estimated_size_mb = current_size_mb * reduction_factor
        savings_mb = current_size_mb - estimated_size_mb
        savings_percent = (savings_mb / current_size_mb * 100) if current_size_mb > 0 else 0
        processing_time = _calculate_processing_time(current_size_mb, default_preset)

    except (OSError, ValueError, RuntimeError):
        # Fallback values if analysis fails
        estimated_ssim = 0.92
        estimated_size_mb = current_size_mb * 0.7
        savings_mb = current_size_mb * 0.3
        savings_percent = 30.0
        processing_time = current_size_mb / 10.0

    return FileAnalysis(
        file_path=video_file,
        file_type="video",
        current_size_mb=current_size_mb,
        best_preset="default",
        estimated_size_mb=estimated_size_mb,
        savings_mb=savings_mb,
        savings_percent=savings_percent,
        predicted_ssim=estimated_ssim,
        estimated_speed_fps=30.0,
        processing_time_min=processing_time,
        alternatives=[],
    )


def _analyze_audio_file(audio_file: Path) -> FileAnalysis:
    """Analyze a single audio file with available presets."""
    current_size_mb = audio_file.stat().st_size / (1024 * 1024)

    try:
        # Get audio info using FFmpegProbe
        audio_info = FFmpegProbe.get_audio_info(audio_file)

        # Get codec and bitrate information
        codec = audio_info.get("codec", "unknown")
        bitrate = audio_info.get("bitrate")

        # Estimate size reduction based on codec and bitrate
        if codec in ["flac", "wav", "aiff", "alac"]:
            # Lossless to lossy conversion - significant savings
            reduction_factor = 0.2  # 80% savings
            best_preset = "music"
        elif bitrate and int(bitrate) > HIGH_BITRATE_THRESHOLD:
            # High bitrate lossy - moderate savings
            reduction_factor = 0.5  # 50% savings
            best_preset = "music"
        elif bitrate and int(bitrate) > MEDIUM_BITRATE_THRESHOLD:
            # Medium bitrate - some savings
            reduction_factor = 0.7  # 30% savings
            best_preset = "low"
        else:
            # Already low bitrate - minimal savings
            reduction_factor = 0.9  # 10% savings
            best_preset = "low"

        estimated_size_mb = current_size_mb * reduction_factor
        savings_mb = current_size_mb - estimated_size_mb
        savings_percent = (savings_mb / current_size_mb * 100) if current_size_mb > 0 else 0

    except (OSError, ValueError, RuntimeError) as e:
        LOG.warning("Failed to analyze audio file %s: %s", audio_file, e)
        # Fallback to conservative estimates
        estimated_size_mb = current_size_mb * 0.7
        savings_mb = current_size_mb * 0.3
        savings_percent = 30.0
        best_preset = "music"

    return FileAnalysis(
        file_path=audio_file,
        file_type="audio",
        current_size_mb=current_size_mb,
        best_preset=best_preset,
        estimated_size_mb=estimated_size_mb,
        savings_mb=savings_mb,
        savings_percent=savings_percent,
        predicted_ssim=None,
        estimated_speed_fps=None,
        processing_time_min=None,
        alternatives=[],
    )


def _find_best_preset(analyses: list[FileAnalysis]) -> str | None:
    """Find the preset that gives best weighted score across files."""
    if not analyses:
        return None

    # For video files, we could apply weighted scoring here too
    # For now, we'll use the same simple approach but this could be extended
    # to calculate weighted scores for multiple presets per file type

    # Group by preset and calculate average metrics
    preset_groups: dict[str, list[FileAnalysis]] = {}
    for analysis in analyses:
        preset = analysis.best_preset
        if preset not in preset_groups:
            preset_groups[preset] = []
        preset_groups[preset].append(analysis)

    # Find preset with best average savings (simplified approach)
    best_preset = None
    best_avg_savings = 0.0

    for preset, group_analyses in preset_groups.items():
        avg_savings = sum(a.savings_percent for a in group_analyses) / len(group_analyses)
        if avg_savings > best_avg_savings:
            best_avg_savings = avg_savings
            best_preset = preset

    return best_preset


def print_detailed_summary(  # noqa: C901, PLR0912, PLR0915
    analyses: list[FileAnalysis], optimal_presets: dict[str, str | None], csv_path: str | None = None
) -> None:
    """Print detailed analysis summary using recommended presets."""
    if not analyses:
        print("No files found to analyze.")
        return

    # Recalculate totals using the recommended presets instead of individual file defaults
    video_analyses = [a for a in analyses if a.file_type == "video"]
    audio_analyses = [a for a in analyses if a.file_type == "audio"]

    total_current_mb = sum(a.current_size_mb for a in analyses)

    # Calculate savings using recommended presets
    recommended_video_preset = optimal_presets.get("video_preset")
    total_savings_mb = 0.0
    avg_ssim = 0.0

    if video_analyses and recommended_video_preset:
        # Recalculate video savings using recommended preset
        config_manager = ConfigManager()
        try:
            recommended_preset_config = config_manager.config.get_video_preset(recommended_video_preset)
            video_savings_mb = 0.0
            ssim_values = []

            for analysis in video_analyses:
                # Get video complexity for this file
                try:
                    video_analysis = analyze_file(analysis.file_path, use_cache=True)
                    complexity = video_analysis.get("complexity")
                    estimated_ssim = 0.92
                except (OSError, ValueError, RuntimeError):
                    complexity = None
                    estimated_ssim = 0.92

                # Calculate with recommended preset
                reduction_factor = _calculate_reduction_factor(complexity, recommended_preset_config.crf)
                estimated_size_mb = analysis.current_size_mb * reduction_factor
                file_savings_mb = analysis.current_size_mb - estimated_size_mb
                video_savings_mb += file_savings_mb

                # Calculate SSIM for recommended preset using actual frame analysis
                recommended_ssim = estimated_ssim  # Use the actual frame-based SSIM threshold
                ssim_values.append(recommended_ssim)

            total_savings_mb += video_savings_mb
            if ssim_values:
                avg_ssim = sum(ssim_values) / len(ssim_values)

        except (OSError, ValueError) as e:
            LOG.warning("Failed to calculate with recommended preset: %s", e)
            # Fallback to original calculation
            total_savings_mb += sum(a.savings_mb for a in video_analyses)
            ssim_values = [a.predicted_ssim for a in video_analyses if a.predicted_ssim is not None]
            if ssim_values:
                avg_ssim = sum(ssim_values) / len(ssim_values)
    else:
        # No video files or no recommended preset, use original calculation
        total_savings_mb += sum(a.savings_mb for a in video_analyses)
        ssim_values = [a.predicted_ssim for a in video_analyses if a.predicted_ssim is not None]
        if ssim_values:
            avg_ssim = sum(ssim_values) / len(ssim_values)

    # Add audio savings (these don't change with video preset)
    total_savings_mb += sum(a.savings_mb for a in audio_analyses)

    total_savings_percent = (total_savings_mb / total_current_mb * 100) if total_current_mb > 0 else 0

    print(f"\n📊 ANALYSIS SUMMARY: {len(analyses)} files")
    print(f"Current size: {total_current_mb:.1f} MB")
    print(f"Potential savings: {total_savings_mb:.1f} MB ({total_savings_percent:.1f}%)")

    # Show SSIM information for video files using recommended preset
    if video_analyses and avg_ssim > 0:
        print(f"Average predicted SSIM: {avg_ssim:.3f}")

    if optimal_presets.get("video_preset"):
        print(f"Video preset: {optimal_presets['video_preset']}")
    if optimal_presets.get("audio_preset"):
        print(f"Audio preset: {optimal_presets['audio_preset']}")

    if csv_path:
        _save_csv(analyses, csv_path)


def compare_video_presets(video_files: list[Path]) -> list[EstimationResult]:
    """Compare video presets to determine potential savings and SSIM."""
    if not video_files:
        return []

    total_size = sum(file.stat().st_size for file in video_files)
    config_manager = ConfigManager()

    # Analyze first video to get complexity and source codec
    try:
        # Get complexity from first video (or use default if not available)
        video_analysis = analyze_file(video_files[0], use_cache=True)
        complexity = video_analysis.get("complexity")

        # Get source codec from first video
        video_info = FFmpegProbe.get_video_info(video_files[0])
        source_codec = video_info.get("codec")
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, FFmpegError) as e:
        LOG.warning("Failed to analyze video complexity or codec: %s", e)
        complexity = None
        source_codec = None

    # Filter presets by codec availability
    from .ffmpeg import FFmpegProcessor

    ffmpeg = FFmpegProcessor()

    preset_results: dict[str, EstimationResult] = {}
    for preset_name, preset_config in config_manager.config.video.presets.items():
        # Skip presets with unavailable codecs
        is_available, error_msg = ffmpeg.validate_codec(preset_config.codec)
        if not is_available:
            LOG.debug("Skipping preset '%s': %s", preset_name, error_msg)
            continue

        # Calculate unified reduction factor with source codec information
        reduction_factor = _calculate_reduction_factor(complexity, preset_config.crf, source_codec, preset_config.codec)

        estimated_size = int(total_size * reduction_factor)
        saving = total_size - estimated_size
        saving_percent = (saving / total_size) * 100 if total_size > 0 else 0

        # Use frame-based SSIM estimation from video analysis
        try:
            video_analysis = analyze_file(video_files[0], use_cache=True)
            estimated_ssim = _calculate_ssim_for_preset(preset_config, video_files[0], complexity)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, FFmpegError):
            # Fallback to dynamic SSIM calculation if frame analysis fails
            estimated_ssim = _calculate_ssim_for_preset(preset_config, video_files[0], complexity)

        # Skip presets that timed out (NaN SSIM)
        import math

        if math.isnan(estimated_ssim):
            LOG.debug("Skipping preset '%s' due to timeout during test encode", preset_name)
            continue

        # Group similar presets by codec and quality level
        codec_key = preset_config.codec
        quality_key = f"crf{preset_config.crf}"
        group_key = f"{codec_key}_{quality_key}"

        if group_key not in preset_results or saving_percent > preset_results[group_key].saving_percent:
            preset_results[group_key] = EstimationResult(
                preset=preset_name,
                current_size=total_size,
                estimated_size=estimated_size,
                saving=saving,
                saving_percent=saving_percent,
                predicted_ssim=estimated_ssim,
            )

    # Return top results only to avoid overwhelming output
    all_results = list(preset_results.values())
    sorted_results = sorted(all_results, key=lambda x: x.saving_percent, reverse=True)
    return sorted_results[:10]  # Top 10 presets


def create_custom_weights(quality: float = 0.4, speed: float = 0.3, savings: float = 0.3) -> dict[str, float]:
    """
    Create custom weights for preset scoring.

    Args:
        quality: Weight for quality (SSIM) component (0.0-1.0)
        speed: Weight for processing speed component (0.0-1.0)
        savings: Weight for file size savings component (0.0-1.0)

    Returns:
        Dictionary with normalized weights that sum to 1.0

    """
    total = quality + speed + savings
    if total == 0:
        # Fallback to default weights if all zero
        return DEFAULT_WEIGHTS.copy()

    # Normalize weights to sum to 1.0
    return {
        "quality": quality / total,
        "speed": speed / total,
        "savings": savings / total,
    }


def calculate_weighted_score(
    ssim: float,
    processing_time: float,
    saving_percent: float,
    weights: dict[str, float],
    max_processing_time: float | None = None,
) -> float:
    """
    Calculate the weighted score for a preset.

    Args:
        ssim: SSIM quality score (0.0-1.0, higher is better)
        processing_time: Processing time in minutes (lower is better)
        saving_percent: File size savings percentage (0-100, higher is better)
        weights: Weights dictionary with 'quality', 'speed', 'savings' keys
        max_processing_time: Maximum processing time for normalization (optional)

    Returns:
        Weighted score between 0.0 and 1.0 (higher is better)

    """
    quality_score = ssim  # Assume SSIM is normalized between 0-1

    # Use maximum normalization for speed if max_processing_time is provided
    if max_processing_time is not None and max_processing_time > 0:
        # Speed score: 1.0 for fastest (min time), 0.0 for slowest (max time)
        speed_score = 1.0 - (processing_time / max_processing_time)
        speed_score = max(0.0, speed_score)  # Ensure non-negative
    else:
        # Fallback to inverse formula if no max time provided
        speed_score = 1.0 / (1.0 + processing_time)

    savings_score = saving_percent / 100.0  # Normalize savings percentage

    return weights["quality"] * quality_score + weights["speed"] * speed_score + weights["savings"] * savings_score


def recommend_video_preset(results: list[EstimationResult], weights: dict[str, float] = DEFAULT_WEIGHTS) -> str:
    """Recommend best video preset using weighted scoring system."""
    if not results:
        return "default"

    config_manager = ConfigManager()

    # First pass: calculate all processing times to find the maximum
    processing_times = []
    for result in results:
        preset_config = config_manager.config.get_video_preset(result.preset)
        proc_time_min = _calculate_processing_time(result.current_size / (1024**2), preset_config)
        processing_times.append(proc_time_min)

    max_processing_time = max(processing_times) if processing_times else 1.0

    # Second pass: calculate weighted scores with maximum normalization
    scored_presets = []
    for i, result in enumerate(results):
        proc_time_min = processing_times[i]
        score = calculate_weighted_score(
            result.predicted_ssim, proc_time_min, result.saving_percent, weights, max_processing_time
        )
        # Calculate normalized speed score for display
        speed_score = 1.0 - (proc_time_min / max_processing_time) if max_processing_time > 0 else 1.0
        speed_score = max(0.0, speed_score)

        scored_presets.append(
            PresetScore(
                preset_name=result.preset,
                quality_score=result.predicted_ssim,
                speed_score=speed_score,
                savings_score=result.saving_percent / 100.0,
                weighted_score=score,
                predicted_ssim=result.predicted_ssim,
                processing_time_min=proc_time_min,
                savings_percent=result.saving_percent,
            )
        )

    # Sort by weighted score decreasing
    scored_presets.sort(key=lambda x: x.weighted_score, reverse=True)

    return scored_presets[0].preset_name if scored_presets else "default"


def print_video_comparison(
    results: list[EstimationResult], recommended: str, weights: dict[str, float] = DEFAULT_WEIGHTS
) -> None:
    """Print a formatted comparison of video presets with weighted scoring."""
    if not results:
        print("\n🎥 No video files found for comparison.")
        return

    print("\n🎥 VIDEO PRESET COMPARISON (Top 10 Presets)")
    # Get total number of presets for info
    config_manager = ConfigManager()
    total_presets = len(config_manager.config.video.presets)
    print(f"Showing top 10 of {total_presets} available presets, sorted by weighted score.")
    print(f"Weights: Quality={weights['quality']:.1f}, Speed={weights['speed']:.1f}, Savings={weights['savings']:.1f}")
    print("=" * 95)
    print(
        f"{'Preset':<25} {'Current':<12} {'Estimated':<12} {'Saving':<12} {'%':<8} {'SSIM':<6} {'Time':<8} {'Score':<6}"
    )
    print("-" * 103)

    # First pass: calculate all processing times to find the maximum
    processing_times = []
    for result in results:
        current_mb = result.current_size / (1024**2)
        preset_config = config_manager.config.get_video_preset(result.preset)
        proc_time = _calculate_processing_time(current_mb, preset_config)
        processing_times.append(proc_time)

    max_processing_time = max(processing_times) if processing_times else 1.0

    # Second pass: calculate weighted scores with maximum normalization
    scored_presets = []
    for i, result in enumerate(results):
        proc_time = processing_times[i]
        score = calculate_weighted_score(
            result.predicted_ssim, proc_time, result.saving_percent, weights, max_processing_time
        )
        scored_presets.append((result, score, proc_time))

    # Sort by weighted score for display
    scored_presets.sort(key=lambda x: x[1], reverse=True)

    for result, score, proc_time in scored_presets:
        current_mb = result.current_size / (1024**2)
        estimated_mb = result.estimated_size / (1024**2)
        saving_mb = result.saving / (1024**2)
        star = " ⭐" if result.preset == recommended else "  "

        # Truncate long preset names
        max_preset_length = 25
        preset_display = result.preset[:22] + "..." if len(result.preset) > max_preset_length else result.preset

        print(
            f"{preset_display:<23}{star} {current_mb:>9.1f} MB {estimated_mb:>9.1f} MB "
            f"{saving_mb:>9.1f} MB {result.saving_percent:>6.1f}% {result.predicted_ssim:>6.3f} "
            f"{proc_time:>6.1f}m {score:>6.3f}"
        )

    print("-" * 103)
    print(f"\n⭐ RECOMMENDED: {recommended}")
    print("\n💡 Note: Time estimates are based on CPU encoding at ~10MB/min. GPU encoding is typically 3-5x faster.")
    print("💡 Actual results and processing times may vary based on your hardware and content.")
    print("💡 Score combines Quality (SSIM), Speed (max-normalized), and Savings (%) using weighted average.")


def _save_csv(analyses: list[FileAnalysis], csv_path: str) -> None:
    """Save detailed analysis to CSV file."""
    csv_file = Path(csv_path)
    with csv_file.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "file_name",
            "file_type",
            "current_size_mb",
            "best_preset",
            "estimated_size_mb",
            "savings_mb",
            "savings_percent",
            "predicted_ssim",
            "estimated_speed_fps",
            "processing_time_min",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for analysis in analyses:
            writer.writerow(
                {
                    "file_name": analysis.file_path.name,
                    "file_type": analysis.file_type,
                    "current_size_mb": round(analysis.current_size_mb, 2),
                    "best_preset": analysis.best_preset,
                    "estimated_size_mb": round(analysis.estimated_size_mb, 2),
                    "savings_mb": round(analysis.savings_mb, 2),
                    "savings_percent": round(analysis.savings_percent, 1),
                    "predicted_ssim": round(analysis.predicted_ssim, 3) if analysis.predicted_ssim else None,
                    "estimated_speed_fps": round(analysis.estimated_speed_fps, 1)
                    if analysis.estimated_speed_fps
                    else None,
                    "processing_time_min": round(analysis.processing_time_min, 1)
                    if analysis.processing_time_min
                    else None,
                }
            )
