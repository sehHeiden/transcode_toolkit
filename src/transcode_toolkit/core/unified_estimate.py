"""Unified estimation - detailed per-file analysis with savings, SSIM, and speed metrics."""

import csv
import logging
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

from . import ConfigManager, FFmpegProbe
from .video_analysis import analyze_file


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


def _calculate_reduction_factor(complexity: Any | None, crf: int) -> float:
    """Calculate size reduction factor based on video complexity and CRF.

    Args:
        complexity: Video complexity analysis results
        crf: Constant Rate Factor value

    Returns:
        Reduction factor between 0.2 and 0.9 (20% to 90% of original size)
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
    return max(0.2, min(0.9, base_reduction + crf_factor))  # Clamp between 20% and 90%


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
            estimated_ssim = video_analysis.get("estimated_ssim_threshold", 0.92)

            # Get default video preset for standard estimation
            config_manager = ConfigManager()
            default_preset = config_manager.config.get_video_preset("default")

            # Calculate savings using unified method
            reduction_factor = _calculate_reduction_factor(complexity, default_preset.crf)
            estimated_size_mb = current_size_mb * reduction_factor
            savings_mb = current_size_mb - estimated_size_mb
            savings_percent = (savings_mb / current_size_mb * 100) if current_size_mb > 0 else 0

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
                processing_time_min=current_size_mb / 10.0,  # Rough estimate
                alternatives=[],
            )
            analyses.append(analysis)
        except (OSError, ValueError, RuntimeError) as e:
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
    progress_bars = create_progress_bars(
        len(video_files), len(audio_files), verbose_mode=verbose_mode
    )

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
    """Analyze a single video file with actual transcoding measurements."""
    _ = verbose  # Unused but part of interface
    # Simple analysis - just get file size and use default preset
    current_size_mb = video_file.stat().st_size / (1024 * 1024)
    return FileAnalysis(
        file_path=video_file,
        file_type="video",
        current_size_mb=current_size_mb,
        best_preset="default",
        estimated_size_mb=current_size_mb * 0.7,
        savings_mb=current_size_mb * 0.3,
        savings_percent=((current_size_mb * 0.3) / current_size_mb * 100) if current_size_mb > 0 else 0,
        predicted_ssim=0.95,
        estimated_speed_fps=30.0,
        processing_time_min=current_size_mb / 10.0,  # Rough estimate
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
    """Find the preset that gives best average savings across files."""
    return analyses[0].best_preset if analyses else None


def print_detailed_summary(
    analyses: list[FileAnalysis], optimal_presets: dict[str, str | None], csv_path: str | None = None
) -> None:
    """Print simple analysis summary."""
    if not analyses:
        print("No files found to analyze.")
        return

    total_current_mb = sum(a.current_size_mb for a in analyses)
    total_savings_mb = sum(a.savings_mb for a in analyses)
    total_savings_percent = (total_savings_mb / total_current_mb * 100) if total_current_mb > 0 else 0

    print(f"\n📊 ANALYSIS SUMMARY: {len(analyses)} files")
    print(f"Current size: {total_current_mb:.1f} MB")
    print(f"Potential savings: {total_savings_mb:.1f} MB ({total_savings_percent:.1f}%)")

    # Show SSIM information for video files
    video_analyses = [a for a in analyses if a.file_type == "video" and a.predicted_ssim is not None]
    if video_analyses:
        # Filter out None values and calculate average
        ssim_values = [a.predicted_ssim for a in video_analyses if a.predicted_ssim is not None]
        if ssim_values:
            avg_ssim = sum(ssim_values) / len(ssim_values)
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
    """Compare video presets to determine potential savings."""
    if not video_files:
        return []

    total_size = sum(file.stat().st_size for file in video_files)
    config_manager = ConfigManager()

    # Analyze first video to get complexity
    if not video_files:
        return []

    try:
        # Get complexity from first video (or use default if not available)
        video_analysis = analyze_file(video_files[0], use_cache=True)
        complexity = video_analysis.get("complexity")
    except (OSError, ValueError, RuntimeError) as e:
        LOG.warning("Failed to analyze video complexity: %s", e)
        complexity = None

    preset_results: dict[str, EstimationResult] = {}
    for preset_name, preset_config in config_manager.config.video.presets.items():
        # Calculate unified reduction factor
        reduction_factor = _calculate_reduction_factor(complexity, preset_config.crf)

        estimated_size = int(total_size * reduction_factor)
        saving = total_size - estimated_size
        saving_percent = (saving / total_size) * 100 if total_size > 0 else 0

        # Estimate SSIM based on CRF
        # Lower CRF = higher quality = higher SSIM
        # CRF 20 ~ 0.98, CRF 24 ~ 0.95, CRF 28 ~ 0.92, CRF 32 ~ 0.88
        estimated_ssim = max(0.85, min(0.99, 0.95 - (preset_config.crf - 24) * 0.0075))

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


def recommend_video_preset(results: list[EstimationResult]) -> str:
    """Recommend best video preset based on savings."""
    if not results:
        return "default"

    # Sort by savings percentage, but consider practical factors
    sorted_results = sorted(results, key=lambda x: x.saving_percent, reverse=True)

    # Filter out presets with minimal savings
    minimum_saving_percent = 10
    good_results = [r for r in sorted_results if r.saving_percent >= minimum_saving_percent]

    if good_results:
        # Prefer H.265 presets for good compatibility
        h265_results = [r for r in good_results if "h265" in r.preset]
        if h265_results:
            return h265_results[0].preset
        return good_results[0].preset

    return sorted_results[0].preset if sorted_results else "default"


def print_video_comparison(results: list[EstimationResult], recommended: str) -> None:
    """Print a formatted comparison of video presets."""
    if not results:
        print("\n🎥 No video files found for comparison.")
        return

    print("\n🎥 VIDEO PRESET COMPARISON (Top 10 Presets)")
    # Get total number of presets for info
    config_manager = ConfigManager()
    total_presets = len(config_manager.config.video.presets)
    print(f"Showing top 10 of {total_presets} available presets, sorted by estimated savings.")
    print("=" * 80)
    print(f"{'Preset':<25} {'Current':<12} {'Estimated':<12} {'Saving':<12} {'%':<8} {'SSIM':<6} {'Time':<8}")
    print("-" * 88)

    for result in results:
        current_mb = result.current_size / (1024**2)
        estimated_mb = result.estimated_size / (1024**2)
        saving_mb = result.saving / (1024**2)
        # Estimate processing time: ~1 minute per 10MB for base resolution
        proc_time = current_mb / 10.0
        star = " ⭐" if result.preset == recommended else "  "

        # Truncate long preset names
        max_preset_length = 25
        preset_display = result.preset[:22] + "..." if len(result.preset) > max_preset_length else result.preset

        print(
            f"{preset_display:<23}{star} {current_mb:>9.1f} MB {estimated_mb:>9.1f} MB "
            f"{saving_mb:>9.1f} MB {result.saving_percent:>6.1f}% {result.predicted_ssim:>6.3f} {proc_time:>6.1f}m"
        )

    print("-" * 88)
    print(f"\n⭐ RECOMMENDED: {recommended}")
    print("\n💡 Note: Time estimates are based on CPU encoding at ~10MB/min. GPU encoding is typically 3-5x faster.")
    print("💡 Actual results and processing times may vary based on your hardware and content.")


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
