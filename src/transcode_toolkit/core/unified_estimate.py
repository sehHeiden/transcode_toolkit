"""Unified estimation - detailed per-file analysis with savings, SSIM, and speed metrics."""

import csv
import json
import logging
from pathlib import Path
from typing import Any, NamedTuple

from tqdm import tqdm

from ..audio import estimate as audio_estimate
from ..cli.commands.video import VideoCommands
from . import ConfigManager, FFmpegProbe
from .video_analysis import analyze_file, calculate_optimal_crf, quick_test_encode

LOG = logging.getLogger(__name__)

# Constants
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

# File extensions are loaded from config.yaml


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


def analyze_directory(
    directory: Path, *, save_settings: bool = False
) -> tuple[list[FileAnalysis], dict[str, str | None]]:
    """
    Analyze all files in directory with detailed per-file metrics.

    Args:
        directory: Directory to analyze
        save_settings: Whether to save settings to JSON file

    """
    verbose_mode = LOG.isEnabledFor(logging.INFO)

    if verbose_mode:
        LOG.info("🔍 Analyzing %s using detailed transcoding tests...", directory)

    # Get file lists
    video_files, audio_files = _get_media_files(directory)
    video_files_with_audio = _get_video_files_with_audio(video_files)

    if verbose_mode:
        LOG.info("Found %d video files, %d audio files", len(video_files), len(audio_files))

    # Create progress tracking
    progress_bars = _create_progress_bars(
        len(video_files), len(audio_files) + len(video_files_with_audio), verbose_mode=verbose_mode
    )

    # Process all files
    analyses = _process_all_files(
        video_files, video_files_with_audio, audio_files, progress_bars, verbose_mode=verbose_mode
    )

    # Cleanup progress bars
    _cleanup_progress_bars(progress_bars)

    # Determine optimal presets and save if requested
    optimal_presets = _determine_optimal_presets(analyses)

    if save_settings:
        _save_settings(directory, optimal_presets, analyses)

    return analyses, optimal_presets


def _get_media_files(directory: Path) -> tuple[list[Path], list[Path]]:
    """Get video and audio files from directory."""
    config_manager = ConfigManager()
    video_extensions = config_manager.config.video.extensions
    audio_extensions = config_manager.config.audio.extensions

    all_files = list(directory.rglob("*"))
    video_files = [f for f in all_files if f.suffix.lower() in video_extensions]
    audio_files = [f for f in all_files if f.suffix.lower() in audio_extensions]

    return video_files, audio_files


def _get_video_files_with_audio(video_files: list[Path]) -> list[Path]:
    """Find video files that have audio tracks."""
    video_files_with_audio = []
    for video_file in video_files:
        try:
            audio_info = FFmpegProbe.get_audio_info(video_file)
            if audio_info and audio_info.get("duration", 0) > 0:
                video_files_with_audio.append(video_file)
        except (OSError, ValueError, RuntimeError) as e:
            LOG.debug("No audio track found in %s: %s", video_file, e)
    return video_files_with_audio


def _create_progress_bars(total_video: int, total_audio: int, *, verbose_mode: bool) -> dict[str, Any]:
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


def _process_all_files(
    video_files: list[Path],
    video_files_with_audio: list[Path],
    audio_files: list[Path],
    progress_bars: dict[str, Any],
    *,
    verbose_mode: bool,
) -> list[FileAnalysis]:
    """Process all media files and return analyses."""
    analyses = []

    # Process video files
    analyses.extend(_process_video_files(video_files, progress_bars["video"], verbose_mode=verbose_mode))

    # Process audio tracks in video files
    analyses.extend(
        _process_video_audio_files(video_files_with_audio, progress_bars["audio"], verbose_mode=verbose_mode)
    )

    # Process standalone audio files
    analyses.extend(_process_audio_files(audio_files, progress_bars["audio"], verbose_mode=verbose_mode))

    return analyses


def _cleanup_progress_bars(progress_bars: dict[str, Any]) -> None:
    """Close progress bars if they exist."""
    for bar in progress_bars.values():
        if bar:
            bar.close()


def _process_video_files(video_files: list[Path], video_bar: tqdm | None, *, verbose_mode: bool) -> list[FileAnalysis]:
    """Process video files and return analyses."""
    analyses = []
    for video_file in video_files:
        try:
            analysis = _analyze_video_file(video_file, verbose=verbose_mode)
            analyses.append(analysis)
        except (OSError, ValueError, RuntimeError) as e:
            if verbose_mode:
                LOG.warning("Failed to analyze video %s: %s", video_file, e)
        finally:
            if video_bar:
                video_bar.update(1)
    return analyses


def _process_video_audio_files(
    video_files_with_audio: list[Path], audio_bar: tqdm | None, *, verbose_mode: bool
) -> list[FileAnalysis]:
    """Process audio tracks in video files and return analyses."""
    analyses = []
    for video_file in video_files_with_audio:
        try:
            audio_analysis = _analyze_video_audio_track(video_file, verbose=verbose_mode)
            if audio_analysis:
                analyses.append(audio_analysis)
        except (OSError, ValueError, RuntimeError) as e:
            if verbose_mode:
                LOG.warning("Failed to analyze audio track in %s: %s", video_file, e)
        finally:
            if audio_bar:
                audio_bar.update(1)
    return analyses


def _process_audio_files(audio_files: list[Path], audio_bar: tqdm | None, *, verbose_mode: bool) -> list[FileAnalysis]:
    """Process standalone audio files and return analyses."""
    analyses = []
    for audio_file in audio_files:
        try:
            analysis = _analyze_audio_file(audio_file, verbose=verbose_mode)
            analyses.append(analysis)
        except (OSError, ValueError, RuntimeError) as e:
            if verbose_mode:
                LOG.warning("Failed to analyze audio %s: %s", audio_file, e)
        finally:
            if audio_bar:
                audio_bar.update(1)
    return analyses


def _determine_optimal_presets(analyses: list[FileAnalysis]) -> dict[str, str | None]:
    """Determine the best presets for video and audio."""
    video_analyses = [a for a in analyses if a.file_type == "video"]
    audio_analyses = [a for a in analyses if a.file_type == "audio"]

    best_video_preset = _find_best_preset(video_analyses) if video_analyses else None
    best_audio_preset = _find_best_preset(audio_analyses) if audio_analyses else None

    return {"video_preset": best_video_preset, "audio_preset": best_audio_preset}


def _analyze_video_file(video_file: Path, *, verbose: bool = False) -> FileAnalysis:  # noqa: C901, PLR0912, PLR0915
    """Analyze a single video file with actual transcoding measurements."""
    # Get video info
    video_info = FFmpegProbe.get_video_info(video_file)
    current_size_mb = video_info["size"] / (1024 * 1024)

    # Analyze content
    analysis = analyze_file(video_file, use_cache=True)
    complexity = analysis["complexity"]

    if verbose:
        LOG.info("🎬 Testing presets on %s with actual transcoding...", video_file.name)

    # Follow config.yaml: Test ALL available presets as defined in config
    config_manager = ConfigManager()

    # Filter to only working presets (those with available codecs)
    video_commands = VideoCommands(config_manager)
    # NOTE: Accessing private method as needed for preset filtering
    working_presets = video_commands._get_working_presets()  # noqa: SLF001

    # Remove 'default' from the list if it exists (it's usually an alias)
    presets_to_test = [p for p in working_presets if p != "default"]

    if verbose:
        LOG.info("Testing ALL %d available presets from config.yaml", len(presets_to_test))
    preset_results = []

    for preset in presets_to_test:
        try:
            LOG.debug("  Testing preset %s...", preset)

            # Get ACTUAL preset configuration from config
            try:
                config_manager = ConfigManager()
                preset_config = config_manager.config.get_video_preset(preset)

                # Use preset's actual CRF and codec, not calculated values
                actual_crf = preset_config.crf
                actual_codec = preset_config.codec
                actual_preset = preset_config.preset

                LOG.debug("    Using %s: codec=%s, CRF=%d, preset=%s", preset, actual_codec, actual_crf, actual_preset)

            except (ValueError, AttributeError, KeyError) as e:
                LOG.warning("Failed to get preset config for %s: %s", preset, e)
                # Fallback to calculation
                quality_decision = calculate_optimal_crf(
                    file_path=video_file,
                    video_info=video_info,
                    complexity=complexity,
                    folder_quality=None,
                    force_preset=preset,
                )
                actual_crf = quality_decision.effective_crf
                actual_codec = "libx265"  # Fallback
                actual_preset = "medium"

            # UNIFIED MEASUREMENT: Single test that measures both SSIM and speed
            # Use intelligent sampling: shorter test for long videos, longer for short
            video_duration = video_info.get("duration", 30)
            if video_duration <= SHORT_VIDEO_DURATION:  # Short video
                test_duration = min(20, video_duration * 0.5)  # Up to 50% or 20s
            elif video_duration <= MEDIUM_VIDEO_DURATION:  # Medium video (5 min)
                test_duration = min(30, video_duration * 0.2)  # Up to 20% or 30s
            else:  # Long video
                test_duration = min(45, video_duration * 0.1)  # Up to 10% or 45s

            is_gpu = "gpu" in preset or "nvenc" in actual_codec or "amf" in actual_codec or "qsv" in actual_codec

            # Single unified test that measures both quality and speed
            temp_file, measured_ssim, speed_metrics = quick_test_encode(
                video_file,
                actual_crf,  # Use ACTUAL preset CRF, not calculated
                int(test_duration),
                gpu=is_gpu,
                codec=actual_codec,  # Pass actual codec from preset
                speed_preset=actual_preset,  # Pass actual speed preset
            )

            if temp_file and temp_file.exists():
                # ACTUAL SIZE MEASUREMENT from test transcode
                test_size = temp_file.stat().st_size
                test_duration_actual = test_duration

                # Scale up to full file size
                full_duration = video_info.get("duration", 0)
                if test_duration_actual > 0 and full_duration > 0:
                    size_ratio = test_size / (test_duration_actual * 1024 * 1024)  # MB per second
                    estimated_size_mb = size_ratio * full_duration
                else:
                    estimated_size_mb = current_size_mb * 0.7  # Fallback estimate

                # Clean up temp file
                temp_file.unlink()

                # ACTUAL MEASUREMENTS (both from same test)
                actual_ssim = measured_ssim if measured_ssim > 0 else 0.92
                actual_speed_fps = speed_metrics.get("fps", 25.0)
                actual_processing_time = speed_metrics.get("processing_time_min", 10.0)

            else:
                # Fallback to estimation if test encode failed
                estimated_size_mb = _estimate_video_size(video_info, preset, actual_crf)
                actual_ssim = 0.92  # Default SSIM estimate
                # Fallback speed estimation
                speed_info = _estimate_processing_speed(video_info, preset, actual_crf)
                actual_speed_fps = speed_info["fps"]
                actual_processing_time = speed_info["total_minutes"]

            savings_mb = current_size_mb - estimated_size_mb
            savings_percent = (savings_mb / current_size_mb * 100) if current_size_mb > 0 else 0

            preset_results.append(
                {
                    "preset": preset,
                    "estimated_size_mb": estimated_size_mb,
                    "savings_mb": savings_mb,
                    "savings_percent": savings_percent,
                    "predicted_ssim": actual_ssim,  # Use ACTUAL measured SSIM
                    "crf": actual_crf,  # Use ACTUAL preset CRF
                    "estimated_fps": actual_speed_fps,
                    "processing_time_min": actual_processing_time,
                }
            )

            LOG.debug("    ✅ %s: %.1f%% savings, SSIM %.3f", preset, savings_percent, actual_ssim)

        except (OSError, ValueError, RuntimeError) as e:
            LOG.warning("Failed to test preset %s on %s: %s", preset, video_file, e)

    # Find best preset using weighted evaluation: 0.5*SSIM² + 0.4*savings + 0.1*speed
    best_result = None
    if preset_results:

        def score_preset(result: dict[str, Any]) -> float:
            ssim = result.get("predicted_ssim", 0.92)  # Already 0-1 range
            savings = result.get("savings_percent", 0) / 100.0  # Convert % to 0-1 range
            speed = 1.0 - min(1.0, result.get("processing_time_min", 10.0) / 60.0)  # Faster = higher score

            # Weighted score with SSIM squared for emphasis
            score = 0.5 * (ssim**2) + 0.4 * savings + 0.1 * speed
            LOG.debug(
                "    %s: SSIM=%.3f² + savings=%.2f + speed=%.2f = %.3f", result["preset"], ssim, savings, speed, score
            )
            return score

        best_result = max(preset_results, key=score_preset)

        LOG.info(
            "Selected %s with SSIM=%.3f, savings=%.1f%%",
            best_result["preset"],
            best_result["predicted_ssim"],
            best_result["savings_percent"],
        )

    if best_result:
        return FileAnalysis(
            file_path=video_file,
            file_type="video",
            current_size_mb=current_size_mb,
            best_preset=best_result["preset"],
            estimated_size_mb=best_result["estimated_size_mb"],
            savings_mb=best_result["savings_mb"],
            savings_percent=best_result["savings_percent"],
            predicted_ssim=best_result["predicted_ssim"],
            estimated_speed_fps=best_result["estimated_fps"],
            processing_time_min=best_result["processing_time_min"],
            alternatives=preset_results,
        )

    # Fallback if no presets worked
    return FileAnalysis(
        file_path=video_file,
        file_type="video",
        current_size_mb=current_size_mb,
        best_preset="default",
        estimated_size_mb=current_size_mb * 0.7,
        savings_mb=current_size_mb * 0.3,
        savings_percent=30.0,
        predicted_ssim=0.92,
        estimated_speed_fps=25.0,
        processing_time_min=10.0,
        alternatives=[],
    )


def _analyze_video_audio_track(video_file: Path, *, verbose: bool = False) -> FileAnalysis | None:  # noqa: C901, PLR0912
    """Analyze the audio track within a video file using direct audio analysis."""
    try:
        # Check if video file has audio track
        audio_info = FFmpegProbe.get_audio_info(video_file)
        if not audio_info or audio_info.get("duration", 0) <= 0:
            return None  # No audio track

        if verbose:
            LOG.info("🎵 Analyzing audio track in %s...", video_file.name)

        # Calculate current audio size within the video file
        duration = audio_info.get("duration", 0)
        current_bitrate = audio_info.get("bitrate", 128000)
        current_audio_size_bytes = (current_bitrate * duration) / 8
        current_size_mb = current_audio_size_bytes / (1024 * 1024)

        # Get audio codec info
        codec = audio_info.get("codec", "unknown")
        sample_rate = audio_info.get("sample_rate", 44100)
        channels = audio_info.get("channels", 2)

        if verbose:
            LOG.info(
                "    Audio: %s, %dbps, %dHz, %dch, %.1fMB",
                codec,
                current_bitrate,
                sample_rate,
                channels,
                current_size_mb,
            )

        # Use audio estimation logic to find best preset
        best_preset = None
        best_savings_percent = 0

        # Simple heuristic based on current audio characteristics
        if channels == 1:  # Mono audio
            if current_bitrate > HIGH_BITRATE_MONO:  # High bitrate mono
                best_preset = "audiobook"
                best_savings_percent = min(50, (current_bitrate - TARGET_MONO_BITRATE) / current_bitrate * 100)
        elif current_bitrate > HIGH_BITRATE_STEREO:  # High bitrate stereo
            best_preset = "high"
            best_savings_percent = min(40, (current_bitrate - TARGET_STEREO_BITRATE) / current_bitrate * 100)
        elif current_bitrate > MEDIUM_BITRATE:  # Medium bitrate
            best_preset = "music"
            best_savings_percent = min(30, (current_bitrate - TARGET_MEDIUM_BITRATE) / current_bitrate * 100)
        elif current_bitrate > LOW_BITRATE:  # Lower bitrate but could still optimize
            best_preset = "music"
            best_savings_percent = min(20, (current_bitrate - TARGET_LOW_BITRATE) / current_bitrate * 100)

        # Only recommend if we can achieve meaningful savings
        if best_preset and best_savings_percent > MIN_SAVINGS_PERCENT:  # At least 15% savings
            estimated_size_mb = current_size_mb * (1 - best_savings_percent / 100)
            savings_mb = current_size_mb - estimated_size_mb

            if verbose:
                LOG.info("    Recommended: %s preset, %.1f%% savings", best_preset, best_savings_percent)

            return FileAnalysis(
                file_path=video_file,
                file_type="audio",  # Mark as audio analysis from video file
                current_size_mb=current_size_mb,
                best_preset=best_preset,
                estimated_size_mb=estimated_size_mb,
                savings_mb=savings_mb,
                savings_percent=best_savings_percent,
                predicted_ssim=None,  # N/A for audio
                estimated_speed_fps=None,  # N/A for audio
                processing_time_min=None,  # Could estimate, but not critical
                alternatives=[
                    {"preset": "music", "savings_percent": max(0, best_savings_percent - 10)},
                    {"preset": "high", "savings_percent": max(0, best_savings_percent - 5)},
                ],
            )
        if verbose:
            LOG.info("    No significant audio optimization needed (current: %dbps)", current_bitrate)

    except (OSError, ValueError, RuntimeError) as e:
        if verbose:
            LOG.debug("Failed to analyze audio track in %s: %s", video_file, e)

    return None  # No significant audio optimization opportunity


def _analyze_audio_file(audio_file: Path, *, verbose: bool = False) -> FileAnalysis:
    """Analyze a single audio file with available presets."""
    current_size_mb = audio_file.stat().st_size / (1024 * 1024)

    # Use existing audio estimation for this file
    try:
        # Test with different presets
        results = audio_estimate.compare_presets(audio_file.parent)
        file_results = [r for r in results if r.current_size > 0]  # Filter for valid results

        if file_results:
            best_result = max(file_results, key=lambda x: x.saving)
            savings_mb = best_result.saving / (1024 * 1024)

            return FileAnalysis(
                file_path=audio_file,
                file_type="audio",
                current_size_mb=current_size_mb,
                best_preset=best_result.preset,
                estimated_size_mb=current_size_mb - savings_mb,
                savings_mb=savings_mb,
                savings_percent=best_result.saving_percent,
                predicted_ssim=None,
                estimated_speed_fps=None,
                processing_time_min=None,
                alternatives=[{"preset": r.preset, "savings_percent": r.saving_percent} for r in file_results],
            )

    except (OSError, ValueError, RuntimeError) as e:
        if verbose:
            LOG.debug("Audio analysis failed for %s: %s", audio_file, e)

    # Fallback
    return FileAnalysis(
        file_path=audio_file,
        file_type="audio",
        current_size_mb=current_size_mb,
        best_preset="music",
        estimated_size_mb=current_size_mb * 0.5,
        savings_mb=current_size_mb * 0.5,
        savings_percent=50.0,
        predicted_ssim=None,
        estimated_speed_fps=None,
        processing_time_min=None,
        alternatives=[],
    )


def _estimate_video_size(video_info: dict[str, Any], preset: str, crf: int) -> float:
    """Estimate video file size after transcoding."""
    # Simple estimation based on resolution and CRF
    width = video_info.get("width", 1920)
    height = video_info.get("height", 1080)
    duration = video_info.get("duration", 0)

    # Bitrate estimation (kbps) based on resolution and CRF
    pixel_count = width * height

    if "av1" in preset:
        base_bitrate = 1000  # AV1 is more efficient
    elif "h265" in preset or "gpu" in preset:
        base_bitrate = 1500  # H.265
    else:
        base_bitrate = 2000  # H.264 fallback

    # Adjust for resolution
    # Constants for pixel counts
    four_k_pixels = 8294400
    fhd_pixels = 2073600
    hd_pixels = 921600

    if pixel_count >= four_k_pixels:  # 4K
        bitrate_kbps = base_bitrate * 4
    elif pixel_count >= fhd_pixels:  # 1080p
        bitrate_kbps = base_bitrate
    elif pixel_count >= hd_pixels:  # 720p
        bitrate_kbps = int(base_bitrate * 0.6)
    else:
        bitrate_kbps = int(base_bitrate * 0.3)

    # Adjust for CRF (higher CRF = lower bitrate)
    crf_factor = 1.0 - ((crf - 23) * 0.05)  # Rough approximation
    final_bitrate_kbps = bitrate_kbps * max(0.3, crf_factor)

    # Calculate size in MB
    return (final_bitrate_kbps * duration) / (8 * 1024)  # Convert to MB


def _estimate_processing_speed(video_info: dict[str, Any], preset: str, crf: int) -> dict[str, float]:
    """Estimate processing speed and time."""
    width = video_info.get("width", 1920)
    height = video_info.get("height", 1080)
    duration = video_info.get("duration", 0)
    fps = video_info.get("fps", 25.0)

    # Base speeds (fps) for different codecs
    if "gpu" in preset or "nvenc" in preset:
        base_fps = 120.0
    elif "av1" in preset:
        base_fps = 15.0  # AV1 is slower
    elif "h265" in preset:
        base_fps = 25.0
    else:
        base_fps = 60.0  # H.264

    # Resolution factor
    pixel_count = width * height
    four_k_pixels = 8294400
    fhd_pixels = 2073600
    min_crf_quality = 24

    if pixel_count >= four_k_pixels:  # 4K
        res_factor = 0.25
    elif pixel_count >= fhd_pixels:  # 1080p
        res_factor = 1.0
    else:
        res_factor = 2.0

    # Quality factor (lower CRF = slower)
    quality_factor = 1.0 if crf >= min_crf_quality else 0.8

    estimated_fps = base_fps * res_factor * quality_factor
    processing_time_min = (duration / 60) / (estimated_fps / fps) if estimated_fps > 0 else 999

    return {"fps": estimated_fps, "total_minutes": processing_time_min}


def _find_best_preset(analyses: list[FileAnalysis]) -> str | None:
    """Find the preset that gives best average savings across files."""
    if not analyses:
        return None

    # Count savings by preset
    preset_savings: dict[str, list[float]] = {}
    for analysis in analyses:
        preset = analysis.best_preset
        if preset not in preset_savings:
            preset_savings[preset] = []
        preset_savings[preset].append(analysis.savings_percent)

    # Find preset with best average savings
    best_preset = None
    best_avg_savings = 0.0

    for preset, savings_list in preset_savings.items():
        avg_savings = sum(savings_list) / len(savings_list)
        if avg_savings > best_avg_savings:
            best_avg_savings = avg_savings
            best_preset = preset

    return best_preset


def _save_settings(directory: Path, optimal_presets: dict[str, str | None], analyses: list[FileAnalysis]) -> None:
    """Save optimal settings and detailed analysis to JSON file."""
    settings_file = directory / ".transcode_settings.json"

    # Calculate total savings
    total_current_size = sum(a.current_size_mb for a in analyses)
    total_savings = sum(a.savings_mb for a in analyses)

    data = {
        "version": "1.0",
        "analysis_summary": {
            "total_files": len(analyses),
            "video_files": len([a for a in analyses if a.file_type == "video"]),
            "audio_files": len([a for a in analyses if a.file_type == "audio"]),
            "total_current_size_mb": total_current_size,
            "total_potential_savings_mb": total_savings,
            "total_savings_percent": (total_savings / total_current_size * 100) if total_current_size > 0 else 0,
        },
        "optimal_presets": optimal_presets,
        "detailed_analysis": [
            {
                "file": str(a.file_path.name),
                "type": a.file_type,
                "current_size_mb": a.current_size_mb,
                "best_preset": a.best_preset,
                "savings_mb": a.savings_mb,
                "savings_percent": a.savings_percent,
                "predicted_ssim": a.predicted_ssim,
                "processing_time_min": a.processing_time_min,
            }
            for a in analyses
        ],
    }

    with settings_file.open("w") as f:
        json.dump(data, f, indent=2)

    LOG.info("💾 Saved detailed analysis to %s", settings_file)


def print_detailed_summary(  # noqa: C901, PLR0912, PLR0915
    analyses: list[FileAnalysis], optimal_presets: dict[str, str | None], csv_path: str | None = None
) -> None:
    """Print detailed per-file analysis with metrics."""
    if not analyses:
        print("No files found to analyze.")
        return

    # Group by type
    video_analyses = [a for a in analyses if a.file_type == "video"]
    audio_analyses = [a for a in analyses if a.file_type == "audio"]

    print("\n🎯 DETAILED TRANSCODING ANALYSIS")
    print("=" * 80)

    # Video files
    if video_analyses:
        print(f"\n📹 VIDEO FILES ({len(video_analyses)} files):")
        print(f"{'File':<30} {'Best Preset':<15} {'Savings':<10} {'SSIM':<6} {'Est. Time':<10}")
        print("-" * 71)

        for analysis in video_analyses:
            savings_str = f"{analysis.savings_percent:.1f}%"
            ssim_str = f"{analysis.predicted_ssim:.3f}" if analysis.predicted_ssim else "N/A"
            time_str = f"{analysis.processing_time_min:.1f}min" if analysis.processing_time_min else "N/A"

            print(
                f"{analysis.file_path.name[:28]:<30} {analysis.best_preset:<15} "
                f"{savings_str:<10} {ssim_str:<6} {time_str:<10}"
            )

            # Show alternatives sorted by SSIM (best quality first)
            if analysis.alternatives and len(analysis.alternatives) > 1:
                alternatives_sorted = sorted(analysis.alternatives, key=lambda x: x["predicted_ssim"], reverse=True)
                best_ssim = alternatives_sorted[0]

                # Show best SSIM alternative if different from best savings preset
                if best_ssim["preset"] != analysis.best_preset:
                    alt_savings = f"{best_ssim['savings_percent']:.1f}%"
                    alt_ssim = f"{best_ssim['predicted_ssim']:.3f}"
                    print(f"  └─ Best SSIM: {best_ssim['preset']:<13} {alt_savings:<10} {alt_ssim:<6} (higher quality)")

                # Show other notable alternatives
                other_alts = [
                    alt
                    for alt in alternatives_sorted[1:3]
                    if alt["preset"] != analysis.best_preset and alt["preset"] != best_ssim["preset"]
                ]
                for alt in other_alts:
                    alt_savings = f"{alt['savings_percent']:.1f}%"
                    alt_ssim = f"{alt['predicted_ssim']:.3f}"
                    print(f"  └─ Alternative: {alt['preset']:<11} {alt_savings:<10} {alt_ssim:<6}")

    # Audio files
    if audio_analyses:
        print(f"\n🔊 AUDIO FILES ({len(audio_analyses)} files):")
        print(f"{'File':<30} {'Preset':<15} {'Size':<10} {'Est. Size':<10} {'Savings':<10}")
        print("-" * 75)

        for analysis in audio_analyses:
            size_str = f"{analysis.current_size_mb:.1f}MB"
            est_size_str = f"{analysis.estimated_size_mb:.1f}MB"
            savings_str = f"{analysis.savings_percent:.1f}%"

            print(
                f"{analysis.file_path.name[:28]:<30} {analysis.best_preset:<15} "
                f"{size_str:<10} {est_size_str:<10} {savings_str:<10}"
            )

            # Show alternatives for audio files too
            if analysis.alternatives and len(analysis.alternatives) > 1:
                sorted_alternatives = sorted(
                    analysis.alternatives, key=lambda x: x.get("savings_percent", 0), reverse=True
                )[:2]
                for alt in sorted_alternatives:
                    if alt["preset"] != analysis.best_preset:
                        alt_savings = f"{alt['savings_percent']:.1f}%"
                        print(f"  └─ Alternative: {alt['preset']:<11} {alt_savings:<10}")

    # Summary
    total_current_mb = sum(a.current_size_mb for a in analyses)
    total_savings_mb = sum(a.savings_mb for a in analyses)
    total_savings_percent = (total_savings_mb / total_current_mb * 100) if total_current_mb > 0 else 0

    print("\n💾 SUMMARY:")
    print(f"   Total files: {len(analyses)}")
    print(f"   Current size: {total_current_mb:.1f} MB ({total_current_mb / 1024:.2f} GB)")
    print(
        f"   Potential savings: {total_savings_mb:.1f} MB "
        f"({total_savings_mb / 1024:.2f} GB, {total_savings_percent:.1f}%)"
    )

    # Recommendations
    print("\n🚀 RECOMMENDED COMMANDS:")
    if optimal_presets.get("video_preset"):
        print(f"   Video: transcode-toolkit video transcode --preset {optimal_presets['video_preset']} .")
    if optimal_presets.get("audio_preset"):
        print(f"   Audio: transcode-toolkit audio transcode --preset {optimal_presets['audio_preset']} .")

    # Save CSV if requested
    if csv_path:
        _save_csv(analyses, csv_path)
        print(f"\n📊 Detailed results saved to {csv_path}")


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
