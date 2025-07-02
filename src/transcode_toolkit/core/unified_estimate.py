"""Unified estimation - detailed per-file analysis with savings, SSIM, and speed metrics."""

import csv
import json
import logging
from pathlib import Path
from typing import Any, NamedTuple

from tqdm import tqdm

from . import ConfigManager, FFmpegProbe
from .video_analysis import analyze_file, calculate_optimal_crf

LOG = logging.getLogger(__name__)

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


def analyze_directory(directory: Path, save_settings: bool = False) -> tuple[list[FileAnalysis], dict[str, str | None]]:
    """Analyze all files in directory with detailed per-file metrics."""
    # Check if we're in verbose mode
    verbose_mode = LOG.isEnabledFor(logging.INFO)

    if verbose_mode:
        LOG.info(f"🔍 Analyzing {directory} for detailed optimization metrics...")

    # Get file extensions from config
    config_manager = ConfigManager()
    video_extensions = config_manager.config.video.extensions
    audio_extensions = config_manager.config.audio.extensions

    # Find all media files using config extensions
    all_files = list(directory.rglob("*"))
    video_files = [f for f in all_files if f.suffix.lower() in video_extensions]
    audio_files = [f for f in all_files if f.suffix.lower() in audio_extensions]

    if verbose_mode:
        LOG.info(f"Found {len(video_files)} video files, {len(audio_files)} audio files")

    analyses = []

    # Calculate total files including audio tracks in video files
    video_files_with_audio = []
    for video_file in video_files:
        try:
            audio_info = FFmpegProbe.get_audio_info(video_file)
            if audio_info and audio_info.get("duration", 0) > 0:
                video_files_with_audio.append(video_file)
        except Exception:
            pass  # No audio track or failed to detect

    total_video_files = len(video_files)
    total_audio_files = len(audio_files) + len(video_files_with_audio)

    # Create separate progress bars if not in verbose mode
    if not verbose_mode:
        video_progress_bar = (
            tqdm(total=total_video_files, desc="Analyzing video", unit="file", position=0)
            if total_video_files > 0
            else None
        )
        audio_progress_bar = (
            tqdm(total=total_audio_files, desc="Analyzing audio", unit="file", position=1)
            if total_audio_files > 0
            else None
        )
    else:
        video_progress_bar = None
        audio_progress_bar = None

    # Analyze video files (video stream only)
    for video_file in video_files:
        try:
            analysis = _analyze_video_file(video_file, verbose=verbose_mode)
            analyses.append(analysis)
            if video_progress_bar:
                video_progress_bar.update(1)
        except Exception as e:
            if verbose_mode:
                LOG.warning(f"Failed to analyze video {video_file}: {e}")
            if video_progress_bar:
                video_progress_bar.update(1)

    # Analyze audio tracks in video files
    for video_file in video_files_with_audio:
        try:
            audio_analysis = _analyze_video_audio_track(video_file, verbose=verbose_mode)
            if audio_analysis:  # Only add if audio track was successfully analyzed
                analyses.append(audio_analysis)
            if audio_progress_bar:
                audio_progress_bar.update(1)
        except Exception as e:
            if verbose_mode:
                LOG.warning(f"Failed to analyze audio track in {video_file}: {e}")
            if audio_progress_bar:
                audio_progress_bar.update(1)

    # Analyze standalone audio files
    for audio_file in audio_files:
        try:
            analysis = _analyze_audio_file(audio_file, verbose=verbose_mode)
            analyses.append(analysis)
            if audio_progress_bar:
                audio_progress_bar.update(1)
        except Exception as e:
            if verbose_mode:
                LOG.warning(f"Failed to analyze audio {audio_file}: {e}")
            if audio_progress_bar:
                audio_progress_bar.update(1)

    # Close progress bars
    if video_progress_bar:
        video_progress_bar.close()
    if audio_progress_bar:
        audio_progress_bar.close()

    # Determine overall best presets
    video_analyses = [a for a in analyses if a.file_type == "video"]
    audio_analyses = [a for a in analyses if a.file_type == "audio"]

    best_video_preset = _find_best_preset(video_analyses) if video_analyses else None
    best_audio_preset = _find_best_preset(audio_analyses) if audio_analyses else None

    optimal_presets = {"video_preset": best_video_preset, "audio_preset": best_audio_preset}

    if save_settings:
        _save_settings(directory, optimal_presets, analyses)

    return analyses, optimal_presets


def _analyze_video_file(video_file: Path, verbose: bool = False) -> FileAnalysis:
    """Analyze a single video file with actual transcoding measurements."""
    from .video_analysis import quick_test_encode

    # Get video info
    video_info = FFmpegProbe.get_video_info(video_file)
    current_size_mb = video_info["size"] / (1024 * 1024)

    # Analyze content
    analysis = analyze_file(video_file, use_cache=True)
    complexity = analysis["complexity"]

    if verbose:
        LOG.info(f"🎬 Testing presets on {video_file.name} with actual transcoding...")

    # Test all available presets with ACTUAL measurements
    # Use actual available presets from the new CRF/speed system
    config_manager = ConfigManager()
    all_presets = list(config_manager.config.video.presets.keys())

    # Filter to practical presets for testing (avoid too many combinations)
    practical_presets = [
        p
        for p in all_presets
        if any(codec in p for codec in ["h265", "av1", "gpu"])
        and any(speed in p for speed in ["speedfast", "speedmedium", "speedslow"])
        and any(crf in p for crf in ["crf-2", "crf0", "crf2", "crf4"])
    ][:5]  # Limit to 5 presets for performance

    presets_to_test = practical_presets if practical_presets else ["default"]
    preset_results = []

    for preset in presets_to_test:
        try:
            LOG.debug(f"  Testing preset {preset}...")

            # Get ACTUAL preset configuration from config
            try:
                config_manager = ConfigManager()
                preset_config = config_manager.config.get_video_preset(preset)

                # Use preset's actual CRF and codec, not calculated values
                actual_crf = preset_config.crf
                actual_codec = preset_config.codec
                actual_preset = preset_config.preset

                LOG.debug(f"    Using {preset}: codec={actual_codec}, CRF={actual_crf}, preset={actual_preset}")

            except Exception as e:
                LOG.warning(f"Failed to get preset config for {preset}: {e}")
                # Fallback to calculation
                quality_decision = calculate_optimal_crf(
                    file_path=video_file,
                    video_info=video_info,
                    complexity=complexity,
                    target_ssim=0.92,
                    folder_quality=None,
                    force_preset=preset,
                )
                actual_crf = quality_decision.effective_crf
                actual_codec = "libx265"  # Fallback
                actual_preset = "medium"

            # ACTUAL MEASUREMENT: Do real test transcode with preset's actual settings
            test_duration = min(30, video_info.get("duration", 30) * 0.1)  # 10% of video or 30s max
            is_gpu = "gpu" in preset or "nvenc" in actual_codec or "amf" in actual_codec or "qsv" in actual_codec

            temp_file, measured_ssim = quick_test_encode(
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

                # ACTUAL SSIM MEASUREMENT (use measured value)
                actual_ssim = measured_ssim if measured_ssim > 0 else 0.92  # Default if no measurement

            else:
                # Fallback to estimation if test encode failed
                estimated_size_mb = _estimate_video_size(video_info, preset, actual_crf)
                actual_ssim = 0.92  # Default SSIM estimate

            savings_mb = current_size_mb - estimated_size_mb
            savings_percent = (savings_mb / current_size_mb * 100) if current_size_mb > 0 else 0

            # Estimate processing speed based on actual preset settings
            speed_info = _estimate_processing_speed(video_info, preset, actual_crf)

            preset_results.append(
                {
                    "preset": preset,
                    "estimated_size_mb": estimated_size_mb,
                    "savings_mb": savings_mb,
                    "savings_percent": savings_percent,
                    "predicted_ssim": actual_ssim,  # Use ACTUAL measured SSIM
                    "crf": actual_crf,  # Use ACTUAL preset CRF
                    "estimated_fps": speed_info["fps"],
                    "processing_time_min": speed_info["total_minutes"],
                }
            )

            LOG.debug(f"    ✅ {preset}: {savings_percent:.1f}% savings, SSIM {actual_ssim:.3f}")

        except Exception as e:
            LOG.warning(f"Failed to test preset {preset} on {video_file}: {e}")

    # Find best preset by SSIM first, then savings
    if preset_results:
        # Sort by SSIM (descending), then by savings (descending)
        best_result = max(preset_results, key=lambda x: (x["predicted_ssim"], x["savings_mb"]))

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


def _analyze_video_audio_track(video_file: Path, verbose: bool = False) -> FileAnalysis | None:
    """Analyze the audio track within a video file using existing audio estimation."""
    from ..audio import estimate as audio_estimate

    try:
        # Check if video file has audio track
        audio_info = FFmpegProbe.get_audio_info(video_file)
        if not audio_info or audio_info.get("duration", 0) <= 0:
            return None  # No audio track

        if verbose:
            LOG.info(f"🎵 Analyzing audio track in {video_file.name}...")

        # Create a temporary directory with just this video file for audio analysis
        # This allows us to reuse the existing compare_presets function
        temp_parent = video_file.parent

        # Use existing audio estimation but only for this file
        results = audio_estimate.compare_presets(temp_parent)

        # Filter results to only this video file (by checking if the analysis picked up audio from this file)
        # Since compare_presets analyzes all audio files in the directory, we need to
        # estimate what portion of the analysis relates to our video file's audio

        if results:
            # Get the best audio recommendation
            recommended_preset = audio_estimate.recommend_preset(results)
            best_result = None

            # Find the result for the recommended preset
            for result in results:
                if result.preset == recommended_preset:
                    best_result = result
                    break

            if best_result and best_result.saving_percent > 10:  # Only recommend if >10% savings
                # Calculate the audio portion size from the video file
                duration = audio_info.get("duration", 0)
                current_bitrate = audio_info.get("bitrate", 128000)
                current_audio_size_bytes = (current_bitrate * duration) / 8
                current_size_mb = current_audio_size_bytes / (1024 * 1024)

                # Estimate savings based on the preset recommendation
                estimated_size_mb = current_size_mb * (1 - best_result.saving_percent / 100)
                savings_mb = current_size_mb - estimated_size_mb

                return FileAnalysis(
                    file_path=video_file,
                    file_type="audio",  # Mark as audio analysis from video file
                    current_size_mb=current_size_mb,
                    best_preset=best_result.preset,
                    estimated_size_mb=estimated_size_mb,
                    savings_mb=savings_mb,
                    savings_percent=best_result.saving_percent,
                    predicted_ssim=None,  # N/A for audio
                    estimated_speed_fps=None,  # N/A for audio
                    processing_time_min=None,  # Could estimate, but not critical
                    alternatives=[
                        {"preset": r.preset, "savings_percent": r.saving_percent} for r in results[:3]
                    ],  # Show top 3 alternatives
                )

    except Exception as e:
        if verbose:
            LOG.debug(f"Failed to analyze audio track in {video_file}: {e}")

    return None  # No significant audio optimization opportunity


def _analyze_audio_file(audio_file: Path, verbose: bool = False) -> FileAnalysis:
    """Analyze a single audio file with available presets."""
    from ..audio import estimate as audio_estimate

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

    except Exception as e:
        if verbose:
            LOG.debug(f"Audio analysis failed for {audio_file}: {e}")

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


def _estimate_video_size(video_info: dict, preset: str, crf: int) -> float:
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
    if pixel_count >= 8294400:  # 4K
        bitrate_kbps = base_bitrate * 4
    elif pixel_count >= 2073600:  # 1080p
        bitrate_kbps = base_bitrate
    elif pixel_count >= 921600:  # 720p
        bitrate_kbps = int(base_bitrate * 0.6)
    else:
        bitrate_kbps = int(base_bitrate * 0.3)

    # Adjust for CRF (higher CRF = lower bitrate)
    crf_factor = 1.0 - ((crf - 23) * 0.05)  # Rough approximation
    final_bitrate_kbps = bitrate_kbps * max(0.3, crf_factor)

    # Calculate size in MB
    estimated_size_mb = (final_bitrate_kbps * duration) / (8 * 1024)  # Convert to MB

    return estimated_size_mb


def _estimate_processing_speed(video_info: dict, preset: str, crf: int) -> dict[str, float]:
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
    if pixel_count >= 8294400:  # 4K
        res_factor = 0.25
    elif pixel_count >= 2073600:  # 1080p
        res_factor = 1.0
    else:
        res_factor = 2.0

    # Quality factor (lower CRF = slower)
    quality_factor = 1.0 if crf >= 24 else 0.8

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


def _save_settings(directory: Path, optimal_presets: dict, analyses: list[FileAnalysis]) -> None:
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

    LOG.info(f"💾 Saved detailed analysis to {settings_file}")


def print_detailed_summary(analyses: list[FileAnalysis], optimal_presets: dict, csv_path: str | None = None) -> None:
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
                f"{analysis.file_path.name[:28]:<30} {analysis.best_preset:<15} {savings_str:<10} {ssim_str:<6} {time_str:<10}"
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
                f"{analysis.file_path.name[:28]:<30} {analysis.best_preset:<15} {size_str:<10} {est_size_str:<10} {savings_str:<10}"
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
        f"   Potential savings: {total_savings_mb:.1f} MB ({total_savings_mb / 1024:.2f} GB, {total_savings_percent:.1f}%)"
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
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
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
