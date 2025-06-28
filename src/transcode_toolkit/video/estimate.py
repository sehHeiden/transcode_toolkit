"""Video size estimation using SSIM-based quality analysis and intelligent CRF selection."""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..core import FFmpegProbe
from ..core.video_analysis import (
    QualityDecision,
    analyze_file,
    analyze_folder_quality,
    calculate_optimal_crf,
)

LOG = logging.getLogger(__name__)

# Common video extensions
VIDEO_EXTS = {".mkv", ".mp4", ".mov", ".avi", ".wmv", ".flv", ".webm", ".m4v"}


def _estimate_transcoding_speed(video_info: dict[str, Any], codec: str, preset_name: str, crf: int) -> dict[str, float]:
    """
    Estimate transcoding speed in frames per second and total time.

    Args:
        video_info: Video metadata from FFprobe
        codec: Target encoding codec
        preset_name: Encoding preset name
        crf: CRF value

    Returns:
        Dict with 'fps', 'total_minutes', 'speed_factor' keys

    """
    try:
        duration = video_info.get("duration", 0)
        width = video_info.get("width", 1920)
        height = video_info.get("height", 1080)
        source_fps = video_info.get("fps", 25.0)

        # Base processing speed (fps) for different hardware/codec combinations
        # These are rough estimates based on typical hardware performance

        # Resolution factor (pixels per frame)
        pixel_count = width * height
        if pixel_count >= 8294400:  # 4K (3840x2160)
            resolution_factor = 0.25
        elif pixel_count >= 3686400:  # 1440p (2560x1440)
            resolution_factor = 0.5
        elif pixel_count >= 2073600:  # 1080p (1920x1080)
            resolution_factor = 1.0
        elif pixel_count >= 921600:  # 720p (1280x720)
            resolution_factor = 2.0
        else:  # 480p and below
            resolution_factor = 4.0

        # Codec base speeds (approximate fps for 1080p medium preset)
        if codec.lower() in ["h264_nvenc", "hevc_nvenc"]:
            # GPU encoding - much faster
            if "h264" in codec.lower():
                base_fps = 200.0  # NVENC H.264 is very fast
            else:
                base_fps = 120.0  # NVENC H.265 is fast but slower than H.264
        elif codec.lower() in ["h264_amf", "hevc_amf"]:
            # AMD GPU encoding
            base_fps = 150.0 if "h264" in codec.lower() else 90.0
        elif codec.lower() in ["h264_qsv", "hevc_qsv"]:
            # Intel QuickSync
            base_fps = 180.0 if "h264" in codec.lower() else 100.0
        elif codec.lower() == "libx264":
            # CPU H.264 encoding
            base_fps = 60.0
        elif codec.lower() == "libx265":
            # CPU H.265 encoding - much slower
            base_fps = 25.0
        else:
            # Unknown codec - conservative estimate
            base_fps = 30.0

        # Preset speed factors

        # Map preset names to speed factors
        preset_factor = 1.0  # Default
        preset_lower = preset_name.lower()

        if "vfast" in preset_lower or "ultrafast" in preset_lower:
            preset_factor = 2.5
        elif "fast" in preset_lower:
            preset_factor = 1.5
        elif "slow" in preset_lower:
            preset_factor = 0.7
        elif "ultra" in preset_lower or "archive" in preset_lower:
            preset_factor = 0.4
        elif "hq" in preset_lower:
            preset_factor = 0.8
        # GPU presets often use different names
        elif "gpu" in preset_lower:
            if "vfast" in preset_lower:
                preset_factor = 2.0
            elif "fast" in preset_lower:
                preset_factor = 1.3
            elif "hq" in preset_lower:
                preset_factor = 0.8
            else:
                preset_factor = 1.0

        # CRF factor (lower CRF = more processing time)
        if crf <= 15:
            crf_factor = 0.8  # High quality takes longer
        elif crf <= 18:
            crf_factor = 0.9
        elif crf >= 28:
            crf_factor = 1.2  # Low quality is faster
        elif crf >= 26:
            crf_factor = 1.1
        else:
            crf_factor = 1.0

        # Calculate estimated processing fps
        estimated_fps = base_fps * resolution_factor * preset_factor * crf_factor

        # Calculate speed factor (processing_fps / source_fps)
        speed_factor = estimated_fps / max(source_fps, 1.0)

        # Calculate total processing time
        if speed_factor > 0:
            total_minutes = duration / (60 * speed_factor)
        else:
            total_minutes = 999.0  # Very slow

        return {
            "fps": estimated_fps,
            "total_minutes": total_minutes,
            "speed_factor": speed_factor,
        }

    except Exception as e:
        LOG.warning(f"Failed to estimate transcoding speed: {e}")
        return {
            "fps": 30.0,
            "total_minutes": duration / 60 if duration > 0 else 10.0,
            "speed_factor": 1.0,
        }


def _predict_ssim_for_crf(crf: int, complexity: float, height: int, baseline_ssim: float = 0.92) -> float:
    """
    Predict SSIM based on CRF value, content complexity, and resolution.

    This uses empirical models to estimate quality without actual encoding.
    """
    # Base SSIM degradation per CRF point (empirically derived)
    # Higher CRF = lower quality = lower SSIM
    base_degradation_per_crf = 0.015  # About 1.5% SSIM loss per CRF point

    # Resolution factor - higher resolution is more sensitive to CRF changes
    if height >= 2160:  # 4K
        resolution_factor = 1.2
    elif height >= 1440:  # 1440p
        resolution_factor = 1.1
    elif height >= 1080:  # 1080p
        resolution_factor = 1.0
    else:  # 720p and below
        resolution_factor = 0.9

    # Complexity factor - complex content is more sensitive to CRF changes
    complexity_factor = 1.0 + (complexity * 0.5)  # 0-50% increase based on complexity

    # Reference CRF (where baseline_ssim applies)
    reference_crf = 23

    # Calculate SSIM degradation
    crf_difference = crf - reference_crf
    degradation = crf_difference * base_degradation_per_crf * resolution_factor * complexity_factor

    # Calculate predicted SSIM
    predicted_ssim = baseline_ssim - degradation

    # Clamp to reasonable bounds
    return max(0.50, min(0.99, predicted_ssim))


def _estimate_size_by_codec(
    video_info: dict[str, Any],
    quality_decision: QualityDecision,
    target_codec: str,  # The codec we're encoding TO
) -> int:
    """
    Estimate HEVC file size based on content analysis and optimal CRF.

    Args:
        video_info: Video metadata from FFprobe
        quality_decision: Optimal encoding parameters
        compression_efficiency: Expected compression ratio vs original

    Returns:
        Estimated file size in bytes

    """
    try:
        original_size = video_info["size"]
        video_info.get("duration", 0)
        video_info.get("height", 1080)
        codec = video_info.get("codec", "").lower()

        # Codec-specific compression efficiency based on ACTUAL target codec
        # H.265 is typically 30-50% more efficient than H.264 at same quality
        if target_codec.lower() in ("hevc_nvenc", "libx265", "hevc"):
            # Encoding TO H.265/HEVC
            if codec in ("hevc", "h265"):
                # Already HEVC - minimal improvement expected
                base_efficiency = 0.9  # 10% improvement at most
            elif codec in ("h264", "avc"):
                # H.264 to HEVC - good compression gains
                base_efficiency = 0.6  # 40% reduction typical
            elif codec in ("mpeg2", "mpeg4", "xvid", "divx"):
                # Older codecs - excellent compression gains
                base_efficiency = 0.4  # 60% reduction possible
            else:
                # Unknown codec - conservative estimate
                base_efficiency = 0.7
        elif target_codec.lower() in ("h264_nvenc", "libx264", "h264", "h264_amf", "h264_qsv"):
            # Encoding TO H.264 (various encoders)
            if codec in ("hevc", "h265"):
                # HEVC to H.264 - quality regression, may increase size
                base_efficiency = 1.3  # 30% size increase expected
            elif codec in ("h264", "avc"):
                # H.264 to H.264 - mainly CRF/quality changes
                base_efficiency = 0.85  # Modest 15% reduction from better encoding
            elif codec in ("mpeg2", "mpeg4", "xvid", "divx"):
                # Older codecs to H.264 - good compression gains
                base_efficiency = 0.5  # 50% reduction
            else:
                # Unknown codec - conservative estimate
                base_efficiency = 0.8
        else:
            # Unknown target codec - very conservative
            base_efficiency = 0.9

        efficiency = base_efficiency

        # Adjust efficiency based on complexity
        complexity_factor = 1.0
        if hasattr(quality_decision, "predicted_ssim"):
            # More complex content (lower SSIM target) compresses less efficiently
            if quality_decision.predicted_ssim < 0.90:
                complexity_factor = 1.2  # 20% larger due to complexity
            elif quality_decision.predicted_ssim > 0.95:
                complexity_factor = 0.8  # 20% smaller for simple content

        # Adjust efficiency based on CRF (much more granular)
        # Reference: CRF 23 = 1.0, each CRF point = ~25% size change
        reference_crf = 23
        crf_difference = quality_decision.effective_crf - reference_crf

        # Exponential relationship: lower CRF = exponentially larger files
        # Each CRF point represents roughly 25% size change
        crf_factor = (1.25) ** (-crf_difference)

        # Clamp factor to reasonable bounds
        crf_factor = max(0.2, min(4.0, crf_factor))

        # Calculate estimated size
        estimated_size = int(original_size * efficiency * complexity_factor * crf_factor)

        # Sanity check - never estimate larger than original for most cases
        if codec not in ("hevc", "h265") and estimated_size > original_size * 0.95:
            estimated_size = int(original_size * 0.85)  # At least 15% reduction

        return max(estimated_size, int(original_size * 0.1))  # Never smaller than 10% of original

    except Exception as e:
        LOG.warning(f"Failed to estimate size: {e}")
        # Fallback: conservative 30% reduction
        return int(video_info.get("size", 0) * 0.7)


def analyze_single_file(file_path: Path, preset_name: str = "default") -> tuple[Path, int, int, QualityDecision] | None:
    """
    Analyze a single video file for size estimation using specific preset.

    Args:
        file_path: Path to video file
        preset_name: Name of preset to use for estimation

    Returns:
        Tuple of (path, current_size, estimated_size, quality_decision) or None if failed

    """
    try:
        if file_path.suffix.lower() not in VIDEO_EXTS:
            return None

        # Get preset configuration to use correct codec
        from ..config import get_config

        config = get_config()
        preset_config = config.get_video_preset(preset_name)
        target_codec = preset_config.codec

        # Get video information and analyze content
        video_info = FFmpegProbe.get_video_info(file_path)
        analysis = analyze_file(file_path, use_cache=True)
        complexity = analysis["complexity"]

        # Calculate optimal encoding parameters for this specific preset
        quality_decision = calculate_optimal_crf(
            file_path=file_path,
            video_info=video_info,
            complexity=complexity,
            target_ssim=analysis["estimated_ssim_threshold"],
            force_preset=preset_name,  # Use the actual preset
        )

        # CRITICAL FIX: Use the actual target codec from the preset
        current_size = video_info["size"]
        estimated_size = _estimate_size_by_codec(video_info, quality_decision, target_codec)

        LOG.debug(
            f"Estimation for {file_path.name}: preset={preset_name}, codec={target_codec}, "
            f"CRF={quality_decision.effective_crf}, {current_size:,} → {estimated_size:,} bytes"
        )

        return (file_path, current_size, estimated_size, quality_decision)

    except Exception as e:
        LOG.warning(f"Failed to analyze {file_path}: {e}")
        return None


def analyse(root: Path) -> list[tuple[Path, int, int, dict[str, Any]]]:
    """
    Analyze all video files in a directory tree for HEVC transcoding estimation.
    Tests all presets from config and returns comprehensive results.

    Returns:
        List of tuples with one entry per file-preset combination:
        (path, current_size, estimated_size, metadata)

    """
    from ..cli.commands.video import VideoCommands
    from ..config import get_config
    from ..core import ConfigManager

    LOG.info(f"Analyzing video files in: {root}")

    # Get only working presets (filtered for available hardware)
    config_manager = ConfigManager()
    video_commands = VideoCommands(config_manager)
    preset_names = video_commands._get_working_presets()

    config = get_config()
    video_presets = {name: config.video.presets[name] for name in preset_names}

    LOG.info(f"Testing {len(preset_names)} working presets: {preset_names}")

    # Find all video files
    all_files = []
    if root.is_file():
        if root.suffix.lower() in VIDEO_EXTS:
            all_files = [root]
    else:
        all_files = [f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXTS]

    LOG.info(f"Found {len(all_files)} video files to analyze")

    # Group files by directory for folder-level quality analysis
    folder_map: dict[Path, list[Path]] = defaultdict(list)
    for f in all_files:
        folder_map[f.parent].append(f)

    # Pre-calculate folder quality thresholds
    folder_quality_cache: dict[Path, float] = {}
    for folder, files_in_dir in folder_map.items():
        folder_quality_cache[folder] = analyze_folder_quality(folder, files_in_dir)
        LOG.debug(f"Folder {folder.name}: quality threshold {folder_quality_cache[folder]:.3f}")

    # Analyze each file with all presets
    results = []
    failed_count = 0

    for file_path in all_files:
        try:
            # Get video information and analyze content once per file
            video_info = FFmpegProbe.get_video_info(file_path)
            analysis = analyze_file(file_path, use_cache=True)
            complexity = analysis["complexity"]
            folder_quality = folder_quality_cache[file_path.parent]
            current_size = video_info["size"]

            # Test each preset for this file
            for preset_name, preset_config in video_presets.items():
                # Calculate quality decision for this specific preset
                quality_decision = calculate_optimal_crf(
                    file_path=file_path,
                    video_info=video_info,
                    complexity=complexity,
                    target_ssim=folder_quality,
                    folder_quality=folder_quality,
                    force_preset=preset_name,  # Force this specific preset
                    force_crf=preset_config.crf,  # Use preset's CRF
                )

                # Calculate actual predicted SSIM for this CRF
                predicted_ssim = _predict_ssim_for_crf(
                    crf=preset_config.crf,
                    complexity=complexity.overall_complexity,
                    height=video_info.get("height", 1080),
                    baseline_ssim=folder_quality,
                )

                # Create a modified quality decision with this preset's CRF
                preset_quality_decision = QualityDecision(
                    effective_crf=preset_config.crf,
                    effective_preset=preset_name,
                    limitation_reason=quality_decision.limitation_reason,
                    predicted_ssim=predicted_ssim,
                )

                # Estimate size for this preset's specific codec and CRF
                estimated_size = _estimate_size_by_codec(video_info, preset_quality_decision, preset_config.codec)

                LOG.debug(
                    f"Preset {preset_name} estimation for {file_path.name}: "
                    f"codec={preset_config.codec}, CRF={preset_config.crf}, "
                    f"{current_size:,} → {estimated_size:,} bytes "
                    f"({((current_size - estimated_size) / current_size * 100):.1f}% savings)"
                )

                # Estimate transcoding speed for this preset
                speed_info = _estimate_transcoding_speed(
                    video_info, preset_config.codec, preset_name, preset_config.crf
                )

                # Prepare metadata for this preset
                metadata = {
                    "crf": preset_config.crf,
                    "preset": preset_name,
                    "predicted_ssim": predicted_ssim,  # Use our calculated SSIM
                    "complexity": complexity.overall_complexity,
                    "resolution": f"{video_info.get('width', 0)}x{video_info.get('height', 0)}",
                    "codec": video_info.get("codec", "unknown"),
                    "target_codec": preset_config.codec,
                    "duration": video_info.get("duration", 0),
                    "limitation_reason": quality_decision.limitation_reason,
                    "preset_description": preset_config.description,
                    "speed_fps": speed_info["fps"],
                    "speed_minutes": speed_info["total_minutes"],
                    "speed_factor": speed_info["speed_factor"],
                }

                results.append((file_path, current_size, estimated_size, metadata))

        except Exception as e:
            LOG.warning(f"Failed to analyze {file_path}: {e}")
            failed_count += 1
            continue

    if failed_count > 0:
        LOG.warning(f"Failed to analyze {failed_count} files")

    LOG.info(f"Successfully analyzed {len(all_files)} video files with {len(preset_names)} presets each")
    return results


def print_summary(rows: list[tuple[Path, int, int, dict[str, Any]]], *, csv_path: str | None = None) -> None:
    """Print estimation summary and optionally save to CSV."""
    if not rows:
        return

    # Group results by file to find best preset for each file
    file_results: dict[Path, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    for file_path, current, estimated, metadata in rows:
        file_results[file_path].append((current, estimated, metadata))

    # Calculate totals using ONLY the best preset for each file
    def calculate_preset_score(current: int, estimated: int, metadata: dict[str, Any]) -> float:
        """Calculate combined score considering savings, SSIM quality, and encoding speed."""
        savings = current - estimated
        savings_percent = (savings / current * 100) if current > 0 else 0
        ssim = metadata.get("predicted_ssim", 0)
        speed_minutes = metadata.get("speed_minutes", 60)  # Default to 1 hour if unknown

        # Normalize savings percentage (0-100) to 0-1 scale
        savings_score = min(savings_percent / 50.0, 1.0)  # Cap at 50% savings = perfect score

        # SSIM is already 0-1 scale
        ssim_score = ssim

        # Speed score: faster = better (inverse relationship)
        # Normalize speed: 0-60 minutes = 1.0 to 0.0 score
        # Cap at 120 minutes (anything slower gets 0 speed score)
        speed_score = max(0.0, 1.0 - (speed_minutes / 120.0))

        # Weight: 30% savings, 60% quality, 10% speed
        return (0.30 * savings_score) + (0.60 * ssim_score) + (0.10 * speed_score)

    current_total = 0
    estimated_total = 0

    # For each file, find the best preset and use its estimates for totals
    for file_path, file_presets in file_results.items():
        if not file_presets:
            continue

        # Find the best preset for this file
        best_preset = max(file_presets, key=lambda x: calculate_preset_score(x[0], x[1], x[2]))

        current_total += best_preset[0]  # current size
        estimated_total += best_preset[1]  # estimated size

    saving_total = current_total - estimated_total
    saving_percent = (saving_total / current_total * 100) if current_total > 0 else 0

    # Print overall summary

    # We already have file_results from above, no need to recreate it

    # Print detailed per-file preset comparison

    for file_path, file_presets in file_results.items():
        file_path.name[:70] + "..." if len(file_path.name) > 73 else file_path.name

        # Find best preset for this file using combined savings, SSIM, and speed score
        def calculate_preset_score(current: int, estimated: int, metadata: dict[str, Any]) -> float:
            """Calculate combined score considering savings, SSIM quality, and encoding speed."""
            savings = current - estimated
            savings_percent = (savings / current * 100) if current > 0 else 0
            ssim = metadata.get("predicted_ssim", 0)
            speed_minutes = metadata.get("speed_minutes", 60)  # Default to 1 hour if unknown

            # Normalize savings percentage (0-100) to 0-1 scale
            savings_score = min(savings_percent / 50.0, 1.0)  # Cap at 50% savings = perfect score

            # SSIM is already 0-1 scale
            ssim_score = ssim

            # Speed score: faster = better (inverse relationship)
            # Normalize speed: 0-60 minutes = 1.0 to 0.0 score
            # Cap at 120 minutes (anything slower gets 0 speed score)
            speed_score = max(0.0, 1.0 - (speed_minutes / 120.0))

            # Weight: 30% savings, 60% quality, 10% speed
            return (0.30 * savings_score) + (0.60 * ssim_score) + (0.10 * speed_score)

        best_score = max(
            calculate_preset_score(current, estimated, metadata) for current, estimated, metadata in file_presets
        )

        # Filter out redundant presets with same SSIM but worse savings/speed
        def filter_redundant_presets(presets):
            """Keep only the best preset for each SSIM level."""
            ssim_groups = {}

            # Group presets by SSIM (rounded to 3 decimal places)
            for current, estimated, metadata in presets:
                ssim = round(metadata.get("predicted_ssim", 0), 3)
                if ssim not in ssim_groups:
                    ssim_groups[ssim] = []
                ssim_groups[ssim].append((current, estimated, metadata))

            filtered_presets = []

            # For each SSIM group, keep only the best preset
            for ssim, group_presets in ssim_groups.items():
                if len(group_presets) == 1:
                    # Only one preset at this SSIM level
                    filtered_presets.extend(group_presets)
                else:
                    # Multiple presets with same SSIM - pick the best one
                    def score_preset(preset_data):
                        current, estimated, metadata = preset_data
                        savings_percent = ((current - estimated) / current * 100) if current > 0 else 0
                        speed_minutes = metadata.get("speed_minutes", 60)

                        # Score: 70% savings, 30% speed (since quality is identical)
                        savings_score = min(savings_percent / 50.0, 1.0)  # Normalize to 0-1
                        speed_score = max(0.0, 1.0 - (speed_minutes / 120.0))  # Faster = better

                        return 0.7 * savings_score + 0.3 * speed_score

                    # Keep the preset with the best savings+speed score
                    best_preset = max(group_presets, key=score_preset)
                    filtered_presets.append(best_preset)

            return filtered_presets

        # Filter redundant presets
        filtered_presets = filter_redundant_presets(file_presets)

        # Sort filtered presets by SSIM quality (highest first)
        sorted_presets = sorted(filtered_presets, key=lambda x: x[2].get("predicted_ssim", 0), reverse=True)

        for current, estimated, metadata in sorted_presets:
            preset = metadata.get("preset", "unknown")
            metadata.get("crf", 0)
            metadata.get("predicted_ssim", 0)
            savings = current - estimated
            (savings / current * 100) if current > 0 else 0

            current / (1024**2)
            estimated / (1024**2)
            savings / (1024**2)

            preset_score = calculate_preset_score(current, estimated, metadata)
            "⭐ YES" if abs(preset_score - best_score) < 0.001 else ""

            # Format speed information
            speed_mins = metadata.get("speed_minutes", 0)
            if speed_mins < 1:
                f"{speed_mins * 60:.0f}s"
            elif speed_mins < 60:
                pass
            else:
                int(speed_mins // 60)
                int(speed_mins % 60)

            metadata.get("target_codec", "unknown")


    # Analyze presets and find the best one
    preset_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "current": 0, "estimated": 0, "savings": 0}
    )

    for _, current, estimated, metadata in rows:
        preset = metadata.get("preset", "unknown")
        saving = current - estimated
        preset_stats[preset]["count"] += 1
        preset_stats[preset]["current"] += current
        preset_stats[preset]["estimated"] += estimated
        preset_stats[preset]["savings"] += saving

    # Find best preset by total savings
    if preset_stats:
        best_preset = max(preset_stats.items(), key=lambda x: x[1]["savings"])
        max_savings = best_preset[1]["savings"]
        (max_savings / best_preset[1]["current"] * 100) if best_preset[1]["current"] > 0 else 0

    # Show breakdown by resolution
    resolution_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "current": 0, "estimated": 0})

    for _, current, estimated, metadata in rows:
        resolution = metadata.get("resolution", "unknown")
        resolution_stats[resolution]["count"] += 1
        resolution_stats[resolution]["current"] += current
        resolution_stats[resolution]["estimated"] += estimated

    if len(resolution_stats) > 1:  # Only show breakdown if there are multiple resolutions
        for resolution, stats in sorted(resolution_stats.items()):
            current_gb = stats["current"] / (1024**3)
            stats["estimated"] / (1024**3)
            saving_gb = (stats["current"] - stats["estimated"]) / (1024**3)
            (saving_gb / current_gb * 100) if current_gb > 0 else 0

    # Show codec breakdown
    codec_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "current": 0, "estimated": 0})

    for _, current, estimated, metadata in rows:
        codec = metadata.get("codec", "unknown")
        codec_stats[codec]["count"] += 1
        codec_stats[codec]["current"] += current
        codec_stats[codec]["estimated"] += estimated

    if len(codec_stats) > 1:  # Only show breakdown if there are multiple codecs
        for codec, stats in sorted(codec_stats.items(), key=lambda x: x[1]["current"], reverse=True):
            current_gb = stats["current"] / (1024**3)
            stats["estimated"] / (1024**3)
            saving_gb = (stats["current"] - stats["estimated"]) / (1024**3)
            (saving_gb / current_gb * 100) if current_gb > 0 else 0

    # Show preset breakdown - ALWAYS show this as it shows all tried presets
    for preset, stats in sorted(preset_stats.items(), key=lambda x: x[1]["savings"], reverse=True):
        current_gb = stats["current"] / (1024**3)
        stats["estimated"] / (1024**3)
        saving_gb = stats["savings"] / (1024**3)
        (saving_gb / current_gb * 100) if current_gb > 0 else 0

    # Save to CSV if requested
    if csv_path:
        csv_path_obj = Path(csv_path)
        csv_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with csv_path_obj.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "file_path",
                    "current_bytes",
                    "estimated_bytes",
                    "saving_bytes",
                    "saving_percent",
                    "crf",
                    "preset",
                    "predicted_ssim",
                    "complexity",
                    "resolution",
                    "codec",
                    "duration_sec",
                    "limitation_reason",
                ]
            )

            for file_path, current, estimated, metadata in rows:
                saving_bytes = current - estimated
                saving_percent = (saving_bytes / current * 100) if current > 0 else 0

                writer.writerow(
                    [
                        str(file_path),
                        current,
                        estimated,
                        saving_bytes,
                        f"{saving_percent:.1f}",
                        metadata.get("crf", ""),
                        metadata.get("preset", ""),
                        f"{metadata.get('predicted_ssim', 0):.3f}",
                        f"{metadata.get('complexity', 0):.3f}",
                        metadata.get("resolution", ""),
                        metadata.get("codec", ""),
                        f"{metadata.get('duration', 0):.1f}",
                        metadata.get("limitation_reason", ""),
                    ]
                )

            # Add totals row
            writer.writerow(
                [
                    "TOTAL",
                    current_total,
                    estimated_total,
                    saving_total,
                    f"{saving_percent:.1f}",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
