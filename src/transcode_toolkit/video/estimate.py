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


def _estimate_hevc_size(
    video_info: dict[str, Any],
    quality_decision: QualityDecision,
    compression_efficiency: float = 0.6,  # HEVC typically 40% smaller than H.264
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

        # Base compression efficiency based on source codec
        if codec in ("hevc", "h265"):
            # Already HEVC - minimal improvement expected
            efficiency = 0.9  # 10% improvement at most
        elif codec in ("h264", "avc"):
            # H.264 to HEVC - good compression gains
            efficiency = compression_efficiency
        elif codec in ("mpeg2", "mpeg4", "xvid", "divx"):
            # Older codecs - excellent compression gains
            efficiency = 0.4  # 60% reduction possible
        else:
            # Unknown codec - conservative estimate
            efficiency = compression_efficiency

        # Adjust efficiency based on complexity
        complexity_factor = 1.0
        if hasattr(quality_decision, "predicted_ssim"):
            # More complex content (lower SSIM target) compresses less efficiently
            if quality_decision.predicted_ssim < 0.90:
                complexity_factor = 1.2  # 20% larger due to complexity
            elif quality_decision.predicted_ssim > 0.95:
                complexity_factor = 0.8  # 20% smaller for simple content

        # Adjust efficiency based on CRF
        crf_factor = 1.0
        if quality_decision.effective_crf < 20:
            crf_factor = 1.3  # Higher quality = larger files
        elif quality_decision.effective_crf > 26:
            crf_factor = 0.7  # Lower quality = smaller files

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


def analyze_single_file(file_path: Path) -> tuple[Path, int, int, QualityDecision] | None:
    """
    Analyze a single video file for size estimation.

    Returns:
        Tuple of (path, current_size, estimated_size, quality_decision) or None if failed

    """
    try:
        if file_path.suffix.lower() not in VIDEO_EXTS:
            return None

        # Get video information and analyze content
        video_info = FFmpegProbe.get_video_info(file_path)
        analysis = analyze_file(file_path, use_cache=True)
        complexity = analysis["complexity"]

        # Calculate optimal encoding parameters
        quality_decision = calculate_optimal_crf(
            file_path=file_path,
            video_info=video_info,
            complexity=complexity,
            target_ssim=analysis["estimated_ssim_threshold"],
        )

        # Estimate HEVC size
        current_size = video_info["size"]
        estimated_size = _estimate_hevc_size(video_info, quality_decision)

        return (file_path, current_size, estimated_size, quality_decision)

    except Exception as e:
        LOG.warning(f"Failed to analyze {file_path}: {e}")
        return None


def analyse(root: Path) -> list[tuple[Path, int, int, dict[str, Any]]]:
    """
    Analyze all video files in a directory tree for HEVC transcoding estimation.

    Returns:
        List of tuples: (path, current_size, estimated_size, metadata)

    """
    LOG.info(f"Analyzing video files in: {root}")

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

    # Analyze each file
    results = []
    failed_count = 0

    for file_path in all_files:
        try:
            # Get video information and analyze content
            video_info = FFmpegProbe.get_video_info(file_path)
            analysis = analyze_file(file_path, use_cache=True)
            complexity = analysis["complexity"]

            # Use folder quality for consistent estimation
            folder_quality = folder_quality_cache[file_path.parent]

            # Calculate optimal encoding parameters
            quality_decision = calculate_optimal_crf(
                file_path=file_path,
                video_info=video_info,
                complexity=complexity,
                target_ssim=folder_quality,
                folder_quality=folder_quality,
            )

            # Estimate HEVC size
            current_size = video_info["size"]
            estimated_size = _estimate_hevc_size(video_info, quality_decision)

            # Prepare metadata
            metadata = {
                "crf": quality_decision.effective_crf,
                "preset": quality_decision.effective_preset,
                "predicted_ssim": quality_decision.predicted_ssim,
                "complexity": complexity.overall_complexity,
                "resolution": f"{video_info.get('width', 0)}x{video_info.get('height', 0)}",
                "codec": video_info.get("codec", "unknown"),
                "duration": video_info.get("duration", 0),
                "limitation_reason": quality_decision.limitation_reason,
            }

            results.append((file_path, current_size, estimated_size, metadata))

        except Exception as e:
            LOG.warning(f"Failed to analyze {file_path}: {e}")
            failed_count += 1
            continue

    if failed_count > 0:
        LOG.warning(f"Failed to analyze {failed_count} files")

    LOG.info(f"Successfully analyzed {len(results)} video files")
    return results


def print_summary(rows: list[tuple[Path, int, int, dict[str, Any]]], *, csv_path: str | None = None) -> None:
    """Print estimation summary and optionally save to CSV."""
    if not rows:
        return

    # Calculate totals
    current_total = sum(r[1] for r in rows)
    estimated_total = sum(r[2] for r in rows)
    saving_total = current_total - estimated_total
    saving_percent = (saving_total / current_total * 100) if current_total > 0 else 0

    # Print overall summary
    print("\n" + "=" * 60)
    print("VIDEO TRANSCODING ESTIMATION SUMMARY")
    print("=" * 60)
    print(f"Files analyzed: {len(rows)}")
    print(f"Current total size: {current_total / (1024**3):.2f} GB")
    print(f"Estimated size (HEVC): {estimated_total / (1024**3):.2f} GB")
    print(f"Total savings: {saving_total / (1024**3):.2f} GB ({saving_percent:.1f}%)")

    # Analyze presets and find the best one
    preset_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "current": 0, "estimated": 0, "savings": 0})

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
        max_savings_percent = (max_savings / best_preset[1]["current"] * 100) if best_preset[1]["current"] > 0 else 0
        
        print(f"\nRECOMMENDED PRESET: {best_preset[0]}")
        print(f"Maximum savings: {max_savings / (1024**3):.2f} GB ({max_savings_percent:.1f}%)")
        print(f"Files using this preset: {best_preset[1]['count']}")


    # Show breakdown by resolution
    resolution_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "current": 0, "estimated": 0})

    for _, current, estimated, metadata in rows:
        resolution = metadata.get("resolution", "unknown")
        resolution_stats[resolution]["count"] += 1
        resolution_stats[resolution]["current"] += current
        resolution_stats[resolution]["estimated"] += estimated

    if len(resolution_stats) > 1:  # Only show breakdown if there are multiple resolutions
        print("\nBREAKDOWN BY RESOLUTION:")
        print("-" * 60)
        for resolution, stats in sorted(resolution_stats.items()):
            current_gb = stats["current"] / (1024**3)
            estimated_gb = stats["estimated"] / (1024**3)
            saving_gb = (stats["current"] - stats["estimated"]) / (1024**3)
            saving_percent_res = (saving_gb / current_gb * 100) if current_gb > 0 else 0
            print(f"{resolution:10} | {stats['count']:3} files | {current_gb:6.2f} GB → {estimated_gb:6.2f} GB | {saving_gb:6.2f} GB ({saving_percent_res:4.1f}%)")

    # Show codec breakdown
    codec_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "current": 0, "estimated": 0})

    for _, current, estimated, metadata in rows:
        codec = metadata.get("codec", "unknown")
        codec_stats[codec]["count"] += 1
        codec_stats[codec]["current"] += current
        codec_stats[codec]["estimated"] += estimated

    if len(codec_stats) > 1:  # Only show breakdown if there are multiple codecs
        print("\nBREAKDOWN BY CODEC:")
        print("-" * 60)
        for codec, stats in sorted(codec_stats.items(), key=lambda x: x[1]["current"], reverse=True):
            current_gb = stats["current"] / (1024**3)
            estimated_gb = stats["estimated"] / (1024**3)
            saving_gb = (stats["current"] - stats["estimated"]) / (1024**3)
            saving_percent_codec = (saving_gb / current_gb * 100) if current_gb > 0 else 0
            print(f"{codec:10} | {stats['count']:3} files | {current_gb:6.2f} GB → {estimated_gb:6.2f} GB | {saving_gb:6.2f} GB ({saving_percent_codec:4.1f}%)")

    # Show preset breakdown - ALWAYS show this as it shows all tried presets
    print("\nBREAKDOWN BY PRESET:")
    print("-" * 60)
    for preset, stats in sorted(preset_stats.items(), key=lambda x: x[1]["savings"], reverse=True):
        current_gb = stats["current"] / (1024**3)
        estimated_gb = stats["estimated"] / (1024**3)
        saving_gb = stats["savings"] / (1024**3)
        saving_percent_preset = (saving_gb / current_gb * 100) if current_gb > 0 else 0
        print(f"{preset:10} | {stats['count']:3} files | {current_gb:6.2f} GB → {estimated_gb:6.2f} GB | {saving_gb:6.2f} GB ({saving_percent_preset:4.1f}%)")
    
    print("=" * 60)


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
