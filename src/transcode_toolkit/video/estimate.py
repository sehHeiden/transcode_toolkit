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


def _estimate_size_by_sampling(
    file_path: Path,
    video_info: dict[str, Any],
    quality_decision: QualityDecision,
    target_codec: str,
    sample_duration: int = 30,  # Sample 30 seconds by default
) -> int:
    """
    Estimate file size by actually encoding a sample segment.
    
    This provides much more accurate estimates than theoretical calculations.
    """
    try:
        from ..core.ffmpeg import FFmpegProcessor
        import tempfile
        import time
        
        duration = video_info.get("duration", 0)
        if duration < sample_duration:
            # For short videos, just use theoretical estimation
            return _estimate_size_by_codec(video_info, quality_decision, target_codec)
        
        # Create temporary files for sample
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as sample_input:
            sample_input_path = Path(sample_input.name)
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as sample_output:
            sample_output_path = Path(sample_output.name)
        
        try:
            ffmpeg = FFmpegProcessor(timeout=300)  # 5 minute timeout for sampling
            
            # Extract sample from middle of video (avoid intros/credits)
            start_time = max(60, duration * 0.3)  # Start at 30% or 1 minute, whichever is later
            
            # Step 1: Extract sample segment
            extract_cmd = [
                "ffmpeg", "-y", "-i", str(file_path),
                "-ss", str(start_time), "-t", str(sample_duration),
                "-c", "copy",  # Copy without re-encoding
                str(sample_input_path)
            ]
            
            LOG.debug(f"Extracting {sample_duration}s sample from {file_path.name}")
            ffmpeg.run_command(extract_cmd, file_path)
            
            if not sample_input_path.exists():
                raise Exception("Failed to extract sample segment")
            
            sample_input_size = sample_input_path.stat().st_size
            
            # Step 2: Encode sample with target settings
            from ..config import get_config
            config = get_config()
            preset_config = config.get_video_preset(quality_decision.effective_preset or "default")
            
            encode_cmd = ffmpeg.build_video_command(
                input_file=sample_input_path,
                output_file=sample_output_path,
                codec=target_codec,
                crf=quality_decision.effective_crf,
                preset=quality_decision.effective_preset or "medium",
                preset_config=preset_config,
                gpu="nvenc" in target_codec or "amf" in target_codec or "qsv" in target_codec
            )
            
            LOG.debug(f"Encoding sample with {target_codec}, CRF {quality_decision.effective_crf}")
            start_time = time.time()
            ffmpeg.run_command(encode_cmd, sample_input_path)
            encoding_time = time.time() - start_time
            
            if not sample_output_path.exists():
                raise Exception("Failed to encode sample segment")
            
            sample_output_size = sample_output_path.stat().st_size
            
            # Calculate compression ratio from actual sample
            compression_ratio = sample_output_size / sample_input_size if sample_input_size > 0 else 1.0
            
            # Estimate full file size
            estimated_size = int(video_info["size"] * compression_ratio)
            
            # Add small safety margin (sampling might not be perfectly representative)
            safety_margin = 1.1  # 10% safety margin
            estimated_size = int(estimated_size * safety_margin)
            
            LOG.info(
                f"Sample encoding: {sample_input_size:,} → {sample_output_size:,} bytes "
                f"(ratio: {compression_ratio:.3f}) in {encoding_time:.1f}s. "
                f"Estimated full size: {estimated_size:,} bytes"
            )
            
            return estimated_size
            
        finally:
            # Clean up temporary files
            for temp_file in [sample_input_path, sample_output_path]:
                if temp_file.exists():
                    temp_file.unlink(missing_ok=True)
                    
    except Exception as e:
        LOG.warning(f"Sampling failed for {file_path.name}: {e}. Falling back to theoretical estimation.")
        return _estimate_size_by_codec(video_info, quality_decision, target_codec)


def _estimate_size_by_codec(
    video_info: dict[str, Any],
    quality_decision: QualityDecision,
    target_codec: str,
) -> int:
    """
    Simple fallback estimation for when sampling fails.
    Much simpler than before since this is only used as fallback.
    """
    try:
        original_size = video_info["size"]
        
        # Very simple fallback - assume modest compression
        # This is only used when sampling fails, so be conservative
        if "h265" in target_codec.lower() or "hevc" in target_codec.lower():
            # H.265 generally compresses better
            fallback_ratio = 0.7  # 30% reduction
        else:
            # H.264 or other codecs
            fallback_ratio = 0.8  # 20% reduction
        
        return int(original_size * fallback_ratio)
        
    except Exception:
        # Last resort fallback
        return int(video_info.get("size", 0) * 0.8)




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
                # Use sampling-based estimation for more accuracy
                estimated_size = _estimate_size_by_sampling(
                    file_path, video_info, preset_quality_decision, preset_config.codec
                )

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


def _calculate_preset_score(current: int, estimated: int, metadata: dict[str, Any]) -> float:
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


def print_summary(rows: list[tuple[Path, int, int, dict[str, Any]]], *, csv_path: str | None = None) -> None:
    """Print estimation summary and optionally save to CSV."""
    if not rows:
        return

    # Group results by file to find best preset for each file
    file_results: dict[Path, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    for file_path, current, estimated, metadata in rows:
        file_results[file_path].append((current, estimated, metadata))

    # Calculate totals using ONLY the best preset for each file

    current_total = 0
    estimated_total = 0

    # For each file, find the best preset and use its estimates for totals
    for file_path, file_presets in file_results.items():
        if not file_presets:
            continue

        # Find the best preset for this file
        best_preset = max(file_presets, key=lambda x: _calculate_preset_score(x[0], x[1], x[2]))

        current_total += best_preset[0]  # current size
        estimated_total += best_preset[1]  # estimated size

    saving_total = current_total - estimated_total
    saving_percent = (saving_total / current_total * 100) if current_total > 0 else 0

    # Print overall summary
    print()
    print("=" * 80)
    print(f"{'VIDEO TRANSCODING ESTIMATION SUMMARY':^80}")
    print("=" * 80)
    print(f"Total files analyzed: {len(file_results)}")
    print(f"Current total size: {current_total / (1024**3):.2f} GB")
    print(f"Estimated total size: {estimated_total / (1024**3):.2f} GB")
    print(f"Total potential savings: {saving_total / (1024**3):.2f} GB ({saving_percent:.1f}%)")
    print()

    # Print detailed per-file preset comparison
    print("📁 DETAILED FILE ANALYSIS")
    print("-" * 80)

    for file_path, file_presets in file_results.items():
        display_name = file_path.name[:70] + "..." if len(file_path.name) > 73 else file_path.name
        print(f"\n📄 {display_name}")

        # Find best preset for this file using combined savings, SSIM, and speed score
        best_score = max(
            _calculate_preset_score(current, estimated, metadata) for current, estimated, metadata in file_presets
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
            crf = metadata.get("crf", 0)
            ssim = metadata.get("predicted_ssim", 0)
            savings = current - estimated
            savings_percent = (savings / current * 100) if current > 0 else 0

            current_mb = current / (1024**2)
            estimated_mb = estimated / (1024**2)
            savings_mb = savings / (1024**2)

            preset_score = _calculate_preset_score(current, estimated, metadata)
            best_indicator = "⭐ YES" if abs(preset_score - best_score) < 0.001 else ""

            # Format speed information
            speed_mins = metadata.get("speed_minutes", 0)
            if speed_mins < 1:
                speed_str = f"{speed_mins * 60:.0f}s"
            elif speed_mins < 60:
                speed_str = f"{speed_mins:.1f}m"
            else:
                hours = int(speed_mins // 60)
                minutes = int(speed_mins % 60)
                speed_str = f"{hours}h{minutes}m"

            target_codec = metadata.get("target_codec", "unknown")

            # Print preset details
            print(
                f"  {preset:12} | CRF {crf:2} | SSIM {ssim:.3f} | {current_mb:6.1f}MB → {estimated_mb:6.1f}MB | Save {savings_mb:5.1f}MB ({savings_percent:4.1f}%) | {speed_str:8} | {target_codec:10} {best_indicator}"
            )

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
        best_preset_key, best_preset_data = max(preset_stats.items(), key=lambda x: x[1]["savings"])
        max_savings = best_preset_data["savings"]
        max_savings_percent = (max_savings / best_preset_data["current"] * 100) if best_preset_data["current"] > 0 else 0

    # Show breakdown by resolution
    resolution_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "current": 0, "estimated": 0})

    for _, current, estimated, metadata in rows:
        resolution = metadata.get("resolution", "unknown")
        resolution_stats[resolution]["count"] += 1
        resolution_stats[resolution]["current"] += current
        resolution_stats[resolution]["estimated"] += estimated

    # Resolution breakdown removed - information already shown per-file

    # Show codec breakdown
    codec_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "current": 0, "estimated": 0})

    for _, current, estimated, metadata in rows:
        codec = metadata.get("codec", "unknown")
        codec_stats[codec]["count"] += 1
        codec_stats[codec]["current"] += current
        codec_stats[codec]["estimated"] += estimated

    # Codec breakdown removed - information already shown per-file

    # Preset breakdown removed - information already shown per-file

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
