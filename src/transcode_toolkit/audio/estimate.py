"""audio.estimate - Opus re-encode size estimator."""

import csv
import json
import logging
import random
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, NamedTuple

import psutil
from tqdm import tqdm

from ..config import get_config
from ..config.constants import (
    ERROR_MESSAGE_TRUNCATE_LENGTH,
    FFPROBE_ERROR_PARTS_COUNT,
    FUTURE_RESULT_TUPLE_LENGTH,
)
from ..config.settings import AudioPreset
from ..core import FFmpegProbe
from ..core.ffmpeg import FFmpegError


class EstimationResult(NamedTuple):
    """Result of preset comparison analysis."""

    preset: str
    current_size: int
    estimated_size: int
    saving: int
    saving_percent: float


LOG = logging.getLogger("audio-est")


def _check_executables() -> None:
    """Check if required executables are available in PATH."""
    required = ["ffprobe"]
    missing = [exe for exe in required if not shutil.which(exe)]

    if missing:
        for _exe in missing:
            pass
        raise SystemExit(1)


def _probe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_entries",
        "format=duration,bit_rate",
        str(path),
    ]
    # S603: subprocess call with shell=False is safe for hardcoded commands
    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,  # Prevent hanging
    )
    fmt = json.loads(result.stdout)["format"]
    return {"duration": float(fmt.get("duration", 0)), "size": path.stat().st_size}


def _calculate_effective_bitrate_cached(audio_cache: dict[str, Any], preset_config: AudioPreset) -> int:
    """Calculate effective bitrate using cached audio info."""
    target_bitrate_bps = int(preset_config.bitrate.rstrip("k")) * 1000

    # Check if SNR scaling is enabled
    if not preset_config.snr_bitrate_scale or preset_config.min_snr_db is None:
        # Simple estimation using target bitrate or input limit
        if audio_cache.get("bitrate"):
            input_bitrate_bps = int(audio_cache["bitrate"])
            effective_bitrate_bps = min(target_bitrate_bps, input_bitrate_bps)
        else:
            effective_bitrate_bps = target_bitrate_bps
    else:
        # SNR-based calculation using cached SNR
        estimated_snr = audio_cache.get("estimated_snr", 60.0)

        # Apply SNR scaling if below threshold
        if estimated_snr < preset_config.min_snr_db:
            snr_ratio = estimated_snr / preset_config.min_snr_db
            snr_adjusted_bps = int(target_bitrate_bps * snr_ratio)

            # Apply minimum floor
            is_voice = preset_config.application in ["voip", "voice"]
            min_bitrate_bps = 32000 if is_voice else 64000
            snr_adjusted_bps = max(snr_adjusted_bps, min_bitrate_bps)
        else:
            snr_adjusted_bps = target_bitrate_bps

        # Apply input bitrate ceiling
        if audio_cache.get("bitrate"):
            input_bitrate_bps = int(audio_cache["bitrate"])
            effective_bitrate_bps = min(snr_adjusted_bps, input_bitrate_bps)
        else:
            effective_bitrate_bps = snr_adjusted_bps

    return effective_bitrate_bps


def _estimate(meta: dict[str, Any], br: int) -> int:
    """Legacy simple estimation (kept for backward compatibility)."""
    return int(meta["duration"] * br / 8)


def analyse(root: Path, target_br: int) -> list[tuple[Path, int, int]]:
    """Analyze directory and estimate sizes for given target bitrate."""
    config = get_config()
    audio_exts = config.audio.extensions

    rows = []
    files = [p for p in root.rglob("*") if p.suffix.lower() in audio_exts]

    # Use more aggressive worker count for I/O bound FFprobe operations
    max_workers = max(1, min(psutil.cpu_count(logical=True) or 1, len(files)))  # Ensure minimum 1 worker

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(_probe, p): p for p in files}
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Probing Audio Files"):
            p = future_to_file[future]
            try:
                meta = future.result()
                rows.append((p, meta["size"], _estimate(meta, target_br)))
            except subprocess.CalledProcessError as exc:
                LOG.warning("ffprobe failed for %s: %s", p, exc)
                continue

    time.time() - start_time
    return rows


def _group_files_by_folder(root: Path, audio_exts: list[str]) -> dict[Path, list[Path]]:
    """Group audio files by their parent folder."""
    folder_groups: dict[Path, list[Path]] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in audio_exts:
            folder = p.parent
            if folder not in folder_groups:
                folder_groups[folder] = []
            folder_groups[folder].append(p)
    return folder_groups


def _create_folder_samples(folder_groups: dict[Path, list[Path]]) -> tuple[list[Path], dict[Path, float]]:
    """Create sample files for SNR estimation from each folder."""
    sample_files = []
    folder_snr_cache = {}
    max_samples_per_folder = 3  # Maximum number of samples per folder for SNR estimation

    for folder, files in folder_groups.items():
        # Sample a few files from each folder for SNR estimation
        folder_samples = (
            files[:max_samples_per_folder]
            if len(files) <= max_samples_per_folder
            else random.sample(files, max_samples_per_folder)
        )
        sample_files.extend(folder_samples)

        # Use default SNR for fast estimation
        folder_snr_cache[folder] = 65.0  # Conservative estimate for mixed content

    return sample_files, folder_snr_cache


def _get_complete_file_info(file_path: Path) -> tuple[Path, dict[str, Any] | None, str | None]:
    """Get both basic metadata and bitrate info in one FFprobe call."""
    try:
        # Use FFmpegProbe.get_audio_info which gets everything we need
        audio_info = FFmpegProbe.get_audio_info(file_path)
        return (
            file_path,
            {
                "duration": audio_info.get("duration", 0),
                "size": audio_info.get("size", file_path.stat().st_size),
                "bitrate": audio_info.get("bitrate"),
                "codec": audio_info.get("codec"),
            },
            None,
        )  # No error
    except (FFmpegError, subprocess.CalledProcessError, FileNotFoundError, PermissionError, OSError) as e:
        reason = _extract_error_reason(e)
        return file_path, None, reason


def _extract_error_reason(e: Exception) -> str:
    """Extract meaningful error reason from exception."""
    if isinstance(e, FFmpegError):
        return _extract_ffmpeg_error_reason(e)
    if isinstance(e, subprocess.CalledProcessError):
        return f"FFprobe subprocess error (code {e.returncode})"
    if isinstance(e, FileNotFoundError):
        return "File not found"
    if isinstance(e, PermissionError):
        return "Permission denied"
    if isinstance(e, OSError):
        error_type = type(e).__name__
        return f"{error_type}: {str(e)[:50]}..."
    return "Unknown error"


def _extract_ffmpeg_error_reason(e: FFmpegError) -> str:
    """Extract meaningful error reason from FFmpegError."""
    error_msg = str(e)
    if "No audio streams found" in error_msg:
        return "No audio streams found"
    if "timed out" in error_msg.lower():
        return "FFprobe timeout"
    if "Invalid JSON" in error_msg:
        return "Invalid JSON response"
    if "ffprobe failed for" in error_msg:
        return _extract_ffprobe_error_details(error_msg, e)
    if len(error_msg) > ERROR_MESSAGE_TRUNCATE_LENGTH:
        return error_msg[:ERROR_MESSAGE_TRUNCATE_LENGTH] + "..."
    return error_msg


def _extract_ffprobe_error_details(error_msg: str, e: FFmpegError) -> str:
    """Extract detailed error information from ffprobe error message."""
    parts = error_msg.split(": ", 2)
    if len(parts) >= FFPROBE_ERROR_PARTS_COUNT:
        actual_error = parts[2].strip()
        if actual_error:
            return actual_error[:100]
    return f"FFprobe error (code {e.return_code})" if e.return_code else "FFprobe failed"


def _process_file_metadata(sample_files: list[Path]) -> tuple[dict[Path, dict[str, Any]], list[tuple[Path, str]]]:
    """Process file metadata for sample files."""
    # Use more conservative worker count for better performance
    max_workers = max(
        1, min(8, psutil.cpu_count(logical=False) or 1, len(sample_files))
    )  # Cap at 8, use physical cores, minimum 1

    file_metadata = {}
    failed_files = []  # Track failed files with reasons
    start_time = time.time()

    # Temporarily disable FFmpeg warnings during bulk processing
    ffmpeg_logger = logging.getLogger("..core.ffmpeg")
    original_level = ffmpeg_logger.level
    ffmpeg_logger.setLevel(logging.ERROR)  # Only show errors, suppress warnings

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(_get_complete_file_info, file_path): file_path for file_path in sample_files
            }
            for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Analyzing Audio Files"):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if len(result) == FUTURE_RESULT_TUPLE_LENGTH:  # New format with error reason
                        _, metadata, error_reason = result
                        if metadata:  # Only add if we got valid metadata
                            file_metadata[file_path] = metadata
                        else:
                            failed_files.append((file_path, error_reason or "Unknown error"))
                    else:  # Legacy format - shouldn't happen but handle gracefully
                        LOG.warning("Unexpected result format from %s: %s", file_path, result)
                        failed_files.append((file_path, "Unexpected result format"))
                except (OSError, ValueError) as exc:
                    # Handle specific exceptions during processing
                    reason = str(exc).split(":")[0] if ":" in str(exc) else str(exc)[:50]
                    failed_files.append((file_path, reason))
                    continue
    finally:
        # Restore original logging level
        ffmpeg_logger.setLevel(original_level)

    time.time() - start_time

    # Show failed files summary if any
    if failed_files:
        reasons = Counter(reason for _, reason in failed_files)
        most_common = reasons.most_common(3)  # Show top 3 reasons

        for _reason, _count in most_common:
            pass

    return file_metadata, failed_files


def _create_files_with_cache(
    folder_groups: dict[Path, list[Path]],
    folder_snr_cache: dict[Path, float],
    file_metadata: dict[Path, dict[str, Any]],
) -> list[tuple[Path, dict[str, Any]]]:
    """Create files with cached metadata for all files in folder groups."""
    files_with_cache = []
    for folder, files in folder_groups.items():
        folder_snr = folder_snr_cache[folder]

        # Get representative metadata from sampled files in this folder
        folder_metadata: list[dict[str, Any]] = [meta for path, meta in file_metadata.items() if path.parent == folder]

        if folder_metadata:
            # Use average bitrate and most common codec from samples
            bitrates = [int(meta.get("bitrate", 0) or 0) for meta in folder_metadata]
            avg_bitrate = sum(bitrates) / len(bitrates) if bitrates else 128000
            codecs: list[str] = [codec for meta in folder_metadata if (codec := meta.get("codec")) is not None]
            representative_codec = max(set(codecs), key=codecs.count) if codecs else "unknown"
        else:
            # Fallback if no samples succeeded in this folder
            avg_bitrate = 128000  # 128k default
            representative_codec = "unknown"

        # Apply representative data to all files in folder
        for file_path in files:
            if file_path in file_metadata:
                # Use actual metadata for sampled files
                meta = file_metadata[file_path]
                cache = {
                    "duration": meta["duration"],
                    "size": meta["size"],
                    "estimated_snr": folder_snr,
                    "bitrate": meta["bitrate"],
                    "codec": meta["codec"],
                }
            else:
                # Extrapolate for non-sampled files using file size and representative data
                file_size = file_path.stat().st_size
                # Estimate duration from file size and bitrate
                estimated_duration = (file_size * 8) / avg_bitrate if avg_bitrate > 0 else 180  # 3min default
                cache = {
                    "duration": estimated_duration,
                    "size": file_size,
                    "estimated_snr": folder_snr,
                    "bitrate": int(avg_bitrate),
                    "codec": representative_codec,
                }
            files_with_cache.append((file_path, cache))

    return files_with_cache


def compare_presets(root: Path) -> list[EstimationResult]:
    """Compare all presets using optimized folder-level SNR sampling."""
    config = get_config()
    audio_exts = config.audio.extensions

    folder_groups = _group_files_by_folder(root, audio_exts)
    sample_files, folder_snr_cache = _create_folder_samples(folder_groups)

    file_metadata, failed_files = _process_file_metadata(sample_files)

    # Combine all the information - extrapolate sample data to all files
    files_with_cache = _create_files_with_cache(folder_groups, folder_snr_cache, file_metadata)

    # Calculate estimates for each preset using cached data
    results = []
    for preset_name, preset_config in config.audio.presets.items():
        current_size = sum(cache["size"] for _, cache in files_with_cache)

        # Use cached SNR data for fast calculation
        estimated_size = sum(
            _calculate_effective_bitrate_cached(cache, preset_config) * cache["duration"] / 8
            for _, cache in files_with_cache
        )

        saving = current_size - estimated_size
        saving_percent = 0 if current_size == 0 else 100 * saving / current_size

        results.append(
            EstimationResult(
                preset=preset_name,
                current_size=current_size,
                estimated_size=estimated_size,
                saving=saving,
                saving_percent=saving_percent,
            )
        )

    return results


def recommend_preset(results: list[EstimationResult]) -> str:
    """Recommend the best preset based on analysis."""
    if not results:
        return "music"  # default fallback

    # Sort by saving percentage (descending)
    sorted_results = sorted(results, key=lambda x: x.saving_percent, reverse=True)

    best = sorted_results[0]

    # If best saving is less than configured threshold, recommend keeping original
    config = get_config()
    min_savings_threshold = config.audio.quality_thresholds.get("min_saving_percent", 5)
    if best.saving_percent < min_savings_threshold:
        return "no_conversion"  # Special case

    # For audiobooks, prefer audiobook presets if they're in top 2
    audiobook_presets = [r for r in sorted_results[:2] if "audiobook" in r.preset]
    if audiobook_presets:
        return audiobook_presets[0].preset

    return best.preset


def print_comparison(results: list[EstimationResult], recommended: str) -> None:
    """Print a formatted comparison of all presets."""
    print()
    print("=" * 60)
    print(f"{'AUDIO PRESET COMPARISON':^60}")
    print("=" * 60)
    print(f"{'Preset':<15} {'Current':<10} {'Estimated':<10} {'Saving':<10} {'%':<8}")
    print("-" * 60)

    sorted_results = sorted(results, key=lambda x: x.saving_percent, reverse=True)

    for result in sorted_results:
        current_mb = result.current_size / (1024**2)
        estimated_mb = result.estimated_size / (1024**2)
        saving_mb = result.saving / (1024**2)
        star = " ★" if result.preset == recommended else "  "

        print(
            f"{result.preset:<15}{star} {current_mb:>7.1f} MB {estimated_mb:>7.1f} MB "
            f"{saving_mb:>7.1f} MB {result.saving_percent:>6.1f}%"
        )

    print("-" * 60)

    if recommended == "no_conversion":
        print("\n⚠️  RECOMMENDATION: Keep original files (insufficient savings)")
    else:
        print(f"\n★ RECOMMENDED: {recommended}")
        config = get_config()
        preset_config = config.audio.presets.get(recommended)
        if preset_config:
            print(f"  → Bitrate: {preset_config.bitrate}")
            print(f"  → Application: {preset_config.application}")
            if preset_config.cutoff:
                print(f"  → Frequency cutoff: {preset_config.cutoff} Hz")
            if preset_config.channels:
                print(f"  → Channels: {preset_config.channels}")
    print()


def print_summary(rows: list[tuple[Path, int, int]], *, preset: str, csv_path: str | None = None) -> None:
    """Print summary of analysis results."""
    if not rows:
        print("No audio files found to analyze.")
        return

    cur = sum(r[1] for r in rows)
    new = sum(r[2] for r in rows)
    diff = cur - new
    pct = 0 if cur == 0 else 100 * diff / cur

    print()
    print("=" * 50)
    print(f"{'AUDIO ESTIMATION SUMMARY':^50}")
    print("=" * 50)
    print(f"Preset: {preset}")
    print(f"Files analyzed: {len(rows)}")
    print(f"Current total size: {cur / (1024**2):.1f} MB")
    print(f"Estimated total size: {new / (1024**2):.1f} MB")
    print(f"Total potential savings: {diff / (1024**2):.1f} MB ({pct:.1f}%)")
    print("=" * 50)

    if csv_path:
        csv_path_obj = Path(csv_path)
        csv_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with csv_path_obj.open("w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["file", "current", "estimated", "saving"])
            for p, c, e in rows:
                wr.writerow([p, c, e, c - e])
            wr.writerow(["TOTAL", cur, new, diff])


def cli(
    path: Path,
    *,
    preset: str | None = None,
    output_paths: tuple[str | None, str | None] = (None, None),
    verbose: bool = False,
    compare: bool = False,
) -> None:
    """
    CLI entry point for audio estimation.

    Args:
        path: Directory to analyze
        preset: Specific preset to analyze (if None, compares all)
        output_paths: Tuple of (csv_path, json_path) for output
        verbose: Enable verbose logging
        compare: Force comparison mode even if preset is specified

    """
    csv_path, json_path = output_paths
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Check if required executables are available
    _check_executables()

    if compare or preset is None:
        # Compare all presets and show recommendations
        results = compare_presets(path)
        recommended = recommend_preset(results)
        print_comparison(results, recommended)

        if json_path:
            config = get_config()
            output_data = {
                "recommended_preset": recommended,
                "available_presets": config.list_audio_presets(),
                "comparisons": [
                    {
                        "preset": r.preset,
                        "current_bytes": r.current_size,
                        "estimated_bytes": r.estimated_size,
                        "saving_bytes": r.saving,
                        "saving_percent": r.saving_percent,
                        "preset_config": config.audio.presets[r.preset].model_dump(),
                    }
                    for r in results
                ],
            }
            Path(json_path).write_text(json.dumps(output_data, indent=2))

        return

    # Original single preset analysis
    config = get_config()
    if preset not in config.audio.presets:
        available = ", ".join(config.list_audio_presets())
        msg = f"Invalid preset: {preset!r}. Available: {available}"
        raise ValueError(msg)

    # Convert bitrate string to bits per second
    preset_config = config.audio.presets[preset]
    target_bitrate = int(preset_config.bitrate.rstrip("k")) * 1000

    rows = analyse(path, target_bitrate)
    print_summary(rows, preset=preset, csv_path=csv_path)

    if json_path:
        cur = sum(r[1] for r in rows)
        new = sum(r[2] for r in rows)
        diff = cur - new
        pct = 0 if cur == 0 else 100 * diff / cur
        json_output_data: dict[str, Any] = {
            "preset_used": preset,
            "preset_config": preset_config.model_dump(),
            "files": len(rows),
            "current_bytes": cur,
            "estimated_bytes": new,
            "saving_bytes": diff,
            "saving_percent": pct,
        }
        Path(json_path).write_text(json.dumps(json_output_data, indent=2))
