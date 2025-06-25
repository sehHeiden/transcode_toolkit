"""audio.estimate - Opus re-encode size estimator."""

import csv
import json
import logging
import random
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, NamedTuple

import psutil
from tqdm import tqdm

from transcode_toolkit.config import get_config


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
    missing = []

    for exe in required:
        if not shutil.which(exe):
            missing.append(exe)

    if missing:
        print("❌ Missing required executables:")
        for exe in missing:
            print(f"   - {exe}")
        print("\n💡 Install FFmpeg to get these tools:")
        print("   • Windows: https://ffmpeg.org/download.html#build-windows")
        print("   • winget install Gyan.FFmpeg")
        print("   • choco install ffmpeg")
        print("   • Or download from: https://www.gyan.dev/ffmpeg/builds/")
        print("\n   Make sure to add FFmpeg to your system PATH.")
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
    fmt = json.loads(
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
    )["format"]
    return {"duration": float(fmt.get("duration", 0)), "size": path.stat().st_size}


def _estimate_folder_snr(folder: Path, audio_files: list[Path]) -> float:
    """Estimate representative SNR for an entire folder by smart sampling."""
    from transcode_toolkit.core import FFmpegProbe

    total_files = len(audio_files)
    if total_files == 0:
        return 60.0  # Default conservative SNR

    # Smart sampling strategy based on folder size
    samples = []

    if total_files <= 3:
        # Small folder: use all files
        samples = audio_files[:]
    elif total_files <= 10:
        # Medium folder: skip first and last, sample middle
        start_idx = 1
        end_idx = total_files - 1
        mid_idx = total_files // 2
        samples = [
            audio_files[start_idx],
            audio_files[mid_idx],
            audio_files[end_idx - 1],
        ]
    else:
        # Large folder: avoid first/last 10% and sample from middle regions
        skip_count = max(1, total_files // 10)  # Skip 10% from each end
        safe_start = skip_count
        safe_end = total_files - skip_count
        safe_range = safe_end - safe_start

        if safe_range <= 0:
            # Fallback if math goes wrong
            samples = [audio_files[total_files // 2]]
        else:
            # Sample from 25%, 50%, 75% of the safe range
            quarter = safe_range // 4
            samples = [
                audio_files[safe_start + quarter],  # 25% through safe range
                audio_files[safe_start + safe_range // 2],  # 50% through safe range
                audio_files[safe_start + 3 * quarter],  # 75% through safe range
            ]

    snr_values = []
    for file_path in samples:
        try:
            audio_info = FFmpegProbe.get_audio_info(file_path)
            snr = FFmpegProbe.estimate_snr(file_path, audio_info)
            snr_values.append(snr)
            LOG.debug("Sample SNR for %s: %.1f dB", file_path.name, snr)
        except Exception as e:
            LOG.debug("Failed to get SNR for sample %s: %s", file_path, e)
            continue

    if not snr_values:
        LOG.warning("No valid SNR samples in folder %s, using default", folder)
        return 60.0

    folder_snr = sum(snr_values) / len(snr_values)
    LOG.debug(
        "Folder %s estimated SNR: %.1f dB (from %d samples out of %d files)",
        folder.name,
        folder_snr,
        len(snr_values),
        total_files,
    )
    return folder_snr


def _calculate_effective_bitrate_cached(audio_cache: dict[str, Any], preset_config) -> int:
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
    print(f"Using {max_workers} workers to probe {len(files)} files...")

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

    elapsed = time.time() - start_time
    print(f"Probed {len(rows)} files in {elapsed:.2f}s ({len(rows) / elapsed:.1f} files/sec)")
    return rows


def compare_presets(root: Path) -> list[EstimationResult]:
    """Compare all presets using optimized folder-level SNR sampling."""
    config = get_config()
    audio_exts = config.audio.extensions

    # Group files by folders for SNR sampling
    folder_groups: dict[Path, list[Path]] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in audio_exts:
            folder = p.parent
            if folder not in folder_groups:
                folder_groups[folder] = []
            folder_groups[folder].append(p)

    total_audio_files = sum(len(files) for files in folder_groups.values())
    print(f"Found {total_audio_files} audio files in {len(folder_groups)} folders")

    # Do folder-level SNR sampling - only probe sample files per folder
    sample_files = []
    folder_snr_cache = {}
    max_samples_per_folder = 3  # Only probe 3 files per folder for SNR estimation

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

    print(f"Sampling {len(sample_files)} files from {len(folder_groups)} folders for metadata analysis")

    # Get all metadata and bitrate info in a single FFprobe operation per file
    def get_complete_file_info(file_path):
        """Get both basic metadata and bitrate info in one FFprobe call."""
        try:
            from transcode_toolkit.core import FFmpegProbe
            from transcode_toolkit.core.ffmpeg import FFmpegError

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
        except FFmpegError as e:
            # Capture the actual FFmpeg error message
            error_msg = str(e)

            # Extract meaningful parts of the error message
            if "No audio streams found" in error_msg:
                reason = "No audio streams found"
            elif "timed out" in error_msg.lower():
                reason = "FFprobe timeout"
            elif "Invalid JSON" in error_msg:
                reason = "Invalid JSON response"
            elif "ffprobe failed for" in error_msg:
                # Extract the actual ffprobe stderr/stdout message
                parts = error_msg.split(": ", 2)
                if len(parts) >= 3:
                    actual_error = parts[2].strip()
                    # Clean up common FFmpeg error patterns
                    if actual_error:
                        reason = actual_error[:100]  # Limit length but show actual error
                    else:
                        reason = f"FFprobe error (code {e.return_code})" if e.return_code else "FFprobe failed"
                else:
                    reason = f"FFprobe error (code {e.return_code})" if e.return_code else "FFprobe failed"
            else:
                # Show the actual error message, truncated if too long
                reason = error_msg[:100] + "..." if len(error_msg) > 100 else error_msg
            return file_path, None, reason
        except subprocess.CalledProcessError as e:
            # Direct subprocess errors
            reason = f"FFprobe subprocess error (code {e.returncode})"
            return file_path, None, reason
        except FileNotFoundError:
            reason = "File not found"
            return file_path, None, reason
        except PermissionError:
            reason = "Permission denied"
            return file_path, None, reason
        except Exception as e:
            # Other unexpected errors
            error_type = type(e).__name__
            reason = f"{error_type}: {str(e)[:50]}..."
            return file_path, None, reason

    # Use more conservative worker count for better performance
    max_workers = max(
        1, min(8, psutil.cpu_count(logical=False) or 1, len(sample_files))
    )  # Cap at 8, use physical cores, minimum 1
    print(f"Using {max_workers} workers to analyze {len(sample_files)} files (single FFprobe per file)...")

    file_metadata = {}
    failed_files = []  # Track failed files with reasons
    start_time = time.time()

    # Temporarily disable FFmpeg warnings during bulk processing
    ffmpeg_logger = logging.getLogger("transcode_toolkit.core.ffmpeg")
    original_level = ffmpeg_logger.level
    ffmpeg_logger.setLevel(logging.ERROR)  # Only show errors, suppress warnings

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(get_complete_file_info, file_path): file_path for file_path in sample_files
            }
            for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Analyzing Audio Files"):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if len(result) == 3:  # New format with error reason
                        _, metadata, error_reason = result
                        if metadata:  # Only add if we got valid metadata
                            file_metadata[file_path] = metadata
                        else:
                            failed_files.append((file_path, error_reason or "Unknown error"))
                    else:  # Legacy format
                        _, metadata = result
                        if metadata:  # Only add if we got valid metadata
                            file_metadata[file_path] = metadata
                        else:
                            failed_files.append((file_path, "No metadata returned"))
                except Exception as exc:
                    # Suppress individual warnings during processing, but track reason
                    reason = str(exc).split(":")[0] if ":" in str(exc) else str(exc)[:50]
                    failed_files.append((file_path, reason))
                    continue
    finally:
        # Restore original logging level
        ffmpeg_logger.setLevel(original_level)

    elapsed = time.time() - start_time
    print(f"Analyzed {len(file_metadata)} files in {elapsed:.2f}s ({len(file_metadata) / elapsed:.1f} files/sec)")

    # Show failed files summary if any
    if failed_files:
        from collections import Counter

        reasons = Counter(reason for _, reason in failed_files)
        most_common = reasons.most_common(3)  # Show top 3 reasons

        print(f"\n⚠️  {len(failed_files)} files failed analysis. Common reasons:")
        for reason, count in most_common:
            print(f"   • {reason} ({count} files)")

    # Combine all the information - extrapolate sample data to all files
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

    # If best saving is less than 5%, recommend keeping original
    if best.saving_percent < 5:
        return "no_conversion"  # Special case

    # For audiobooks, prefer audiobook presets if they're in top 2
    audiobook_presets = [r for r in sorted_results[:2] if "audiobook" in r.preset]
    if audiobook_presets:
        return audiobook_presets[0].preset

    return best.preset


def print_comparison(results: list[EstimationResult], recommended: str) -> None:
    """Print a formatted comparison of all presets."""
    print("\n=== PRESET COMPARISON ===")
    print(f"{'Preset':<20} {'Current':>10} {'Estimated':>10} {'Saving':>10} {'%':>8}")
    print("-" * 60)

    for result in sorted(results, key=lambda x: x.saving_percent, reverse=True):
        marker = " ★" if result.preset == recommended else "  "
        print(
            f"{result.preset + marker:<20} "
            f"{result.current_size / 2**20:8.1f} MB "
            f"{result.estimated_size / 2**20:8.1f} MB "
            f"{result.saving / 2**20:8.1f} MB "
            f"{result.saving_percent:6.1f}%"
        )

    print(f"\n★ RECOMMENDED: {recommended}")

    if recommended == "no_conversion":
        print("  → Files are already well compressed. Conversion may not be worth it.")
    else:
        config = get_config()
        preset_config = config.audio.presets.get(recommended)
        if preset_config:
            print(f"  → Bitrate: {preset_config.bitrate}")
            print(f"  → Application: {preset_config.application}")
            print(f"  → Description: {preset_config.description}")
            if preset_config.cutoff:
                print(f"  → Frequency cutoff: {preset_config.cutoff} Hz")
            if preset_config.channels:
                print(f"  → Channels: {preset_config.channels}")


def print_summary(rows, *, preset: str, csv_path: str | None = None) -> None:
    cur = sum(r[1] for r in rows)
    new = sum(r[2] for r in rows)
    diff = cur - new
    pct = 0 if cur == 0 else 100 * diff / cur
    print(f"{preset} files analysed : {len(rows):,}")
    print(f"Current size            : {cur / 2**30:.2f} GiB")
    print(f"Estimated Opus          : {new / 2**30:.2f} GiB")
    print(f"Potential saving        : {diff / 2**20:.1f} MiB ({pct:.1f} %)")

    if csv_path:
        csv_path_obj = Path(csv_path)
        csv_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with csv_path_obj.open("w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["file", "current", "estimated", "saving"])
            for p, c, e in rows:
                wr.writerow([p, c, e, c - e])
            wr.writerow(["TOTAL", cur, new, diff])
        print("CSV report →", csv_path_obj)


def cli(
    path: Path,
    *,
    preset: str | None = None,
    csv_path: str | None = None,
    json_path: str | None = None,
    verbose: bool = False,
    compare: bool = False,
) -> None:
    """
    CLI entry point for audio estimation.

    Args:
        path: Directory to analyze
        preset: Specific preset to analyze (if None, compares all)
        csv_path: Path to save CSV report
        json_path: Path to save JSON report
        verbose: Enable verbose logging
        compare: Force comparison mode even if preset is specified

    """
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
            print(f"\nJSON report → {json_path}")

        print(
            f"\n💡 To convert with recommended settings:\n   python -m src.transcode_toolkit audio transcode '{path}' --preset {recommended}"
        )
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
        print(f"\nJSON report → {json_path}")
