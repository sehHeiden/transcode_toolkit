"""Shared audio analysis functions for estimation and transcoding."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .ffmpeg import FFmpegProbe

if TYPE_CHECKING:
    from pathlib import Path

LOG = logging.getLogger(__name__)

# Simple module-level caches
_file_cache: dict[Path, dict[str, Any]] = {}
_folder_snr_cache: dict[Path, float] = {}


@dataclass
class BitrateDecision:
    """Result of effective bitrate calculation."""

    effective_bitrate_bps: int
    effective_bitrate_str: str
    limitation_reason: str | None


def analyze_file(file_path: Path, use_cache: bool = True) -> dict[str, Any]:
    """Get audio analysis for a single file with caching."""
    if use_cache and file_path in _file_cache:
        return _file_cache[file_path]

    try:
        audio_info = FFmpegProbe.get_audio_info(file_path)
        estimated_snr = FFmpegProbe.estimate_snr(file_path, audio_info)

        analysis = {
            "duration": audio_info["duration"],
            "size": audio_info["size"],
            "codec": audio_info["codec"] or "unknown",
            "bitrate": int(audio_info["bitrate"]) if audio_info["bitrate"] else None,
            "estimated_snr": estimated_snr,
            "sample_rate": audio_info.get("sample_rate"),
            "channels": audio_info.get("channels"),
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
            "bitrate": None,
            "estimated_snr": 60.0,  # Conservative default
            "sample_rate": None,
            "channels": None,
        }

        if use_cache:
            _file_cache[file_path] = analysis

        return analysis


def analyze_folder_snr(folder: Path, audio_files: list[Path], sample_percentage: float = 0.2) -> float:
    """Analyze folder SNR using conservative sampling."""
    if folder in _folder_snr_cache:
        return _folder_snr_cache[folder]

    total_files = len(audio_files)
    if total_files == 0:
        return 60.0

    # Conservative sampling strategy
    if total_files <= 5:
        samples = audio_files[:]  # All files
    elif total_files <= 20:
        samples = audio_files[1:-1]  # Skip first/last
    else:
        # Large folder: sample percentage, minimum 5 files
        sample_count = max(5, int(total_files * sample_percentage))
        skip_count = max(1, total_files // 10)  # Skip 10% from edges

        safe_start = skip_count
        safe_end = total_files - skip_count
        safe_range = safe_end - safe_start

        if safe_range <= 0:
            samples = [audio_files[total_files // 2]]
        else:
            samples = []
            for i in range(sample_count):
                idx = safe_start + (i * safe_range) // sample_count
                samples.append(audio_files[idx])

    # Analyze samples
    snr_values = []
    for file_path in samples:
        try:
            analysis = analyze_file(file_path, use_cache=True)
            snr_values.append(analysis["estimated_snr"])
        except Exception:
            continue

    folder_snr = sum(snr_values) / len(snr_values) if snr_values else 60.0

    LOG.debug(f"Folder {folder.name} SNR: {folder_snr:.1f} dB ({len(snr_values)} samples from {total_files} files)")

    _folder_snr_cache[folder] = folder_snr
    return folder_snr


def calculate_effective_bitrate(
    file_analysis: dict[str, Any], preset_config, folder_snr: float | None = None
) -> BitrateDecision:
    """Calculate effective bitrate using unified logic."""
    target_bitrate_bps = int(preset_config.bitrate.rstrip("k")) * 1000

    # Use folder SNR if available, otherwise file SNR
    effective_snr = folder_snr if folder_snr is not None else file_analysis["estimated_snr"]

    # Check if SNR scaling is disabled
    if not getattr(preset_config, "snr_bitrate_scale", True) or getattr(preset_config, "min_snr_db", None) is None:
        # Only apply input bitrate limit
        if file_analysis["bitrate"] and target_bitrate_bps > file_analysis["bitrate"]:
            effective_bitrate_bps = file_analysis["bitrate"]
            limitation_reason = "input limit"
        else:
            effective_bitrate_bps = target_bitrate_bps
            limitation_reason = None
    else:
        # SNR scaling enabled
        limitation_reasons = []

        if effective_snr < preset_config.min_snr_db:
            # Scale down based on SNR
            snr_ratio = effective_snr / preset_config.min_snr_db
            snr_adjusted_bps = int(target_bitrate_bps * snr_ratio)

            # Apply minimum floor
            is_voice = preset_config.application in ["voip", "voice"]
            min_bitrate_bps = 32000 if is_voice else 64000
            snr_adjusted_bps = max(snr_adjusted_bps, min_bitrate_bps)

            limitation_reasons.append(f"SNR-limited ({effective_snr:.1f}dB < {preset_config.min_snr_db}dB)")
        else:
            snr_adjusted_bps = target_bitrate_bps

        # Apply input bitrate ceiling
        if file_analysis["bitrate"] and snr_adjusted_bps > file_analysis["bitrate"]:
            effective_bitrate_bps = file_analysis["bitrate"]
            limitation_reasons.append("input limit")
        else:
            effective_bitrate_bps = snr_adjusted_bps

        limitation_reason = " + ".join(limitation_reasons) if limitation_reasons else None

    return BitrateDecision(
        effective_bitrate_bps=effective_bitrate_bps,
        effective_bitrate_str=f"{effective_bitrate_bps // 1000}k",
        limitation_reason=limitation_reason,
    )


def estimate_file_size(file_analysis: dict[str, Any], effective_bitrate_bps: int) -> int:
    """Estimate output file size."""
    return int(file_analysis["duration"] * effective_bitrate_bps / 8)


def clear_caches() -> None:
    """Clear all cached data."""
    _file_cache.clear()
    _folder_snr_cache.clear()


def get_cache_stats() -> dict[str, int]:
    """Get cache statistics."""
    return {
        "files_cached": len(_file_cache),
        "folders_cached": len(_folder_snr_cache),
    }
