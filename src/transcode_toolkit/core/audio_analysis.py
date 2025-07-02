"""Shared audio analysis functions for estimation and transcoding."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..config.constants import (
    AUDIO_MEDIUM_FOLDER_THRESHOLD,
    AUDIO_SMALL_FOLDER_THRESHOLD,
)
from .ffmpeg import FFmpegProbe

if TYPE_CHECKING:
    from pathlib import Path

LOG = logging.getLogger(__name__)

# Simple module-level caches
_file_cache: dict[Path, dict[str, Any]] = {}
_folder_snr_cache: dict[Path, float] = {}


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
    if total_files <= AUDIO_SMALL_FOLDER_THRESHOLD:
        samples = audio_files[:]  # All files
    elif total_files <= AUDIO_MEDIUM_FOLDER_THRESHOLD:
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
