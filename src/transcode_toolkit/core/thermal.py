"""Unified thermal management and worker count optimization for all processors."""

from __future__ import annotations

import logging
import time
from typing import Literal

import psutil

LOG = logging.getLogger(__name__)

MediaType = Literal["audio", "video"]


def get_thermal_safe_worker_count(configured_workers: int | None, media_type: MediaType = "audio") -> int:
    """
    Get a thermally safe number of workers that doesn't overwhelm the system.

    Uses psutil to monitor system load and temperature (where available),
    ensuring we don't use too many cores that could cause thermal issues.

    Args:
        configured_workers: The configured worker count, or None for auto-detection
        media_type: Type of media processing - affects thermal limits

    Returns:
        Safe number of workers considering thermal and system constraints

    """
    if configured_workers is not None and configured_workers > 0:
        # Still apply thermal safety checks even for manual configuration
        safe_workers = min(configured_workers, _get_thermal_limit(media_type))
        if safe_workers < configured_workers:
            LOG.warning(
                "Reducing configured workers from %d to %d due to thermal/system load constraints for %s processing",
                configured_workers,
                safe_workers,
                media_type,
            )
        return safe_workers

    # Get actual core counts using psutil
    try:
        physical_cores = psutil.cpu_count(logical=False) or 1
        logical_cores = psutil.cpu_count(logical=True) or 1

        # Check current system load
        cpu_percent = psutil.cpu_percent(interval=1.0)

        # Different strategies based on media type
        if media_type == "video":
            # Very conservative approach for video encoding:
            # - Use at most 1/3 of physical cores for heavy video transcoding
            max_workers = max(1, physical_cores // 3)
            cpu_threshold = 60  # Lower threshold for video
            max_cap = 2  # Cap at 2 workers for video
        else:  # audio
            # Conservative approach for audio transcoding:
            # - Use at most half the physical cores
            max_workers = max(1, physical_cores // 2)
            cpu_threshold = 70  # Higher threshold for audio
            max_cap = 4  # Cap at 4 workers for audio

        # Apply thermal limits
        thermal_limit = _get_thermal_limit(media_type)
        max_workers = min(max_workers, thermal_limit)

        # If system is already under high load, reduce workers further
        if cpu_percent > cpu_threshold:
            max_workers = max(1, max_workers // 2)
            LOG.warning(
                "High CPU load detected (%.1f%%), reducing %s workers to %d", cpu_percent, media_type, max_workers
            )

        # Apply media-specific cap
        max_workers = min(max_workers, max_cap)

        LOG.info(
            "%s thermal-safe configuration: %d physical cores, %d logical cores, "
            "CPU load: %.1f%%. Using %d workers for thermal safety.",
            media_type.title(),
            physical_cores,
            logical_cores,
            cpu_percent,
            max_workers,
        )

    except (OSError, AttributeError, ValueError) as e:
        # Very conservative fallback if monitoring fails
        LOG.warning("Failed to detect system specs with psutil: %s. Using conservative fallback of 1 worker.", e)
        return 1
    else:
        return max_workers


# Temperature thresholds (Celsius)
VIDEO_TEMP_HIGH = 65
VIDEO_TEMP_MODERATE = 55
AUDIO_TEMP_HIGH = 70
AUDIO_TEMP_MODERATE = 60

# Memory usage thresholds (percentage)
VIDEO_MEMORY_THRESHOLD = 75
AUDIO_MEMORY_THRESHOLD = 80

# Default worker limits
VIDEO_DEFAULT_WORKERS = 2
AUDIO_DEFAULT_WORKERS = 4
VIDEO_FALLBACK_WORKERS = 1
AUDIO_FALLBACK_WORKERS = 2


def _get_max_temperature() -> float:
    """Get maximum current temperature from all available sensors."""
    if not hasattr(psutil, "sensors_temperatures"):
        return 0.0

    temps = psutil.sensors_temperatures()
    if not temps:
        return 0.0

    max_temp = 0.0
    for entries in temps.values():
        for entry in entries:
            if entry.current and entry.current > max_temp:
                max_temp = entry.current
    return max_temp


def _get_thermal_limit_by_temperature(media_type: MediaType, max_temp: float) -> int | None:
    """Get worker limit based on temperature thresholds."""
    if max_temp <= 0:
        return None

    if media_type == "video":
        if max_temp > VIDEO_TEMP_HIGH:
            return 1
        if max_temp > VIDEO_TEMP_MODERATE:
            return 2
    else:  # audio
        if max_temp > AUDIO_TEMP_HIGH:
            return 2
        if max_temp > AUDIO_TEMP_MODERATE:
            return 3

    return None


def _get_thermal_limit_by_memory(media_type: MediaType) -> int | None:
    """Get worker limit based on memory usage."""
    try:
        memory = psutil.virtual_memory()
        threshold = VIDEO_MEMORY_THRESHOLD if media_type == "video" else AUDIO_MEMORY_THRESHOLD

        if memory.percent > threshold:
            return VIDEO_FALLBACK_WORKERS if media_type == "video" else AUDIO_FALLBACK_WORKERS
    except (AttributeError, OSError):
        pass
    return None


def _get_thermal_limit(media_type: MediaType) -> int:
    """
    Determine thermal limits based on system capabilities and media type.

    Args:
        media_type: Type of media processing

    Returns:
        Maximum recommended workers considering thermal constraints

    """
    try:
        # Check temperature-based limits first
        max_temp = _get_max_temperature()
        temp_limit = _get_thermal_limit_by_temperature(media_type, max_temp)
        if temp_limit is not None:
            return temp_limit

        # Check memory-based limits
        memory_limit = _get_thermal_limit_by_memory(media_type)
        if memory_limit is not None:
            return memory_limit

        # Return default thermal-safe limits

    except (OSError, AttributeError, ValueError):
        # If we can't monitor anything, be very conservative
        return VIDEO_FALLBACK_WORKERS if media_type == "video" else AUDIO_FALLBACK_WORKERS
    else:
        return VIDEO_DEFAULT_WORKERS if media_type == "video" else AUDIO_DEFAULT_WORKERS


def check_thermal_throttling(media_type: MediaType) -> None:
    """
    Check for thermal throttling during processing.

    Args:
        media_type: Type of media processing for appropriate thresholds

    """
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        threshold = 85 if media_type == "video" else 90

        if cpu_percent > threshold:
            LOG.warning("High CPU usage detected during %s processing: %.1f%%", media_type, cpu_percent)
            # Brief pause to allow cooling
            time.sleep(1.0)

    except (OSError, AttributeError):
        # Log this issue since it might indicate a real problem
        LOG.debug("Could not check thermal throttling for %s processing", media_type)
