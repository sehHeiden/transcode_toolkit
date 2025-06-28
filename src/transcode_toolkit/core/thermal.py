"""Unified thermal management and worker count optimization for all processors."""

from __future__ import annotations

import logging
import time
from typing import Literal

import psutil

LOG = logging.getLogger(__name__)

MediaType = Literal["audio", "video"]


def get_thermal_safe_worker_count(
    configured_workers: int | None, 
    media_type: MediaType = "audio"
) -> int:
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
                f"Reducing configured workers from {configured_workers} to {safe_workers} "
                f"due to thermal/system load constraints for {media_type} processing"
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
            LOG.warning(f"High CPU load detected ({cpu_percent:.1f}%), reducing {media_type} workers to {max_workers}")

        # Apply media-specific cap
        max_workers = min(max_workers, max_cap)

        LOG.info(
            f"{media_type.title()} thermal-safe configuration: {physical_cores} physical cores, {logical_cores} logical cores, "
            f"CPU load: {cpu_percent:.1f}%. Using {max_workers} workers for thermal safety."
        )

        return max_workers

    except Exception as e:
        # Very conservative fallback if monitoring fails
        LOG.warning(f"Failed to detect system specs with psutil: {e}. Using conservative fallback of 1 worker.")
        return 1


def _get_thermal_limit(media_type: MediaType) -> int:
    """
    Determine thermal limits based on system capabilities and media type.

    Args:
        media_type: Type of media processing

    Returns:
        Maximum recommended workers considering thermal constraints
    """
    try:
        # Try to get temperature sensors (Linux systems mostly)
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if temps:
                max_temp = 0
                for entries in temps.values():
                    for entry in entries:
                        if entry.current and entry.current > max_temp:
                            max_temp = entry.current

                # Different temperature thresholds based on media type
                if media_type == "video":
                    # More conservative for video encoding
                    if max_temp > 65:  # Above 65°C, reduce workers for video
                        return 1
                    if max_temp > 55:  # Above 55°C, moderate reduction
                        return 2
                else:  # audio
                    # Less conservative for audio encoding
                    if max_temp > 70:  # Above 70°C, reduce workers
                        return 2
                    if max_temp > 60:  # Above 60°C, moderate reduction
                        return 3

        # Check available memory as another thermal/stability indicator
        memory = psutil.virtual_memory()
        memory_threshold = 75 if media_type == "video" else 80
        
        if memory.percent > memory_threshold:
            return 1 if media_type == "video" else 2

        # Default thermal-safe limits by media type
        return 2 if media_type == "video" else 4

    except Exception:
        # If we can't monitor anything, be very conservative
        return 1 if media_type == "video" else 2


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
            LOG.warning(f"High CPU usage detected during {media_type} processing: {cpu_percent:.1f}%")
            # Brief pause to allow cooling
            time.sleep(1.0)

    except Exception:
        pass  # Don't fail processing due to monitoring issues
