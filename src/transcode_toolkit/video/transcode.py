"""Video transcoding functionality."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

LOG = logging.getLogger(__name__)


def ffmpeg_cmd(input_path: Path, output_path: Path, *, crf: int = 23, gpu: bool = False) -> list[str]:
    """
    Generate FFmpeg command for video transcoding.

    Args:
        input_path: Source video file
        output_path: Output video file
        crf: Constant Rate Factor (quality setting, lower = better quality)
        gpu: Whether to use GPU acceleration (NVENC)

    Returns:
        FFmpeg command as list of strings

    """
    cmd = ["ffmpeg", "-i", str(input_path)]

    if gpu:
        # Use NVIDIA hardware encoding
        cmd.extend(
            [
                "-c:v",
                "hevc_nvenc",
                "-preset",
                "fast",
                "-crf",
                str(crf),
            ]
        )
    else:
        # Use software encoding
        cmd.extend(
            [
                "-c:v",
                "libx265",
                "-preset",
                "medium",
                "-crf",
                str(crf),
            ]
        )

    # Copy audio without re-encoding
    cmd.extend(["-c:a", "copy"])

    # Add output file
    cmd.append(str(output_path))

    return cmd
