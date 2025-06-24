"""Video transcoding functionality."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Any, List

LOG = logging.getLogger(__name__)


def _should_skip(meta: Dict[str, Any]) -> bool:
    """Check if video should be skipped based on metadata.

    Skip if already using HEVC codec with reasonable bitrate.
    """
    codec = meta.get("codec_name", "").lower()
    bit_rate = meta.get("bit_rate", 0)
    height = meta.get("height", 1080)

    # Skip if already HEVC
    if codec in ("hevc", "h265"):
        # Define reasonable bitrate thresholds by resolution
        thresholds = {
            480: 2_500_000,
            720: 4_000_000,
            1080: 8_000_000,
            1440: 16_000_000,
            2160: 35_000_000,
        }

        # Find appropriate threshold for this resolution
        threshold = 35_000_000  # Default for very high res
        for res, thresh in sorted(thresholds.items()):
            if height <= res:
                threshold = thresh
                break

        # Skip if bitrate is already reasonable for resolution
        if bit_rate <= threshold:
            return True

    return False


def _ffmpeg_cmd(
    input_path: Path, output_path: Path, *, crf: int = 23, gpu: bool = False
) -> List[str]:
    """Generate FFmpeg command for video transcoding.

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


def transcode_video(
    input_path: Path, output_path: Path, *, crf: int = 23, gpu: bool = False
) -> bool:
    """Transcode a video file to HEVC.

    Args:
        input_path: Source video file
        output_path: Output video file
        crf: Constant Rate Factor (quality setting)
        gpu: Whether to use GPU acceleration

    Returns:
        True if successful, False otherwise
    """
    import subprocess

    cmd = _ffmpeg_cmd(input_path, output_path, crf=crf, gpu=gpu)

    try:
        LOG.info(f"Transcoding {input_path} -> {output_path}")
        LOG.debug(f"FFmpeg command: {' '.join(cmd)}")

        subprocess.run(cmd, capture_output=True, text=True, check=True)

        LOG.info(f"Successfully transcoded {input_path}")
        return True

    except subprocess.CalledProcessError as e:
        LOG.error(f"FFmpeg failed for {input_path}: {e}")
        LOG.error(f"FFmpeg stderr: {e.stderr}")
        return False
    except Exception as e:
        LOG.error(f"Unexpected error transcoding {input_path}: {e}")
        return False
