"""Test for the deprecated transcode module (compatibility testing)."""

import sys
from pathlib import Path

# Add src to Python path so imports work
sys.path.insert(0, "src")

from transcode_toolkit.core import unified_estimate as ue


def test_ffmpeg_cmd_cpu() -> None:
    """Test CPU FFmpeg command generation."""
    cmd = ue.ffmpeg_cmd(Path("in.mkv"), Path("tmp.mkv"), crf=24, gpu=False)
    assert "libx265" in cmd
    assert "hevc_nvenc" not in cmd


def test_ffmpeg_cmd_gpu() -> None:
    """Test GPU FFmpeg command generation."""
    cmd = ue.ffmpeg_cmd(Path("in.mkv"), Path("tmp.mkv"), crf=24, gpu=True)
    assert "hevc_nvenc" in cmd
