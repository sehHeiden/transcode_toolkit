import sys
from pathlib import Path

# Add project root to path so we can import video module
sys.path.insert(0, str(Path(__file__).parent.parent))

from video import transcode as vt


def test_should_skip_detection() -> None:
    meta = {"codec_name": "hevc", "bit_rate": 3_000_000, "height": 1080}
    assert vt._should_skip(meta)


def test_ffmpeg_cmd_gpu() -> None:
    cmd = vt._ffmpeg_cmd(Path("in.mkv"), Path("tmp.mkv"), crf=24, gpu=True)
    assert "hevc_nvenc" in cmd
