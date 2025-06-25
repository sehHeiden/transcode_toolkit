from pathlib import Path

from src.transcode_toolkit.video import transcode as vt


def test_should_skip_detection() -> None:
    meta = {"codec_name": "hevc", "bit_rate": 3_000_000, "height": 1080}
    assert vt._should_skip(meta)


def test_ffmpeg_cmd_gpu() -> None:
    cmd = vt._ffmpeg_cmd(Path("in.mkv"), Path("tmp.mkv"), crf=24, gpu=True)
    assert "hevc_nvenc" in cmd
