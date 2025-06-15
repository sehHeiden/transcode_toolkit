"""video.transcode – batch H.265/HEVC transcoder."""
from __future__ import annotations
import json
import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any

VIDEO_EXTS = {".mkv", ".mp4", ".mov", ".avi", ".wmv"}
DEFAULT_MARGIN = 1.10          # allow 10 % head-room
SIZE_KEEP_RATIO = 0.95         # keep re-encode only if ≥ 5 % smaller
LOG = logging.getLogger("transcode")

# ──────────────────────────────────────────────────────────────────────────────
def _probe(path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,bit_rate",
        str(path),
    ]
    data = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    if not data["streams"]:
        raise RuntimeError("no video stream found")
    return data["streams"][0]


def _should_skip(meta: Dict[str, Any]) -> bool:
    codec = meta.get("codec_name")
    bitrate = meta.get("bit_rate")
    return codec in {"hevc", "av1"} and bitrate and int(bitrate) < 8_000_000


def _ffmpeg_cmd(src: Path, tmp: Path, *, crf: int, gpu: bool) -> list[str]:
    cmd = ["ffmpeg", "-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a?"]
    if gpu:
        cmd += ["-c:v", "hevc_nvenc", "-preset", "p5", "-cq", str(crf)]
    else:
        cmd += ["-c:v", "libx265", "-preset", "slow", "-x265-params", f"crf={crf}"]
    cmd += ["-c:a", "copy", str(tmp)]
    return cmd


def _transcode_one(path: Path, *, crf: int, gpu: bool) -> None:
    try:
        meta = _probe(path)
    except Exception as exc:
        LOG.warning("probe failed for %s: %s", path, exc)
        return

    if _should_skip(meta):
        LOG.info("skip %s – already efficient", path)
        return

    tmp = path.with_suffix(path.suffix + ".tmp.mkv")
    if subprocess.run(_ffmpeg_cmd(path, tmp, crf=crf, gpu=gpu)).returncode != 0:
        LOG.error("ffmpeg failed for %s", path)
        tmp.unlink(missing_ok=True)
        return

    old, new = path.stat().st_size, tmp.stat().st_size
    if new < SIZE_KEEP_RATIO * old:
        shutil.move(path, path.with_suffix(path.suffix + ".bak"))
        shutil.move(tmp, path)
        LOG.info("replaced %s (%.1f → %.1f MiB)", path, old / 2**20, new / 2**20)
    else:
        tmp.unlink()
        LOG.info("discarded %s (no size gain)", path)


def process_directory(root: Path, *, crf: int = 24, workers: int | None = None, gpu: bool = False) -> None:
    files = [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    LOG.info("found %d video files", len(files))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_transcode_one, p, crf=crf, gpu=gpu): p for p in files}
        for fut in as_completed(futs):
            fut.result()       # propagate exceptions
