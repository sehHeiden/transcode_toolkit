"""video.estimate – rough H.265 size-saving estimator."""

from __future__ import annotations
import csv
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

BITRATE_LIMITS: Dict[int, int] = {
    480: 2_500_000,
    720: 4_000_000,
    1080: 8_000_000,
    1440: 16_000_000,
    2160: 35_000_000,
}
VIDEO_EXTS = {".mkv", ".mp4", ".mov", ".avi", ".wmv"}
LOG = logging.getLogger("video-est")


# ──────────────────────────────────────────────────────────────────────────────
def _height_category(h: int) -> int:
    for ref in sorted(BITRATE_LIMITS):
        if h <= ref:
            return ref
    return 2160


def _probe(path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_entries",
        "format=duration,size,bit_rate",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=height",
        str(path),
    ]
    data = json.loads(
        subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    )
    fmt, st = data["format"], data["streams"][0]
    return {
        "duration": float(fmt.get("duration", 0.0)),
        "height": int(st.get("height", 1080)),
        "size": int(fmt.get("size", path.stat().st_size)),
    }


def _estimate(meta: Dict[str, Any]) -> int:
    target_br = BITRATE_LIMITS[_height_category(meta["height"])]
    return int(meta["duration"] * target_br / 8)


def analyse(root: Path) -> List[Tuple[Path, int, int]]:
    rows: List[Tuple[Path, int, int]] = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        try:
            meta = _probe(p)
        except subprocess.CalledProcessError as exc:
            LOG.warning("ffprobe failed for %s: %s", p, exc)
            continue
        rows.append((p, meta["size"], _estimate(meta)))
    return rows


def print_summary(rows, *, csv_path: str | None = None) -> None:
    cur = sum(r[1] for r in rows)
    new = sum(r[2] for r in rows)
    diff = cur - new
    pct = 0 if cur == 0 else 100 * (diff / cur)
    print(f"Video files analysed : {len(rows):,}")
    print(f"Current size         : {cur / 2**30:.2f} GiB")
    print(f"Estimated HEVC       : {new / 2**30:.2f} GiB")
    print(f"Potential saving     : {diff / 2**20:.1f} MiB  ({pct:.1f} %)")

    if csv_path:
        csv_path_obj = Path(csv_path)
        csv_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with csv_path_obj.open("w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["file", "current_bytes", "estimated_bytes", "saving_bytes"])
            for p, c, e in rows:
                wr.writerow([p, c, e, c - e])
            wr.writerow(["TOTAL", cur, new, diff])
        print("CSV report →", csv_path_obj)
