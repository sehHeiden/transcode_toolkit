"""audio.estimate – Opus re-encode size estimator."""
import csv
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

TARGETS = {"music": 128_000, "audiobook": 48_000}   # bits/s
AUDIO_EXTS = {
    ".flac", ".wav", ".aiff", ".aif", ".alac", ".ape",
    ".m4a", ".aac", ".mp3", ".ogg", ".opus",
}
LOG = logging.getLogger("audio-est")


def _probe(path: Path) -> Dict[str, Any]:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_entries", "format=duration,bit_rate", str(path)]
    fmt = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)["format"]
    return {"duration": float(fmt.get("duration", 0)), "size": path.stat().st_size}


def _estimate(meta: Dict[str, Any], br: int) -> int:
    return int(meta["duration"] * br / 8)


def analyse(root: Path, target_br: int) -> List[Tuple[Path, int, int]]:
    rows = []
    for p in root.rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS:
            try:
                meta = _probe(p)
            except subprocess.CalledProcessError as exc:
                LOG.warning("ffprobe failed for %s: %s", p, exc)
                continue
            rows.append((p, meta["size"], _estimate(meta, target_br)))
    return rows


def print_summary(rows, *, preset: str, csv_path: str | None = None) -> None:
    cur = sum(r[1] for r in rows)
    new = sum(r[2] for r in rows)
    diff = cur - new
    pct = 0 if cur == 0 else 100 * diff / cur
    print(f"{preset} files analysed : {len(rows):,}")
    print(f"Current size            : {cur/2**30:.2f} GiB")
    print(f"Estimated Opus          : {new/2**30:.2f} GiB")
    print(f"Potential saving        : {diff/2**20:.1f} MiB ({pct:.1f} %)")

    if csv_path:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["file", "current", "estimated", "saving"])
            for p, c, e in rows:
                wr.writerow([p, c, e, c - e])
            wr.writerow(["TOTAL", cur, new, diff])
        print("CSV report →", csv_path)
