#!/usr/bin/env python3
"""audio_quality.py -- propose Opus settings per audio file

This script inspects each given audio file (or every file in a folder),
classifies it as *music* or *audiobook/speech*, and prints a JSON line with
sensible Opus encoding parameters.

Heuristics
==========
1. Channels:   mono  -> likely speech,  stereo+ -> music
2. Bandwidth:  90th‑percentile < 8 kHz -> speech
3. Bit‑rate:   < 96 kb/s **and** mono  -> speech

Targets
=======
* music      -> Opus ~128 kb/s VBR,   48 kHz, stereo
* audiobook  -> Opus  48  kb/s speech, 24 kHz, mono

CLI
===
    python audio_quality.py PATH [--json out.ndjson] [-v]

The script prints **one NDJSON line per file** so that the output can be piped
into further tooling (e.g. a batch ffmpeg wrapper).
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AUDIO_EXTS = {
    ".flac", ".wav", ".aiff", ".aif", ".alac", ".ape",
    ".m4a", ".aac", ".mp3", ".ogg", ".opus",
}

MUSIC_OPUS_BITRATE = 128_000  # bits/s (Opus quality ~5)
SPEECH_OPUS_BITRATE = 48_000  # bits/s (application=speech)

LOG = logging.getLogger("audioqual")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def probe_basic(path: Path) -> Dict[str, Any]:
    """Return basic stream info via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_entries", "stream=channels,sample_rate,bit_rate",
        str(path),
    ]
    data = json.loads(
        subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    )
    st = data["streams"][0]
    return {
        "channels": int(st.get("channels", 2)),
        "sample_rate": int(st.get("sample_rate", 48_000)),
        "bit_rate": int(st.get("bit_rate", 0)),
    }


def spectral_bandwidth(path: Path) -> float:
    """Return 90‑percentile bandwidth (kHz) using ffmpeg astats."""
    cmd = [
        "ffmpeg", "-v", "quiet", "-i", str(path),
        "-af", "astats=metadata=1:reset=1,ametadata=print:key=Bandwidth",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    bw_hz: List[float] = []
    for line in proc.stderr.splitlines():
        if "Bandwidth=" in line:
            try:
                bw_hz.append(float(line.rsplit("=", 1)[-1]))
            except ValueError:
                continue
    if not bw_hz:
        return 0.0
    return statistics.quantiles(bw_hz, n=10)[-1] / 1000  # kHz


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(path: Path) -> Dict[str, Any]:
    meta = probe_basic(path)
    bw90 = spectral_bandwidth(path)

    is_speech = (
        meta["channels"] == 1
        or bw90 < 8.0
        or (meta["channels"] == 1 and meta["bit_rate"] < 96_000)
    )

    if is_speech:
        return {
            "file": str(path),
            "mode": "audiobook",
            "target_bitrate": SPEECH_OPUS_BITRATE,
            "target_sr": 24_000,
            "channels": 1,
        }

    return {
        "file": str(path),
        "mode": "music",
        "target_bitrate": MUSIC_OPUS_BITRATE,
        "target_sr": 48_000,
        "channels": 2,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_cli() -> None:
    ap = argparse.ArgumentParser(
        description="Analyse audio files and propose Opus target parameters."
    )
    ap.add_argument("path", type=Path, help="File or directory")
    ap.add_argument("--json", type=Path, help="Write NDJSON output to file")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    target = args.path.expanduser()
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = [f for f in target.rglob("*") if f.suffix.lower() in AUDIO_EXTS]
    else:
        raise SystemExit("Path not found: " + str(target))

    out_handle = args.json.open("w") if args.json else None
    try:
        for f in files:
            try:
                result = classify(f)
            except Exception as exc:
                LOG.warning("Skip %s: %s", f, exc)
                continue
            line = json.dumps(result, ensure_ascii=False)
            print(line)
            if out_handle:
                out_handle.write(line + "\n")
    finally:
        if out_handle:
            out_handle.close()


if __name__ == "__main__":
    run_cli()
