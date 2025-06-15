#!/usr/bin/env python3
"""blur_quality.py – ermittelt automatisch Ziel‑Auflösung und CRF je Video‑Datei

Motivation
----------
* Alte Quellen (VHS‑Rip, SD‑TV, frühe Handys) liefern nominell grosse Frames,
  enthalten aber durch Blur/Rauschen wenig echte Details.
* Je unschärfer das Bild, desto weniger Pixel und Bitrate werden benötigt, um
  visuell verlustfrei zu bleiben.

Das Skript …
* zieht stichprobenartig Frames,
* bewertet Schärfe mit *Laplacian‑Varianz* und *Edge‑Density*,
* kombiniert das mit der Original‑Auflösung aus ffprobe,
* schlägt eine **Zielhöhe** (360 p – 1080 p) und einen **CRF** (HEVC) vor.

CLI
----
    python blur_quality.py  <video_or_folder>

Gibt pro Datei ein JSON‑Objekt aus:
    {"file": "…", "orig_h": 1080, "target_h": 720, "crf": 28,
     "lap_var": 480.3, "edge": 0.027}

Abhängigkeiten
--------------
* ffmpeg / ffprobe im PATH
* opencv‑python >= 4.x (wird kontrolliert importiert)
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError as exc:  # pragma: no cover
    sys.exit("opencv‑python und numpy werden benötigt:  pip install opencv-python numpy")

_log = logging.getLogger("blurqual")

# ──────────────────────────────────────────────────────────────────────────────
#  Parameter (ggf. anpassen)
# ──────────────────────────────────────────────────────────────────────────────
EDGE_MIN = 0.03        # < 3 % Kanten = unscharf
LAPLACIAN_VAR_MIN = 600.0  # < 600 → unscharf (8‑bit, 1080 p)
TARGET_STEPS = [360, 480, 576, 720, 1080]
CRF_BASE = 24          # bei scharfen Inhalten
CRF_BLUR_ADD = 4       # unscharfe Inhalte: CRF += 4


# ──────────────────────────────────────────────────────────────────────────────
#  Hilfsfunktionen
# ──────────────────────────────────────────────────────────────────────────────

def ffprobe_meta(path: Path) -> Dict[str, Any]:
    """Gibt {"height": …, "duration": …} zurück."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-select_streams", "v:0",
        "-show_entries", "stream=height,avg_frame_rate",  # Höhe für spätere Logik
        "-show_entries", "format=duration",
        str(path),
    ]
    data = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    stream = data["streams"][0]
    return {
        "height": int(stream.get("height", 1080)),
        "duration": float(data["format"].get("duration", 0.0)),
    }


def sample_frames(path: Path, n: int = 12) -> List["np.ndarray"]:
    """Liest *n* gleichverteilte Frames (BGR) mit OpenCV."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Kann {path} nicht öffnen")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        raise RuntimeError("Keine Frames erkannt")
    step = max(total // n, 1)
    frames = []
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
        if len(frames) >= n:
            break
    cap.release()
    return frames


def sharpness_metrics(frame: "np.ndarray") -> Tuple[float, float]:
    """Berechnet (Laplacian‑Varianz, Edge‑Density)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = lap.var()
    edge_density = float(cv2.countNonZero(cv2.convertScaleAbs(lap))) / lap.size
    return var, edge_density


def decide_target(meta: Dict[str, Any], var_m: float, edge_m: float) -> Tuple[int, int]:
    """Gibt (target_height, crf) zurück."""
    blur = var_m < LAPLACIAN_VAR_MIN or edge_m < EDGE_MIN
    # Zielhöhe bestimmen – kleinste Stufe, die >= 0.9 × Original und nicht unter blur‑Schwelle liegt
    orig_h = meta["height"]
    target_h = orig_h
    for h in TARGET_STEPS:
        if h >= orig_h:
            target_h = orig_h  # nichts ändern, wir sind schon darunter
            break
        if blur and h <= orig_h:  # unscharf: wir dürfen herunterskalieren
            target_h = h
            break
    crf = CRF_BASE + (CRF_BLUR_ADD if blur else 0)
    return target_h, crf


# ──────────────────────────────────────────────────────────────────────────────
#  Hauptlogik pro Datei
# ──────────────────────────────────────────────────────────────────────────────

def analyse_file(path: Path, n_frames: int = 12) -> Dict[str, Any]:
    meta = ffprobe_meta(path)
    frames = sample_frames(path, n_frames)
    vars_, edges_ = zip(*(sharpness_metrics(f) for f in frames))
    var_m, edge_m = statistics.mean(vars_), statistics.mean(edges_)
    target_h, crf = decide_target(meta, var_m, edge_m)
    return {
        "file": str(path),
        "orig_h": meta["height"],
        "target_h": target_h,
        "crf": crf,
        "lap_var": round(var_m, 1),
        "edge": round(edge_m, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def run_cli() -> None:
    ap = argparse.ArgumentParser(description="Ermittelt Zielhöhe & CRF anhand von Blur‑Metriken.")
    ap.add_argument("path", type=Path, help="Video‑Datei oder Ordner")
    ap.add_argument("--json", type=Path, help="Ergebnis als JSON‑Zeile (NDJSON) schreiben")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")

    files: List[Path] = []
    p = args.path.expanduser()
    if p.is_dir():
        files = [f for f in p.rglob("*") if f.suffix.lower() in {".mkv", ".mp4", ".mov", ".avi", ".wmv"}]
    elif p.is_file():
        files = [p]
    else:
        sys.exit("Pfad nicht gefunden")

    out_fh = args.json.open("w") if args.json else None
    for f in files:
        try:
            res = analyse_file(f)
        except Exception as e:
            _log.warning("Überspringe %s: %s", f, e)
            continue
        line = json.dumps(res, ensure_ascii=False)
        print(line)
        if out_fh:
            out_fh.write(line + "\n")
    if out_fh:
        out_fh.close()

if __name__ == "__main__":
    run_cli()
