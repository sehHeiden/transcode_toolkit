#!/usr/bin/env python3
"""media_toolkit.py – unified CLI front‑end for the whole project

Usage (sub‑commands)
--------------------
    media‑toolkit video estimate  PATH [--csv out.csv]  [-v]
    media‑toolkit video transcode PATH [--gpu] [--workers N] [-v]
    media‑toolkit video blur      PATH [--json out.ndjson] [-v]

    media‑toolkit audio estimate  PATH --mode {music,audiobook} [--csv out.csv]
    media‑toolkit audio quality   PATH [--json out.ndjson]

The script is a thin wrapper that delegates the heavy work to the modules
already present in this repository (transcode_estimate.py, transcode.py,
blur_quality.py, audio_estimate.py, audio_quality.py).

Advantages
==========
* **Single entry‑point** for end‑users (easy Nuitka packing).
* Common logging style and argument validation.
* Keeps individual modules small and testable.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

LOG = logging.getLogger("toolkit")


def _common_logging(verbose: int) -> None:
    level = logging.WARNING if verbose == 0 else logging.INFO if verbose == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _print_summary(label: str, data: dict[str, str]) -> None:
    print(f"\n=== {label} ===")
    for k, v in data.items():
        print(f"{k:<15}: {v}")


# ---------------------------------------------------------------------------
# Entrypoints for sub‑commands (thin shims calling existing modules)
# ---------------------------------------------------------------------------

def _video_estimate(args: argparse.Namespace) -> None:
    mod = importlib.import_module("transcode_estimate")
    rows = mod.analyse(Path(args.path).expanduser())
    total_current = sum(r[1] for r in rows)
    total_new = sum(r[2] for r in rows)
    perc = 0 if total_current == 0 else 100 * (1 - total_new / total_current)
    _print_summary("VIDEO ESTIMATE", {
        "files": f"{len(rows):,}",
        "current": f"{total_current/2**30:.2f} GiB",
        "estimated": f"{total_new/2**30:.2f} GiB",
        "saving": f"{perc:.1f} %",
    })
    if args.csv:
        mod.csv_output(rows, Path(args.csv))  # helper function you can add


def _video_transcode(args: argparse.Namespace) -> None:
    mod = importlib.import_module("transcode")
    mod.process_directory(Path(args.path).expanduser(), crf=args.crf, workers=args.workers, gpu=args.gpu)


def _video_blur(args: argparse.Namespace) -> None:
    mod = importlib.import_module("blur_quality")
    mod.run_cli_from_parent(args)  # expose wrapper or re‑call analyse_file


def _audio_estimate(args: argparse.Namespace) -> None:
    mod = importlib.import_module("audio_estimate")
    target_br = mod.TARGETS[args.mode]
    rows = mod.analyse(Path(args.path).expanduser(), target_br)
    total_current = sum(r[1] for r in rows)
    total_new = sum(r[2] for r in rows)
    perc = 0 if total_current == 0 else 100 * (1 - total_new / total_current)
    _print_summary("AUDIO ESTIMATE", {
        "preset": args.mode,
        "files": f"{len(rows):,}",
        "current": f"{total_current/2**30:.2f} GiB",
        "estimated": f"{total_new/2**30:.2f} GiB",
        "saving": f"{perc:.1f} %",
    })


def _audio_quality(args: argparse.Namespace) -> None:
    mod = importlib.import_module("audio_quality")
    # direct reuse of its CLI helper
    mod.run_cli_from_parent(args)

# ---------------------------------------------------------------------------
# Build the CLI hierarchy
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="media‑toolkit", description="Unified video/audio transcoding helper")
    p.add_argument("-v", action="count", default=0, help="-v = info, -vv = debug")

    sub = p.add_subparsers(dest="section", required=True)

    # ── video group ───────────────────────────────────────────────────────────
    vid = sub.add_parser("video", help="Video‑related commands")
    vid_sub = vid.add_subparsers(dest="action", required=True)

    ve = vid_sub.add_parser("estimate", help="Estimate size savings")
    ve.add_argument("path")
    ve.add_argument("--csv")
    ve.set_defaults(func=_video_estimate)

    vt = vid_sub.add_parser("transcode", help="Actual transcoding")
    vt.add_argument("path")
    vt.add_argument("--crf", type=int, default=24)
    vt.add_argument("--gpu", action="store_true")
    vt.add_argument("--workers", type=int)
    vt.set_defaults(func=_video_transcode)

    vb = vid_sub.add_parser("blur", help="Analyse blur / propose resolution & CRF")
    vb.add_argument("path")
    vb.add_argument("--json")
    vb.set_defaults(func=_video_blur)

    # ── audio group ───────────────────────────────────────────────────────────
    aud = sub.add_parser("audio", help="Audio‑related commands")
    aud_sub = aud.add_subparsers(dest="action", required=True)

    ae = aud_sub.add_parser("estimate", help="Estimate audio savings")
    ae.add_argument("path")
    ae.add_argument("--mode", choices=["music", "audiobook"], default="music")
    ae.add_argument("--csv")
    ae.set_defaults(func=_audio_estimate)

    aq = aud_sub.add_parser("quality", help="Analyse & select Opus parameters")
    aq.add_argument("path")
    aq.add_argument("--json")
    aq.set_defaults(func=_audio_quality)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _common_logging(args.v)
    args.func(args)


if __name__ == "__main__":
    main()
