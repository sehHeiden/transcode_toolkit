#!/usr/bin/env python3
"""media_toolkit.py – single CLI for everything (video + audio)."""

from __future__ import annotations
import argparse
import importlib
import logging
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


# ────────────────────────── dispatch functions ───────────────────────────────
def _video_estimate(ns: argparse.Namespace) -> None:
    from video import estimate as ve
    rows = ve.analyse(Path(ns.path).expanduser())
    ve.print_summary(rows, csv_path=ns.csv)


def _video_transcode(ns: argparse.Namespace) -> None:
    from video import transcode as vt
    vt.process_directory(Path(ns.path).expanduser(),
                         crf=ns.crf, workers=ns.workers, gpu=ns.gpu)


def _video_blur(ns: argparse.Namespace) -> None:
    # placeholder for your later blur implementation
    from video import blur_quality as vb
    vb.cli(Path(ns.path).expanduser(), json_path=ns.json, verbose=ns.verbose)


def _audio_estimate(ns: argparse.Namespace) -> None:
    from audio import estimate as ae
    rows = ae.analyse(Path(ns.path).expanduser(), ae.TARGETS[ns.mode])
    ae.print_summary(rows, preset=ns.mode, csv_path=ns.csv)


def _audio_quality(ns: argparse.Namespace) -> None:
    from audio import opus_quality as aq
    aq.cli(Path(ns.path).expanduser(), json_path=ns.json, verbose=ns.verbose)


# ────────────────────────────── CLI builder ───────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="media-toolkit", description="Unified transcoding helper")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v info | -vv debug")

    sub = p.add_subparsers(dest="group", required=True)

    # video commands
    vid = sub.add_parser("video")
    vs = vid.add_subparsers(dest="action", required=True)

    ve = vs.add_parser("estimate")
    ve.add_argument("path")
    ve.add_argument("--csv")
    ve.set_defaults(func=_video_estimate)

    vt = vs.add_parser("transcode")
    vt.add_argument("path")
    vt.add_argument("--crf", type=int, default=24)
    vt.add_argument("--gpu", action="store_true")
    vt.add_argument("--workers", type=int)
    vt.set_defaults(func=_video_transcode)

    vb = vs.add_parser("blur")
    vb.add_argument("path")
    vb.add_argument("--json")
    vb.set_defaults(func=_video_blur)

    # audio commands
    aud = sub.add_parser("audio")
    asub = aud.add_subparsers(dest="action", required=True)

    ae = asub.add_parser("estimate")
    ae.add_argument("path")
    ae.add_argument("--mode", choices=["music", "audiobook"], default="music")
    ae.add_argument("--csv")
    ae.set_defaults(func=_audio_estimate)

    aq = asub.add_parser("quality")
    aq.add_argument("path")
    aq.add_argument("--json")
    aq.set_defaults(func=_audio_quality)

    return p


def main() -> None:
    ns = _build_parser().parse_args()
    _setup_logging(ns.verbose)
    ns.func(ns)


if __name__ == "__main__":
    main()

