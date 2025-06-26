"""Video processing CLI commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

    from ...core import ConfigManager

LOG = logging.getLogger(__name__)


class VideoCommands:
    """Video processing command handlers."""

    def __init__(self, config_manager: ConfigManager) -> None:
        """Initialize video commands handler."""
        self.config_manager = config_manager

    def add_subcommands(self, parser: argparse.ArgumentParser) -> None:
        """Add video subcommands to parser."""
        subparsers = parser.add_subparsers(dest="video_command", help="Video commands")

        # Transcode command
        transcode_parser = subparsers.add_parser("transcode", help="Transcode video files")
        transcode_parser.add_argument("path", type=Path, help="Path to video file or directory")
        transcode_parser.add_argument("--crf", "-c", type=int, default=24, help="Constant Rate Factor (quality)")
        transcode_parser.add_argument("--gpu", "-g", action="store_true", help="Use GPU acceleration")
        transcode_parser.add_argument(
            "--recursive",
            "-r",
            action="store_true",
            help="Process directories recursively",
        )
        transcode_parser.add_argument("--no-backups", action="store_true", help="Don't create backup files")

        # Estimate command
        estimate_parser = subparsers.add_parser("estimate", help="Estimate size savings")
        estimate_parser.add_argument("path", type=Path, help="Path to video file or directory")
        estimate_parser.add_argument("--csv", help="Save results to CSV file")

    def handle_command(self, args: argparse.Namespace) -> int:
        """Handle video command execution."""
        if not hasattr(args, "video_command") or args.video_command is None:
            LOG.error("No video command specified")
            return 1

        if args.video_command == "transcode":
            return self._handle_transcode(args)
        if args.video_command == "estimate":
            return self._handle_estimate(args)
        LOG.error("Unknown video command: %s", args.video_command)
        return 1

    def _handle_transcode(self, args: argparse.Namespace) -> int:
        """Handle video transcoding."""
        try:
            from ...processors import VideoProcessor

            processor = VideoProcessor(self.config_manager)

            if args.path.is_file():
                result = processor.process_file(args.path, crf=args.crf, gpu=args.gpu)
                if result.status.value == "success":
                    LOG.info("Successfully processed %s", args.path)
                    return 0
                LOG.error("Failed to process %s: %s", args.path, result.message)
                return 1
            results = processor.process_directory(args.path, recursive=args.recursive, crf=args.crf, gpu=args.gpu)
            successful = [r for r in results if r.status.value == "success"]
            LOG.info("Processed %d/%d files successfully", len(successful), len(results))
            return 0 if len(successful) == len(results) else 1

        except Exception:
            LOG.exception("Video transcoding failed")
            return 1

    def _handle_estimate(self, args: argparse.Namespace) -> int:
        """Handle video size estimation."""
        try:
            from ...video import estimate

            rows = estimate.analyse(args.path)
            estimate.print_summary(rows, csv_path=args.csv)
            return 0

        except Exception:
            LOG.exception("Video estimation failed")
            return 1
