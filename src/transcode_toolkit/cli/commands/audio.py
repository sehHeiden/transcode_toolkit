"""Audio processing CLI commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

    from ...core import ConfigManager

LOG = logging.getLogger(__name__)


class AudioCommands:
    """Audio processing command handlers."""

    def __init__(self, config_manager: ConfigManager) -> None:
        """Initialize audio commands handler."""
        self.config_manager = config_manager

    def add_subcommands(self, parser: argparse.ArgumentParser) -> None:
        """Add audio subcommands to parser."""
        subparsers = parser.add_subparsers(dest="audio_command", help="Audio commands")

        # Transcode command
        transcode_parser = subparsers.add_parser("transcode", help="Transcode audio files")
        transcode_parser.add_argument("path", type=Path, help="Path to audio file or directory")
        transcode_parser.add_argument(
            "--preset", "-p", default="music", type=str.lower, help="Audio preset to use (case-insensitive)"
        )
        transcode_parser.add_argument(
            "--recursive",
            "-r",
            action="store_true",
            help="Process directories recursively",
        )
        transcode_parser.add_argument("--no-backups", action="store_true", help="Don't create backup files")

        # Estimate command
        estimate_parser = subparsers.add_parser("estimate", help="Estimate size savings")
        estimate_parser.add_argument("path", type=Path, help="Path to audio file or directory")
        estimate_parser.add_argument(
            "--preset",
            "-p",
            type=str.lower,
            help="Specific audio preset to analyze (case-insensitive, default: compare all)",
        )
        estimate_parser.add_argument(
            "--no-compare",
            action="store_true",
            help="Skip comparison and use specific preset only",
        )
        estimate_parser.add_argument("--csv", help="Save results to CSV file")

    def handle_command(self, args: argparse.Namespace) -> int:
        """Handle audio command execution."""
        if not hasattr(args, "audio_command") or args.audio_command is None:
            LOG.error("No audio command specified")
            return 1

        if args.audio_command == "transcode":
            return self._handle_transcode(args)
        if args.audio_command == "estimate":
            return self._handle_estimate(args)
        LOG.error("Unknown audio command: %s", args.audio_command)
        return 1

    def _handle_transcode(self, args: argparse.Namespace) -> int:
        """Handle audio transcoding."""
        try:
            from ...processors import AudioProcessor

            processor = AudioProcessor(self.config_manager)

            if args.path.is_file():
                result = processor.process_file(args.path, preset=args.preset)
                if result.status.value == "success":
                    LOG.info("Successfully processed %s", args.path)
                    return 0
                LOG.error("Failed to process %s: %s", args.path, result.message)
                return 1
            results = processor.process_directory(args.path, recursive=args.recursive, preset=args.preset)
            successful = [r for r in results if r.status.value == "success"]
            failed = [r for r in results if r.status.value == "error"]

            LOG.info("Processed %d/%d files successfully", len(successful), len(results))

            # Show failure table if there are failures
            if failed:
                from . import print_failure_table

                print_failure_table(failed, "audio")

            return 0 if len(successful) == len(results) else 1

        except Exception:
            LOG.exception("Audio transcoding failed")
            return 1

    def _handle_estimate(self, args: argparse.Namespace) -> int:
        """Handle audio size estimation."""
        try:
            from ...audio import estimate

            # Check if we're in verbose mode
            verbose_mode = getattr(args, "verbose", 0) > 0

            if not verbose_mode:
                print("🎵 Analyzing audio files for optimization opportunities...")
                print("💡 Use -v for detailed progress information")

            # Default to comparison mode unless --no-compare is used with a specific preset
            if args.no_compare and args.preset:
                # Single preset estimation
                config = self.config_manager.config
                preset_config = config.audio.presets.get(args.preset)
                if not preset_config:
                    LOG.error("Unknown preset: %s", args.preset)
                    return 1

                target_br = int(preset_config.bitrate.rstrip("k")) * 1000
                rows = estimate.analyse(args.path, target_br)
                estimate.print_summary(rows, preset=args.preset, csv_path=args.csv)
            else:
                # Compare all presets and show recommendations (default behavior)
                results = estimate.compare_presets(args.path)
                recommended = estimate.recommend_preset(results)
                estimate.print_comparison(results, recommended)

                if not verbose_mode:
                    print(f"\n🚀 To apply the recommended preset '{recommended}':")
                    print(f"   transcode-toolkit audio transcode {args.path} --preset {recommended}")

        except Exception:
            LOG.exception("Audio estimation failed")
            return 1
        else:
            return 0
