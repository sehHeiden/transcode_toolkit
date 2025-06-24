"""Audio processing CLI commands."""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core import ConfigManager

LOG = logging.getLogger(__name__)


class AudioCommands:
    """Audio processing command handlers."""

    def __init__(self, config_manager: "ConfigManager"):
        self.config_manager = config_manager

    def add_subcommands(self, parser: argparse.ArgumentParser) -> None:
        """Add audio subcommands to parser."""
        subparsers = parser.add_subparsers(dest="audio_command", help="Audio commands")

        # Transcode command
        transcode_parser = subparsers.add_parser(
            "transcode", help="Transcode audio files"
        )
        transcode_parser.add_argument(
            "path", type=Path, help="Path to audio file or directory"
        )
        transcode_parser.add_argument(
            "--preset", default="music", help="Audio preset to use"
        )
        transcode_parser.add_argument(
            "--recursive",
            "-r",
            action="store_true",
            help="Process directories recursively",
        )
        transcode_parser.add_argument(
            "--no-backups", action="store_true", help="Don't create backup files"
        )

        # Estimate command
        estimate_parser = subparsers.add_parser(
            "estimate", help="Estimate size savings"
        )
        estimate_parser.add_argument(
            "path", type=Path, help="Path to audio file or directory"
        )
        estimate_parser.add_argument(
            "--preset", help="Specific audio preset to analyze (default: compare all)"
        )
        estimate_parser.add_argument(
            "--no-compare", action="store_true", help="Skip comparison and use specific preset only"
        )
        estimate_parser.add_argument("--csv", help="Save results to CSV file")

    def handle_command(self, args: argparse.Namespace) -> int:
        """Handle audio command execution."""
        if not hasattr(args, "audio_command") or args.audio_command is None:
            LOG.error("No audio command specified")
            return 1

        if args.audio_command == "transcode":
            return self._handle_transcode(args)
        elif args.audio_command == "estimate":
            return self._handle_estimate(args)
        else:
            LOG.error(f"Unknown audio command: {args.audio_command}")
            return 1

    def _handle_transcode(self, args: argparse.Namespace) -> int:
        """Handle audio transcoding."""
        try:
            from processors import AudioProcessor

            processor = AudioProcessor(self.config_manager)

            if args.path.is_file():
                result = processor.process_file(args.path, preset=args.preset)
                if result.status.value == "success":
                    LOG.info(f"Successfully processed {args.path}")
                    return 0
                else:
                    LOG.error(f"Failed to process {args.path}: {result.message}")
                    return 1
            else:
                results = processor.process_directory(
                    args.path, recursive=args.recursive, preset=args.preset
                )
                successful = [r for r in results if r.status.value == "success"]
                LOG.info(
                    f"Processed {len(successful)}/{len(results)} files successfully"
                )
                return 0 if len(successful) == len(results) else 1

        except Exception as e:
            LOG.error(f"Audio transcoding failed: {e}")
            return 1

    def _handle_estimate(self, args: argparse.Namespace) -> int:
        """Handle audio size estimation."""
        try:
            from audio import estimate

            # Default to comparison mode unless --no-compare is used with a specific preset
            if args.no_compare and args.preset:
                # Single preset estimation
                config = self.config_manager.config
                preset_config = config.audio.presets.get(args.preset)
                if not preset_config:
                    LOG.error(f"Unknown preset: {args.preset}")
                    return 1

                target_br = int(preset_config.bitrate.rstrip("k")) * 1000
                rows = estimate.analyse(args.path, target_br)
                estimate.print_summary(rows, preset=args.preset, csv_path=args.csv)
            else:
                # Compare all presets and show recommendations (default behavior)
                results = estimate.compare_presets(args.path)
                recommended = estimate.recommend_preset(results)
                estimate.print_comparison(results, recommended)
                
                # Show usage tip
                print(f"\n💡 To convert with recommended settings:")
                print(f"   python main.py audio transcode '{args.path}' --preset {recommended}")

            return 0

        except Exception as e:
            LOG.error(f"Audio estimation failed: {e}")
            return 1
