"""Main CLI interface for the media toolkit."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from core import ConfigManager, ProcessingOptions, with_config_overrides

from .commands import AudioCommands, UtilityCommands, VideoCommands


class MediaToolkitCLI:
    """Main CLI interface with enhanced architecture."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_manager = ConfigManager(config_path)
        self.audio_commands = AudioCommands(self.config_manager)
        self.video_commands = VideoCommands(self.config_manager)
        self.utility_commands = UtilityCommands(self.config_manager)

    @staticmethod
    def setup_logging(verbosity: int) -> None:
        """Setup logging based on verbosity level."""
        level_map = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG,
        }

        level = level_map.get(verbosity, logging.DEBUG)

        # Setup enhanced logging format
        log_format = "%(levelname)s: %(name)s: %(message)s" if verbosity >= 2 else "%(levelname)s: %(message)s"

        logging.basicConfig(level=level, format=log_format, handlers=[logging.StreamHandler(sys.stderr)])

        # Set FFmpeg logs to higher level to reduce noise
        if verbosity < 2:
            logging.getLogger("core.ffmpeg").setLevel(logging.WARNING)

    def build_parser(self) -> argparse.ArgumentParser:
        """Build the argument parser."""
        parser = argparse.ArgumentParser(
            prog="media-toolkit",
            description="Enhanced media transcoding toolkit",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Audio transcoding
  media-toolkit audio transcode /path/to/audio --preset audiobook

  # Estimate sizes before transcoding
  media-toolkit audio estimate /path/to/audio --compare

  # Video transcoding with GPU acceleration
  media-toolkit video transcode /path/to/video --gpu --crf 22

  # Clean up backup files
  media-toolkit utils cleanup /path/to/media --force
            """,
        )

        # Global options
        parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            default=0,
            help="Increase verbosity (-v for info, -vv for debug)",
        )

        parser.add_argument("--config", type=Path, help="Path to configuration file")

        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be done without actually doing it",
        )

        # Subcommands
        subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

        # Add audio commands
        audio_parser = subparsers.add_parser("audio", help="Audio processing commands")
        self.audio_commands.add_subcommands(audio_parser)

        # Add video commands
        video_parser = subparsers.add_parser("video", help="Video processing commands")
        self.video_commands.add_subcommands(video_parser)

        # Add utility commands
        utils_parser = subparsers.add_parser("utils", help="Utility commands")
        self.utility_commands.add_subcommands(utils_parser)

        return parser

    @staticmethod
    def create_processing_options(args: argparse.Namespace) -> ProcessingOptions:
        """Create processing options from CLI arguments."""
        return ProcessingOptions(
            create_backups=not getattr(args, "no_backups", False),  # Invert no_backups
            cleanup_backups=getattr(args, "cleanup_backups", None),
            workers=getattr(args, "workers", None),
            timeout=getattr(args, "timeout", None),
            dry_run=getattr(args, "dry_run", False),
            verbose=getattr(args, "verbose", 0) > 0,
        )

    def run(self, args: list[str] | None = None) -> int:
        """Run the CLI with given arguments."""
        parser = self.build_parser()
        parsed_args = parser.parse_args(args)

        # Setup logging
        self.setup_logging(parsed_args.verbose)

        # Update config manager if custom config provided
        if getattr(parsed_args, "config", None):
            self.config_manager = ConfigManager(parsed_args.config)
            # Update command instances with new config
            self.audio_commands.config_manager = self.config_manager
            self.video_commands.config_manager = self.config_manager
            self.utility_commands.config_manager = self.config_manager

        # Create processing options and apply them
        processing_options = self.create_processing_options(parsed_args)

        try:
            # Use configuration context for temporary overrides
            with with_config_overrides(self.config_manager) as config_mgr:
                config_mgr.apply_processing_options(processing_options)

                # Route to appropriate command handler
                if parsed_args.command == "audio":
                    return self.audio_commands.handle_command(parsed_args)
                if parsed_args.command == "video":
                    return self.video_commands.handle_command(parsed_args)
                if parsed_args.command == "utils":
                    return self.utility_commands.handle_command(parsed_args)
                parser.error(f"Unknown command: {parsed_args.command}")

        except KeyboardInterrupt:
            logging.getLogger(__name__).info("Operation cancelled by user")
            return 130  # Standard exit code for SIGINT
        except Exception as e:
            logging.getLogger(__name__).exception(f"Unexpected error: {e}")
            if parsed_args.verbose >= 2:
                import traceback

                traceback.print_exc()
            return 1

        return 0


def main() -> int:
    """Entry point for the CLI."""
    cli = MediaToolkitCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
