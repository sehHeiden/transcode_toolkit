"""Utility CLI commands."""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core import ConfigManager

LOG = logging.getLogger(__name__)


class UtilityCommands:
    """Utility command handlers."""

    def __init__(self, config_manager: "ConfigManager"):
        self.config_manager = config_manager

    def add_subcommands(self, parser: argparse.ArgumentParser) -> None:
        """Add utility subcommands to parser."""
        subparsers = parser.add_subparsers(dest="util_command", help="Utility commands")

        # Cleanup command
        cleanup_parser = subparsers.add_parser("cleanup", help="Clean up backup files")
        cleanup_parser.add_argument("path", type=Path, help="Path to directory")
        cleanup_parser.add_argument(
            "--force", action="store_true", help="Force removal of all backup files"
        )
        cleanup_parser.add_argument(
            "--dry-run", action="store_true", help="Show what would be deleted"
        )

        # Info command
        subparsers.add_parser("info", help="Show configuration and system info")

    def handle_command(self, args: argparse.Namespace) -> int:
        """Handle utility command execution."""
        if not hasattr(args, "util_command") or args.util_command is None:
            LOG.error("No utility command specified")
            return 1

        if args.util_command == "cleanup":
            return self._handle_cleanup(args)
        elif args.util_command == "info":
            return self._handle_info(args)
        else:
            LOG.error(f"Unknown utility command: {args.util_command}")
            return 1

    def _handle_cleanup(self, args: argparse.Namespace) -> int:
        """Handle backup cleanup."""
        try:
            from core import BackupStrategy, FileManager

            if args.force:
                file_manager = FileManager(BackupStrategy.NEVER)
                if args.dry_run:
                    backup_files = list(args.path.rglob("*.bak"))
                    LOG.info(f"Would remove {len(backup_files)} backup files")
                    for backup_file in backup_files:
                        LOG.info(f"  {backup_file}")
                else:
                    count = file_manager.force_cleanup_all_backups(args.path)
                    LOG.info(f"Removed {count} backup files")
            else:
                file_manager = FileManager()
                if args.dry_run:
                    backup_files = list(args.path.rglob("*.bak"))
                    LOG.info(f"Would clean up {len(backup_files)} old backup files")
                else:
                    count = file_manager.cleanup_old_backups(args.path)
                    LOG.info(f"Cleaned up {count} old backup files")

            return 0

        except Exception as e:
            LOG.error(f"Backup cleanup failed: {e}")
            return 1

    def _handle_info(self, args: argparse.Namespace) -> int:
        """Handle info display."""
        try:
            import shutil
            from pathlib import Path

            print("=== Media Toolkit Configuration ===")

            config = self.config_manager.config

            print(f"Audio presets: {list(config.audio.presets.keys())}")
            print(f"Video presets: {list(config.video.presets.keys())}")
            print("Global settings:")
            print(f"  Workers: {config.global_.default_workers or 'auto'}")
            print(f"  Log level: {config.global_.log_level}")
            print(f"  Create backups: {config.global_.create_backups}")
            print(f"  Cleanup backups: {config.global_.cleanup_backups}")

            print("\n=== System Information ===")

            # Check for required executables
            executables = ["ffmpeg", "ffprobe"]
            for exe in executables:
                path = shutil.which(exe)
                status = "✓ Found" if path else "✗ Missing"
                print(f"{exe}: {status}")
                if path:
                    print(f"  Path: {path}")

            # Check config file
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
            config_status = "✓ Found" if config_path.exists() else "✗ Missing"
            print(f"Config file: {config_status}")
            if config_path.exists():
                print(f"  Path: {config_path}")

            return 0

        except Exception as e:
            LOG.error(f"Info display failed: {e}")
            return 1
