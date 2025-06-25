"""Utility CLI commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

    from ...core import ConfigManager

LOG = logging.getLogger(__name__)


class UtilityCommands:
    """Utility command handlers."""

    def __init__(self, config_manager: ConfigManager) -> None:
        self.config_manager = config_manager

    def add_subcommands(self, parser: argparse.ArgumentParser) -> None:
        """Add utility subcommands to parser."""
        subparsers = parser.add_subparsers(dest="util_command", help="Utility commands")

        # Cleanup command
        cleanup_parser = subparsers.add_parser("cleanup", help="Clean up backup files")
        cleanup_parser.add_argument("path", type=Path, help="Path to directory")
        cleanup_parser.add_argument("--force", "-f", action="store_true", help="Force removal of all backup files")
        cleanup_parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be deleted")

        # Info command
        subparsers.add_parser("info", help="Show configuration and system info")

        # Duplicate detection command
        duplicates_parser = subparsers.add_parser("duplicates", help="Find duplicate files")
        duplicates_parser.add_argument("paths", nargs="+", type=Path, help="Paths to search for duplicates")
        duplicates_parser.add_argument("--extensions", nargs="*", help="File extensions to include (e.g., .mp3 .flac)")
        duplicates_parser.add_argument(
            "--workers", "-w", type=int, help="Number of parallel workers for hash calculation"
        )
        duplicates_parser.add_argument("--summary-only", "-s", action="store_true", help="Show only summary statistics")
        duplicates_parser.add_argument("--output", "-o", type=Path, help="Save detailed results to file")

    def handle_command(self, args: argparse.Namespace) -> int:
        """Handle utility command execution."""
        if not hasattr(args, "util_command") or args.util_command is None:
            LOG.error("No utility command specified")
            return 1

        if args.util_command == "cleanup":
            return self._handle_cleanup(args)
        if args.util_command == "info":
            return self._handle_info(args)
        if args.util_command == "duplicates":
            return self._handle_duplicates(args)
        LOG.error(f"Unknown utility command: {args.util_command}")
        return 1

    def _handle_cleanup(self, args: argparse.Namespace) -> int:
        """Handle backup cleanup."""
        try:
            if not args.force:
                LOG.info("Note: Only force cleanup is available with current backup strategy.")
                LOG.info("Use --force to remove all .bak files in the directory.")
                return 0

            # Find all backup files
            backup_files = list(args.path.rglob("*.bak"))

            if args.dry_run:
                LOG.info(f"Would remove {len(backup_files)} backup files")
                for backup_file in backup_files:
                    LOG.info(f"  {backup_file}")
            else:
                cleaned_count = 0
                for backup_file in backup_files:
                    try:
                        backup_file.unlink()
                        cleaned_count += 1
                        LOG.debug(f"Removed backup: {backup_file}")
                    except Exception as e:
                        LOG.warning(f"Failed to remove backup {backup_file}: {e}")

                LOG.info(f"Removed {cleaned_count} backup files")

            return 0

        except Exception as e:
            LOG.exception(f"Backup cleanup failed: {e}")
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
            LOG.exception(f"Info display failed: {e}")
            return 1

    def _handle_duplicates(self, args: argparse.Namespace) -> int:
        """Handle duplicate file detection."""
        try:
            import json
            import time

            from ...core import DuplicateFinder

            # Validate input paths
            for path in args.paths:
                if not path.exists():
                    LOG.error(f"Path does not exist: {path}")
                    return 1

            # Prepare extensions set
            extensions = None
            if args.extensions:
                extensions = {ext if ext.startswith(".") else f".{ext}" for ext in args.extensions}
                LOG.info(f"Filtering by extensions: {extensions}")

            # Initialize duplicate finder
            max_workers = args.workers
            if max_workers:
                LOG.info(f"Using {max_workers} parallel workers")

            finder = DuplicateFinder(max_workers=max_workers)

            # Progress callback for user feedback using class to avoid nonlocal
            class ProgressCallback:
                def __init__(self) -> None:
                    self.last_update_time = time.time()
                
                def __call__(self, message: str) -> None:
                    current_time = time.time()
                    if current_time - self.last_update_time > 1.0:  # Update every second
                        print(f"Progress: {message}")
                        self.last_update_time = current_time
            
            progress_callback = ProgressCallback()

            print("Starting duplicate file search...")
            start_time = time.time()

            # Find duplicates
            duplicates = finder.find_duplicates(
                paths=args.paths, extensions=extensions, progress_callback=progress_callback
            )

            elapsed_time = time.time() - start_time

            # Get summary
            summary = finder.get_duplicate_summary(duplicates)

            # Display results
            print("\n=== Duplicate Detection Complete ===")
            print(f"Search completed in {elapsed_time:.1f} seconds")
            print(f"Found {summary['total_groups']} duplicate groups containing {summary['total_files']} files")

            if summary["wasted_space"] > 0:
                wasted_mb = summary["wasted_space"] / (1024 * 1024)
                wasted_gb = wasted_mb / 1024
                if wasted_gb > 1:
                    print(f"Estimated wasted space: {wasted_gb:.2f} GB")
                else:
                    print(f"Estimated wasted space: {wasted_mb:.1f} MB")

            if not duplicates:
                print("No duplicate files found.")
                return 0

            # Output detailed results
            if not args.summary_only:
                print("\n=== Duplicate Groups ===")
                for i, group in enumerate(summary["groups"][:10], 1):  # Show top 10 groups
                    size_mb = group["size_each"] / (1024 * 1024)
                    wasted_mb = group["wasted_space"] / (1024 * 1024)

                    print(f"\nGroup {i}: {group['count']} files ({size_mb:.1f} MB each, {wasted_mb:.1f} MB wasted)")
                    print(f"Hash: {group['hash']}")
                    for file_path in group["files"]:
                        print(f"  {file_path}")

                if len(summary["groups"]) > 10:
                    print(f"\n... and {len(summary['groups']) - 10} more groups")
                    print("Use --summary-only to show only statistics")

            # Save to file if requested
            if args.output:
                try:
                    output_data = {
                        "search_info": {
                            "paths": [str(p) for p in args.paths],
                            "extensions": list(extensions) if extensions else None,
                            "search_time": elapsed_time,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                        "summary": summary,
                        "duplicates": {hash_val: [str(p) for p in paths] for hash_val, paths in duplicates.items()},
                    }

                    with args.output.open("w", encoding="utf-8") as f:
                        json.dump(output_data, f, indent=2, ensure_ascii=False)

                    print(f"\nDetailed results saved to: {args.output}")

                except Exception as e:
                    LOG.warning(f"Failed to save results to file: {e}")

            return 0

        except KeyboardInterrupt:
            print("\nDuplicate search cancelled by user")
            return 130
        except Exception as e:
            LOG.exception(f"Duplicate detection failed: {e}")
            return 1
