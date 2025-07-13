"""Utility CLI commands."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse

    from ...core import ConfigManager

from ...config.constants import MAX_DISPLAYED_GROUPS
from ...core.duplicate_finder import DuplicateFinder

LOG = logging.getLogger(__name__)


class UtilityCommands:
    """Utility command handlers."""

    def __init__(self, config_manager: ConfigManager) -> None:
        """Initialize utility commands handler."""
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
        LOG.error("Unknown utility command: %s", args.util_command)
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
                LOG.info("Would remove %d backup files", len(backup_files))
                for backup_file in backup_files:
                    LOG.info("  %s", backup_file)
            else:
                cleaned_count = 0
                for backup_file in backup_files:
                    try:
                        backup_file.unlink()
                        cleaned_count += 1
                        LOG.debug("Removed backup: %s", backup_file)
                    except (OSError, PermissionError) as e:
                        LOG.warning("Failed to remove backup %s: %s", backup_file, e)

                LOG.info("Removed %d backup files", cleaned_count)
                return 0

        except (OSError, PermissionError):
            LOG.exception("Backup cleanup failed")
            return 1

        return 0

    def _handle_info(self, _args: argparse.Namespace) -> int:
        """Handle info display."""
        try:
            import shutil
            from pathlib import Path

            # Check for required executables
            executables = ["ffmpeg", "ffprobe"]
            for exe in executables:
                path = shutil.which(exe)
                if path:
                    pass

            # Check config file
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
            "✓ Found" if config_path.exists() else "✗ Missing"
            if config_path.exists():
                return 0

        except (OSError, PermissionError):
            LOG.exception("Info display failed")
            return 1

        return 0

    def _handle_duplicates(self, args: argparse.Namespace) -> int:
        """Handle duplicate file detection."""
        try:
            # Validate input paths
            if not self._validate_paths(args.paths):
                return 1

            # Prepare extensions set
            extensions = self._prepare_extensions(args.extensions)

            # Initialize duplicate finder
            finder = self._initialize_finder(args.workers)

            # Create progress callback
            progress_callback = self._create_progress_callback()

            # Find duplicates
            start_time = time.time()
            duplicates = finder.find_duplicates(
                paths=args.paths,
                extensions=extensions,
                progress_callback=progress_callback if callable(progress_callback) else None,
            )
            elapsed_time = time.time() - start_time

            # Get summary
            summary = finder.get_duplicate_summary(duplicates)

            # Display results
            self._display_results(summary, duplicates, summary_only=args.summary_only)

            # Save to file if requested
            if args.output:
                return self._save_results(
                    args,
                    extensions=extensions,
                    elapsed_time=elapsed_time,
                    summary=summary,
                    duplicates=duplicates,
                )

        except KeyboardInterrupt:
            return 130
        except Exception:
            LOG.exception("Duplicate detection failed")
            return 1

        return 0

    def _validate_paths(self, paths: list) -> bool:
        """Validate input paths."""
        for path in paths:
            if not path.exists():
                LOG.error("Path does not exist: %s", path)
                return False
        return True

    def _prepare_extensions(self, extensions: list | None) -> set | None:
        """Prepare extensions set."""
        if not extensions:
            return None
        ext_set = {ext if ext.startswith(".") else f".{ext}" for ext in extensions}
        LOG.info("Filtering by extensions: %s", ext_set)
        return ext_set

    def _initialize_finder(self, max_workers: int | None) -> DuplicateFinder:
        """Initialize duplicate finder."""
        if max_workers:
            LOG.info("Using %d parallel workers", max_workers)
        return DuplicateFinder(max_workers=max_workers)

    def _create_progress_callback(self) -> object:
        """Create progress callback."""

        class ProgressCallback:
            def __init__(self) -> None:
                self.last_update_time = time.time()

            def __call__(self, _message: str) -> None:
                current_time = time.time()
                if current_time - self.last_update_time > 1.0:  # Update every second
                    self.last_update_time = current_time

        return ProgressCallback()

    def _display_results(self, summary: dict, duplicates: dict, *, summary_only: bool) -> None:
        """Display duplicate detection results."""
        if summary["wasted_space"] > 0:
            wasted_mb = summary["wasted_space"] / (1024 * 1024)
            wasted_gb = wasted_mb / 1024
            if wasted_gb > 1:
                pass
            else:
                pass

        if not duplicates:
            return

        # Output detailed results
        if not summary_only:
            for _i, group in enumerate(summary["groups"][:MAX_DISPLAYED_GROUPS], 1):
                group["size_each"] / (1024 * 1024)
                wasted_mb = group["wasted_space"] / (1024 * 1024)

                for _file_path in group["files"]:
                    pass

            if len(summary["groups"]) > MAX_DISPLAYED_GROUPS:
                pass

    def _save_results(
        self,
        args: argparse.Namespace,
        *,
        extensions: set[str] | None,
        elapsed_time: float,
        summary: dict[str, Any],
        duplicates: dict[str, list[Path]],
    ) -> int:
        """Save results to file."""
        try:
            import json

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

        except (OSError, PermissionError) as e:
            LOG.warning("Failed to save results to file: %s", e)
            return 1
        else:
            return 0
