"""Unified directory processing logic for all media processors."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from .thermal import MediaType, check_thermal_throttling, get_thermal_safe_worker_count

if TYPE_CHECKING:
    from pathlib import Path

    from .base import ProcessingResult
from .base import ProcessingResult, ProcessingStatus

LOG = logging.getLogger(__name__)


def process_directory_unified(
    processor: Any,  # MediaProcessor instance
    directory: Path,
    recursive: bool = True,
    media_type: MediaType = "audio",
    preset: str = "default",
    max_workers: int | None = None,
    folder_analysis_func: Callable[[Path, list[Path]], Any] | None = None,
    **kwargs,
) -> list[ProcessingResult]:
    """
    Unified directory processing logic for all media processors.

    Args:
        processor: The MediaProcessor instance
        directory: Directory to process
        recursive: Process subdirectories recursively
        media_type: Type of media being processed for thermal management
        preset: Preset to use for processing
        max_workers: Maximum number of workers
        folder_analysis_func: Function to analyze folder-level properties
        **kwargs: Additional arguments passed to process_file

    Returns:
        List of processing results

    """
    # Get worker count from config if not specified
    if max_workers is None:
        configured_workers = processor.config_manager.get_value("global_.default_workers")
        max_workers = get_thermal_safe_worker_count(configured_workers, media_type)
    else:
        # Even if specified, ensure it's thermally safe
        max_workers = get_thermal_safe_worker_count(max_workers, media_type)

    # Get all compatible files with progress feedback
    pattern = "**/*" if recursive else "*"
    processor.logger.info(f"Scanning directory: {directory} (recursive: {recursive})")

    # First pass: find all media files
    all_files = [f for f in directory.glob(pattern) if f.is_file() and processor.can_process(f)]
    processor.logger.info(f"Found {len(all_files)} {media_type} files to analyze")

    # Second pass: filter files that need processing with multicore analysis
    files = _analyze_files_parallel(processor, all_files, preset, max_workers, **kwargs)

    # Pre-compute folder properties if analysis function provided
    folder_cache: dict[Path, Any] = {}
    if folder_analysis_func:
        folder_map: dict[Path, list[Path]] = defaultdict(list)
        for f in all_files:
            folder_map[f.parent].append(f)

        for folder, files_in_dir in folder_map.items():
            folder_cache[folder] = folder_analysis_func(folder, files_in_dir)

    # Initialize progress bar
    progress_bar = tqdm(
        total=len(files),
        desc=f"Processing {media_type.title()}s",
        unit="file",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    processor.logger.info(f"Processing {len(files)} {media_type} files with preset '{preset}'")

    results = []

    try:
        if max_workers == 1:
            # Single-threaded processing
            for file_path in files:
                progress_bar.set_description(f"Processing {file_path.name}")

                # Add folder cache data to kwargs if available
                process_kwargs = kwargs.copy()
                if folder_cache and file_path.parent in folder_cache:
                    if media_type == "audio":
                        process_kwargs["folder_snr"] = folder_cache[file_path.parent]
                    else:  # video
                        process_kwargs["folder_quality"] = folder_cache[file_path.parent]

                result = processor.process_file(file_path, preset=preset, **process_kwargs)
                results.append(result)
                progress_bar.update(1)

                # Check for thermal issues even in single-threaded mode
                check_thermal_throttling(media_type)
        else:
            # Multi-threaded processing with thermal monitoring
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {}

                for file_path in files:
                    # Add folder cache data to kwargs if available
                    process_kwargs = kwargs.copy()
                    if folder_cache and file_path.parent in folder_cache:
                        if media_type == "audio":
                            process_kwargs["folder_snr"] = folder_cache[file_path.parent]
                        else:  # video
                            process_kwargs["folder_quality"] = folder_cache[file_path.parent]

                    future = executor.submit(processor.process_file, file_path, preset=preset, **process_kwargs)
                    future_to_file[future] = file_path

                completed_count = 0
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                        # Update description with completion status
                        if result.status.value == "success":
                            progress_bar.set_description(f"✓ Completed {file_path.name}")
                        elif result.status.value == "skipped":
                            progress_bar.set_description(f"⏭ Skipped {file_path.name}")
                        else:
                            progress_bar.set_description(f"✗ Error {file_path.name}")
                    except Exception as e:
                        processor.logger.exception(f"Error processing {file_path}: {e}")
                        # Import ProcessingResult and ProcessingStatus from processor's module

                        results.append(
                            ProcessingResult(
                                source_file=file_path,
                                status=ProcessingStatus.ERROR,
                                message=f"Threading error: {e}",
                            )
                        )
                        progress_bar.set_description(f"✗ Error {file_path.name}")
                    finally:
                        progress_bar.update(1)
                        completed_count += 1

                        # Check thermal throttling every 5 files
                        if completed_count % 5 == 0:
                            check_thermal_throttling(media_type)
    finally:
        # Close progress bar
        progress_bar.close()

        # Always clean up session backups
        processor.file_manager.cleanup_session_backups()

        # Log summary
        summary = processor.file_manager.get_session_summary()
        processor.logger.info(
            f"Processing complete: {summary['successful_operations']} successful, "
            f"{summary['failed_operations']} failed, "
            f"{summary['backups_created']} backups created"
        )

    return results


def _analyze_files_parallel(
    processor: Any, all_files: list[Path], preset: str, max_workers: int, **kwargs
) -> list[Path]:
    """Analyze files in parallel to determine which need processing."""
    processor.logger.info(f"Analyzing {len(all_files)} files to determine processing needs...")

    files_to_process = []

    # Use smaller worker count for analysis to avoid overwhelming system
    analysis_workers = min(max_workers, 4)

    if analysis_workers == 1 or len(all_files) < 10:
        # Single-threaded analysis for small sets
        for file_path in all_files:
            if processor.should_process(file_path, **kwargs):
                files_to_process.append(file_path)
    else:
        # Multi-threaded analysis
        with ThreadPoolExecutor(max_workers=analysis_workers) as executor:
            future_to_file = {
                executor.submit(processor.should_process, file_path, **kwargs): file_path for file_path in all_files
            }

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    if future.result():  # should_process returned True
                        files_to_process.append(file_path)
                except Exception as e:
                    processor.logger.warning(f"Error analyzing {file_path}: {e}")
                    # Include file anyway if analysis fails
                    files_to_process.append(file_path)

    processor.logger.info(
        f"Will process {len(files_to_process)} files (skipping {len(all_files) - len(files_to_process)})"
    )
    return files_to_process
