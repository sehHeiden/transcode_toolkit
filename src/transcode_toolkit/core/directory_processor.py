"""Unified directory processing logic for all media processors."""

from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from .base import MediaProcessor
    from .config import ConfigManager
    from .file_manager import FileManager

from tqdm import tqdm

from .base import ProcessingResult, ProcessingStatus
from .thermal import MediaType, check_thermal_throttling, get_thermal_safe_worker_count

# Constants
MIN_FILES_FOR_PARALLEL_ANALYSIS = 10

LOG = logging.getLogger(__name__)


@dataclass
class DirectoryProcessingConfig:
    """Configuration for directory processing operations."""

    recursive: bool = True
    media_type: MediaType = "audio"
    preset: str = "default"
    max_workers: int | None = None
    folder_analysis_func: Callable[[Path, list[Path]], object] | None = None


@dataclass
class ProcessingContext:
    """Context data for processing operations."""

    processor: MediaProcessor
    preset: str
    media_type: MediaType
    folder_cache: dict[Path, object]
    kwargs: dict[str, object]


def process_directory_unified(
    processor: MediaProcessor,
    directory: Path,
    config: DirectoryProcessingConfig | None = None,
    **kwargs: object,
) -> list[ProcessingResult]:
    """
    Unified directory processing logic for all media processors.

    Args:
        processor: The MediaProcessor instance
        directory: Directory to process
        config: Processing configuration (optional, uses defaults if None)
        **kwargs: Additional arguments passed to process_file

    Returns:
        List of processing results

    """
    if config is None:
        config = DirectoryProcessingConfig()

    # Determine thermal-safe worker count
    max_workers = _get_thermal_safe_workers(processor, config.max_workers, config.media_type)

    # Find and analyze files
    all_files = _discover_media_files(processor, directory, recursive=config.recursive, media_type=config.media_type)
    files = _analyze_files_parallel(processor, all_files, max_workers, **kwargs)

    # Build folder cache if needed
    folder_cache = _build_folder_cache(config.folder_analysis_func, all_files) if config.folder_analysis_func else {}

    # Process files with progress tracking
    context = ProcessingContext(
        processor=processor,
        preset=config.preset,
        media_type=config.media_type,
        folder_cache=folder_cache,
        kwargs=kwargs,
    )
    return _process_files_with_progress(context, files, max_workers)


def _get_thermal_safe_workers(processor: MediaProcessor, max_workers: int | None, media_type: MediaType) -> int:
    """Get thermal-safe worker count."""
    if max_workers is None:
        configured_workers = 4
        if hasattr(processor, "config_manager") and hasattr(processor.config_manager, "get_value"):
            # Type cast to ConfigManager to access get_value method
            config_manager = cast("ConfigManager", processor.config_manager)
            configured_workers_value = config_manager.get_value("global_.default_workers", 4)
            # Safely convert to int, falling back to 4 if the value isn't convertible
            if isinstance(configured_workers_value, int):
                configured_workers = configured_workers_value
            elif isinstance(configured_workers_value, str):
                try:
                    configured_workers = int(configured_workers_value)
                except ValueError:
                    configured_workers = 4
            else:
                configured_workers = 4
        return get_thermal_safe_worker_count(configured_workers, media_type)
    return get_thermal_safe_worker_count(max_workers, media_type)


def _discover_media_files(
    processor: MediaProcessor, directory: Path, *, recursive: bool, media_type: MediaType
) -> list[Path]:
    """Discover all compatible media files in directory."""
    pattern = "**/*" if recursive else "*"
    processor.logger.info("Scanning directory: %s (recursive: %s)", directory, recursive)

    all_files = [f for f in directory.glob(pattern) if f.is_file() and processor.can_process(f)]
    processor.logger.info("Found %d %s files to analyze", len(all_files), media_type)
    return all_files


def _build_folder_cache(
    folder_analysis_func: Callable[[Path, list[Path]], object], all_files: list[Path]
) -> dict[Path, object]:
    """Build cache of folder-level analysis data."""
    folder_cache: dict[Path, object] = {}
    folder_map: dict[Path, list[Path]] = defaultdict(list)

    for f in all_files:
        folder_map[f.parent].append(f)

    for folder, files_in_dir in folder_map.items():
        folder_cache[folder] = folder_analysis_func(folder, files_in_dir)

    return folder_cache


def _add_folder_data_to_kwargs(
    kwargs: dict[str, object], folder_cache: dict[Path, object], file_path: Path, media_type: MediaType
) -> dict[str, object]:
    """Add folder-specific data to processing kwargs."""
    process_kwargs = kwargs.copy()
    if folder_cache and file_path.parent in folder_cache:
        if media_type == "audio":
            process_kwargs["folder_snr"] = folder_cache[file_path.parent]
        else:  # video
            process_kwargs["folder_quality"] = folder_cache[file_path.parent]
    return process_kwargs


def _process_single_threaded(
    context: ProcessingContext, files: list[Path], progress_bar: tqdm
) -> list[ProcessingResult]:
    """Process files in single-threaded mode."""
    results = []
    for file_path in files:
        progress_bar.set_description(f"Processing {file_path.name}")

        process_kwargs = _add_folder_data_to_kwargs(context.kwargs, context.folder_cache, file_path, context.media_type)
        result = context.processor.process_file(file_path, preset=context.preset, **process_kwargs)
        results.append(result)
        progress_bar.update(1)

        # Check for thermal issues
        check_thermal_throttling(context.media_type)

    return results


def _process_multi_threaded(
    context: ProcessingContext, files: list[Path], max_workers: int, progress_bar: tqdm
) -> list[ProcessingResult]:
    """Process files in multi-threaded mode."""
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {}

        # Submit all tasks
        for file_path in files:
            process_kwargs = _add_folder_data_to_kwargs(
                context.kwargs, context.folder_cache, file_path, context.media_type
            )
            future = executor.submit(context.processor.process_file, file_path, preset=context.preset, **process_kwargs)
            future_to_file[future] = file_path

        # Collect results
        completed_count = 0
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                _update_progress_description(progress_bar, result, file_path)
            except Exception as e:
                context.processor.logger.exception("Error processing %s", file_path)
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

                # Check thermal throttling periodically
                if completed_count % 5 == 0:
                    check_thermal_throttling(context.media_type)

    return results


def _update_progress_description(progress_bar: tqdm, result: ProcessingResult, file_path: Path) -> None:
    """Update progress bar description based on result status."""
    if result.status.value == "success":
        progress_bar.set_description(f"✓ Completed {file_path.name}")
    elif result.status.value == "skipped":
        progress_bar.set_description(f"⏭ Skipped {file_path.name}")
    else:
        progress_bar.set_description(f"✗ Error {file_path.name}")


def _process_files_with_progress(
    context: ProcessingContext, files: list[Path], max_workers: int
) -> list[ProcessingResult]:
    """Process files with progress tracking and cleanup."""
    progress_bar = tqdm(
        total=len(files),
        desc=f"Processing {context.media_type.title()}s",
        unit="file",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    context.processor.logger.info(
        "Processing %d %s files with preset '%s'", len(files), context.media_type, context.preset
    )

    try:
        if max_workers == 1:
            results = _process_single_threaded(context, files, progress_bar)
        else:
            results = _process_multi_threaded(context, files, max_workers, progress_bar)
    finally:
        progress_bar.close()
        _cleanup_and_log_summary(context.processor)

    return results


def _cleanup_and_log_summary(processor: MediaProcessor) -> None:
    """Clean up session backups and log processing summary."""
    if hasattr(processor, "file_manager"):
        # Type cast to FileManager for proper method access
        file_manager = cast("FileManager", processor.file_manager)
        file_manager.cleanup_session_backups()

        summary = file_manager.get_session_summary()
        processor.logger.info(
            "Processing complete: %d successful, %d failed, %d backups created",
            summary["successful_operations"],
            summary["failed_operations"],
            summary["backups_created"],
        )
    else:
        processor.logger.info("Processing complete")


def _analyze_files_parallel(
    processor: MediaProcessor, all_files: list[Path], max_workers: int, **kwargs: object
) -> list[Path]:
    """Analyze files in parallel to determine which need processing."""
    processor.logger.info("Analyzing %d files to determine processing needs...", len(all_files))

    files_to_process: list[Path] = []
    analysis_workers = min(max_workers, 4)

    if analysis_workers == 1 or len(all_files) < MIN_FILES_FOR_PARALLEL_ANALYSIS:
        # Single-threaded analysis for small sets
        files_to_process.extend(file_path for file_path in all_files if processor.should_process(file_path, **kwargs))
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
                except (OSError, ValueError, RuntimeError) as e:
                    processor.logger.warning("Error analyzing %s: %s", file_path, e)
                    # Include file anyway if analysis fails
                    files_to_process.append(file_path)

    processor.logger.info(
        "Will process %d files (skipping %d)", len(files_to_process), len(all_files) - len(files_to_process)
    )
    return files_to_process
