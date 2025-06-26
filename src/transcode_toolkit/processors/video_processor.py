"""Video processor implementation using the new core architecture with SSIM-based quality analysis."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import psutil
from tqdm import tqdm

from ..core import (
    BackupStrategy,
    ConfigManager,
    FFmpegError,
    FFmpegProbe,
    FFmpegProcessor,
    FileManager,
    MediaProcessor,
    ProcessingError,
    ProcessingResult,
    ProcessingStatus,
)

if TYPE_CHECKING:
    from pathlib import Path


def get_video_thermal_safe_worker_count(configured_workers: int | None) -> int:
    """
    Get a thermally safe number of workers for video processing.

    Video encoding is more intensive than audio, so we use more conservative limits.
    """
    logger = logging.getLogger(__name__)

    if configured_workers is not None and configured_workers > 0:
        # Still apply thermal safety checks even for manual configuration
        safe_workers = min(configured_workers, _get_video_thermal_limit())
        if safe_workers < configured_workers:
            logger.warning(
                f"Reducing configured workers from {configured_workers} to {safe_workers} "
                "due to thermal/system load constraints for video processing"
            )
        return safe_workers

    try:
        physical_cores = psutil.cpu_count(logical=False) or 1
        logical_cores = psutil.cpu_count(logical=True) or 1

        # Check current system load
        cpu_percent = psutil.cpu_percent(interval=1.0)

        # Very conservative approach for video encoding:
        # - Use at most 1/3 of physical cores for heavy video transcoding
        # - Leave plenty of headroom for system cooling
        max_workers = max(1, physical_cores // 3)

        # Apply additional thermal limits
        thermal_limit = _get_video_thermal_limit()
        max_workers = min(max_workers, thermal_limit)

        # If system is already under high load, reduce workers further
        if cpu_percent > 60:  # Lower threshold for video
            max_workers = max(1, max_workers // 2)
            logger.warning(f"High CPU load detected ({cpu_percent:.1f}%), reducing video workers to {max_workers}")

        # Cap at 2 workers for video to prevent thermal issues
        max_workers = min(max_workers, 2)

        logger.info(
            f"Video thermal-safe configuration: {physical_cores} physical cores, {logical_cores} logical cores, "
            f"CPU load: {cpu_percent:.1f}%. Using {max_workers} workers for thermal safety."
        )

        return max_workers

    except Exception as e:
        # Very conservative fallback if monitoring fails
        logger.warning(f"Failed to detect system specs with psutil: {e}. Using conservative fallback of 1 worker.")
        return 1


def _get_video_thermal_limit() -> int:
    """Determine thermal limits for video encoding (more conservative than audio)."""
    try:
        # Try to get temperature sensors (Linux systems mostly)
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if temps:
                max_temp = 0
                for entries in temps.values():
                    for entry in entries:
                        if entry.current and entry.current > max_temp:
                            max_temp = entry.current

                # If we can read temps and they're high, be more conservative
                if max_temp > 65:  # Above 65°C, reduce workers for video
                    return 1
                if max_temp > 55:  # Above 55°C, moderate reduction
                    return 2

        # Check available memory as another thermal/stability indicator
        memory = psutil.virtual_memory()
        if memory.percent > 75:  # High memory usage can indicate thermal stress
            return 1

        # Default thermal-safe limit for video
        return 2

    except Exception:
        # If we can't monitor anything, be very conservative
        return 1


def _check_video_thermal_throttling() -> None:
    """Check for thermal throttling during video processing."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > 85:
            logger = logging.getLogger(__name__)
            logger.warning(f"High CPU usage detected during video processing: {cpu_percent:.1f}%")

            # Brief pause to allow cooling
            time.sleep(1.0)

    except Exception:
        pass  # Don't fail processing due to monitoring issues


class VideoProcessor(MediaProcessor):
    """Video transcoding processor using H.265/HEVC codec with SSIM-based quality optimization."""

    def __init__(self, config_manager: ConfigManager) -> None:
        super().__init__("VideoProcessor")
        self.config_manager = config_manager
        self.ffmpeg = FFmpegProcessor()

        # Initialize file manager with backup strategy from config
        backup_strategy_str = config_manager.get_value("global_.cleanup_backups", "on_success")
        backup_strategy = BackupStrategy(backup_strategy_str)

        self.file_manager = FileManager(backup_strategy)

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a supported video format."""
        video_extensions = self.config_manager.get_value("video.extensions", set())
        return file_path.suffix.lower() in video_extensions

    def should_process(self, file_path: Path, **kwargs) -> bool:
        """Check if file should be processed (not already optimized)."""
        try:
            video_info = FFmpegProbe.get_video_info(file_path)
            codec = video_info.get("codec", "").lower()
            bitrate = video_info.get("bitrate")
            height = video_info.get("height", 1080)

            # Get target codec from preset to check if already using it
            default_preset = self.config_manager.config.get_video_preset("default")
            target_codec = default_preset.codec.lower()
            
            # Check if already using target codec - simple comparison
            if codec == target_codec:
                if bitrate:
                    current_bps = int(bitrate)

                    # Define reasonable bitrate thresholds by resolution
                    thresholds = {
                        480: 2_500_000,
                        720: 4_000_000,
                        1080: 8_000_000,
                        1440: 16_000_000,
                        2160: 35_000_000,
                    }

                    # Find appropriate threshold for this resolution
                    threshold = 35_000_000  # Default for very high res
                    for res, thresh in sorted(thresholds.items()):
                        if height <= res:
                            threshold = thresh
                            break

                    # Skip if bitrate is already reasonable for resolution
                    if current_bps <= threshold * 1.2:  # 20% tolerance
                        self.logger.info(
                            f"Skipping {file_path.name}: Already optimal {codec.upper()} ({current_bps / 1_000_000:.1f}Mbps vs target {threshold / 1_000_000:.1f}Mbps)"
                        )
                        return False

                    # Allow reprocessing if significantly higher bitrate
                    self.logger.info(
                        f"Will reprocess {file_path.name}: {codec.upper()} {current_bps / 1_000_000:.1f}Mbps → {threshold / 1_000_000:.1f}Mbps"
                    )
                    return True

                # If bitrate is unknown, conservatively skip
                self.logger.info(f"Skipping {file_path.name}: Already {codec.upper()} format (bitrate unknown)")
                return False

            return True

        except Exception as e:
            self.logger.warning(f"Could not analyze {file_path}: {e}")
            return False

    def process_file(self, file_path: Path, preset: str = "default", **kwargs) -> ProcessingResult:
        """Process a single video file with SSIM-guided optimization."""
        start_time = time.time()

        try:
            # Get preset configuration
            preset_config = self.config_manager.config.get_video_preset(preset)

            # Get file info and analyze content
            video_info = FFmpegProbe.get_video_info(file_path)
            original_size = video_info["size"]

            # Create temporary output file
            temp_file = file_path.with_suffix(".tmp.mp4")

            # Determine effective encoding parameters using SSIM-based analysis
            folder_quality = kwargs.get("folder_quality")
            quality_decision = self._calculate_effective_parameters(
                file_path, video_info, preset_config, folder_quality
            )

            # Build and run FFmpeg command
            gpu = kwargs.get("gpu", False)
            command = self.ffmpeg.build_video_command(
                input_file=file_path,
                output_file=temp_file,
                codec=preset_config.codec,
                crf=quality_decision.effective_crf,
                preset=quality_decision.effective_preset,
                gpu=gpu,
            )

            # Execute transcoding
            self.ffmpeg.run_command(command, file_path)
            processing_time = time.time() - start_time

            # Check output file
            if not temp_file.exists():
                msg = f"Output file not created: {temp_file}"
                raise ProcessingError(msg)

            new_size = temp_file.stat().st_size
            size_ratio = self.config_manager.get_value("video.size_keep_ratio", 0.95)

            # Decide whether to keep the transcoded file
            size_improvement = new_size < (size_ratio * original_size)

            # Always convert if source is not target codec, even with minimal size improvement
            source_codec = video_info.get("codec", "").lower()
            target_preset = self.config_manager.config.get_video_preset(preset)
            target_codec = target_preset.codec.lower()
            is_non_target_codec = source_codec != target_codec

            if size_improvement or is_non_target_codec:
                # Replace original with transcoded version
                create_backups = self.config_manager.get_value("global_.create_backups", True)
                operation = self.file_manager.atomic_replace(
                    source_path=file_path,
                    temp_path=temp_file,
                    create_backup=create_backups,
                )

                return ProcessingResult(
                    source_file=file_path,
                    status=ProcessingStatus.SUCCESS,
                    message=f"Transcoded with {preset} preset (CRF {quality_decision.effective_crf})",
                    output_file=operation.target_path,
                    original_size=original_size,
                    new_size=new_size,
                    processing_time=processing_time,
                    metadata={
                        "preset": preset,
                        "codec": preset_config.codec,
                        "crf": quality_decision.effective_crf,
                        "predicted_ssim": quality_decision.predicted_ssim,
                        "size_reduction_mb": (original_size - new_size) / (1024 * 1024),
                        "backup_created": operation.backup_path is not None,
                        "limitation_reason": quality_decision.limitation_reason,
                    },
                )

            # Remove temporary file - no significant improvement
            temp_file.unlink()

            return ProcessingResult(
                source_file=file_path,
                status=ProcessingStatus.SKIPPED,
                message="No significant size reduction achieved",
                original_size=original_size,
                new_size=new_size,
                processing_time=processing_time,
                metadata={
                    "preset": preset,
                    "crf": quality_decision.effective_crf,
                    "size_reduction_mb": (original_size - new_size) / (1024 * 1024),
                    "size_improvement": (original_size - new_size) / original_size * 100,
                },
            )

        except Exception as e:
            processing_time = time.time() - start_time

            # Clean up any temporary files
            temp_file = file_path.with_suffix(".tmp.mp4")
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)

            error_msg = str(e) if isinstance(e, (FFmpegError, ProcessingError)) else f"Unexpected error: {e}"

            return ProcessingResult(
                source_file=file_path,
                status=ProcessingStatus.ERROR,
                message=error_msg,
                processing_time=processing_time,
                metadata={"preset": preset, "error": str(e)},
            )

    def _calculate_effective_parameters(
        self,
        file_path: Path,
        video_info: dict[str, Any],
        preset_config: Any,
        folder_quality: float | None = None,
    ):
        """Calculate effective encoding parameters using SSIM-based analysis."""
        try:
            from ..core.video_analysis import analyze_file, calculate_optimal_crf

            # Analyze video content
            analysis = analyze_file(file_path, use_cache=True)
            complexity = analysis["complexity"]

            # Calculate optimal CRF based on content and target quality
            target_ssim = folder_quality or analysis["estimated_ssim_threshold"]

            quality_decision = calculate_optimal_crf(
                file_path=file_path,
                video_info=video_info,
                complexity=complexity,
                target_ssim=target_ssim,
                folder_quality=folder_quality,
            )

            self.logger.debug(
                f"Calculated optimal parameters for {file_path.name}: "
                f"CRF {quality_decision.effective_crf}, preset {quality_decision.effective_preset}, "
                f"predicted SSIM {quality_decision.predicted_ssim:.3f}"
            )

            return quality_decision

        except Exception as e:
            self.logger.warning(f"Failed to calculate optimal parameters for {file_path}: {e}")
            # Fallback to preset defaults
            from ..core.video_analysis import QualityDecision

            return QualityDecision(
                effective_crf=preset_config.crf,
                effective_preset=preset_config.preset,
                limitation_reason="Fallback due to analysis error",
                predicted_ssim=0.92,
            )

    def process_directory(
        self,
        directory: Path,
        recursive: bool = True,
        **kwargs,
    ) -> list[ProcessingResult]:
        """Process all video files in directory with multithreading."""
        # Extract parameters from kwargs
        preset = kwargs.pop("preset", "default")  # Remove from kwargs to avoid conflicts
        max_workers = kwargs.pop("max_workers", None)  # Remove from kwargs

        # Get worker count from config if not specified
        if max_workers is None:
            configured_workers = self.config_manager.get_value("global_.default_workers")
            max_workers = get_video_thermal_safe_worker_count(configured_workers)
        else:
            # Even if specified, ensure it's thermally safe
            max_workers = get_video_thermal_safe_worker_count(max_workers)

        # Get all compatible files with progress feedback
        pattern = "**/*" if recursive else "*"
        self.logger.info(f"Scanning directory: {directory} (recursive: {recursive})")

        # First pass: find all video files
        all_files = [f for f in directory.glob(pattern) if f.is_file() and self.can_process(f)]
        self.logger.info(f"Found {len(all_files)} video files to analyze")

        # Second pass: filter files that need processing with multicore analysis
        files = self._analyze_files_parallel(all_files, preset, max_workers, **kwargs)

        # Pre-compute folder quality once per directory so each thread can reuse it
        from collections import defaultdict

        from ..core.video_analysis import analyze_folder_quality

        folder_map: dict[Path, list[Path]] = defaultdict(list)
        for f in all_files:
            folder_map[f.parent].append(f)
        folder_quality_cache: dict[Path, float] = {}
        for folder, files_in_dir in folder_map.items():
            folder_quality_cache[folder] = analyze_folder_quality(folder, files_in_dir)

        progress_bar = tqdm(
            total=len(files),
            desc="Processing Videos",
            unit="file",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        self.logger.info(f"Processing {len(files)} video files with preset '{preset}'")

        results = []

        try:
            if max_workers == 1:
                # Single-threaded processing
                for file_path in files:
                    progress_bar.set_description(f"Processing {file_path.name}")
                    result = self.process_file(
                        file_path,
                        preset=preset,
                        folder_quality=folder_quality_cache[file_path.parent],
                        **kwargs,
                    )
                    results.append(result)
                    progress_bar.update(1)

                    # Check for thermal issues even in single-threaded mode
                    _check_video_thermal_throttling()
            else:
                # Multi-threaded processing with thermal monitoring
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_file = {
                        executor.submit(
                            self.process_file,
                            file_path,
                            preset=preset,
                            folder_quality=folder_quality_cache[file_path.parent],
                            **kwargs,
                        ): file_path
                        for file_path in files
                    }

                    completed_count = 0
                    for future in as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            result = future.result()
                            results.append(result)
                            # Update description with completion status
                            if result.status == ProcessingStatus.SUCCESS:
                                progress_bar.set_description(f"✓ Completed {file_path.name}")
                            elif result.status == ProcessingStatus.SKIPPED:
                                progress_bar.set_description(f"⏭ Skipped {file_path.name}")
                            else:
                                progress_bar.set_description(f"✗ Error {file_path.name}")

                            progress_bar.update(1)
                            completed_count += 1

                            # Thermal monitoring during processing
                            if completed_count % 5 == 0:  # Check every 5 completions
                                _check_video_thermal_throttling()

                        except Exception as e:
                            self.logger.exception(f"Error processing {file_path}: {e}")
                            progress_bar.update(1)

        finally:
            progress_bar.close()

        # Summary
        successful = [r for r in results if r.status == ProcessingStatus.SUCCESS]
        skipped = [r for r in results if r.status == ProcessingStatus.SKIPPED]
        errors = [r for r in results if r.status == ProcessingStatus.ERROR]

        self.logger.info(
            f"Video processing completed: {len(successful)} successful, {len(skipped)} skipped, {len(errors)} errors"
        )

        return results

    def _analyze_files_parallel(self, all_files: list[Path], preset: str, max_workers: int, **kwargs) -> list[Path]:
        """Analyze files in parallel to determine which need processing."""
        self.logger.info(f"Analyzing {len(all_files)} files to determine processing needs...")

        files_to_process = []

        # Use smaller worker count for analysis to avoid overwhelming system
        analysis_workers = min(max_workers, 4)

        if analysis_workers == 1 or len(all_files) < 10:
            # Single-threaded analysis for small sets
            for file_path in all_files:
                if self.should_process(file_path, **kwargs):
                    files_to_process.append(file_path)
        else:
            # Multi-threaded analysis
            with ThreadPoolExecutor(max_workers=analysis_workers) as executor:
                future_to_file = {
                    executor.submit(self.should_process, file_path, **kwargs): file_path for file_path in all_files
                }

                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        if future.result():
                            files_to_process.append(file_path)
                    except Exception as e:
                        self.logger.warning(f"Error analyzing {file_path}: {e}")

        self.logger.info(f"Analysis complete: {len(files_to_process)} files need processing")
        return files_to_process
