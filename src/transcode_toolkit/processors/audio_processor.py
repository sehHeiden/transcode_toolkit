"""Audio processor implementation using the new core architecture."""

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


def get_thermal_safe_worker_count(configured_workers: int | None) -> int:
    """
    Get a thermally safe number of workers that doesn't overwhelm the system.

    Uses psutil to monitor system load and temperature (where available),
    ensuring we don't use too many cores that could cause thermal issues.

    Args:
        configured_workers: The configured worker count, or None for auto-detection

    Returns:
        Safe number of workers considering thermal and system constraints

    """
    logger = logging.getLogger(__name__)

    if configured_workers is not None and configured_workers > 0:
        # Still apply thermal safety checks even for manual configuration
        safe_workers = min(configured_workers, _get_thermal_limit())
        if safe_workers < configured_workers:
            logger.warning(
                f"Reducing configured workers from {configured_workers} to {safe_workers} "
                "due to thermal/system load constraints"
            )
        return safe_workers

    # Get actual core counts using psutil
    try:
        physical_cores = psutil.cpu_count(logical=False) or 1
        logical_cores = psutil.cpu_count(logical=True) or 1

        # Check current system load
        cpu_percent = psutil.cpu_percent(interval=1.0)

        # Very conservative approach for thermal safety:
        # - Use at most half the physical cores for heavy transcoding
        # - Leave plenty of headroom for system cooling
        max_workers = max(1, physical_cores // 2)

        # Apply additional thermal limits
        thermal_limit = _get_thermal_limit()
        max_workers = min(max_workers, thermal_limit)

        # If system is already under high load, reduce workers further
        if cpu_percent > 70:
            max_workers = max(1, max_workers // 2)
            logger.warning(f"High CPU load detected ({cpu_percent:.1f}%), reducing workers to {max_workers}")

        # Cap at 4 workers to prevent thermal issues on most systems
        max_workers = min(max_workers, 4)

        logger.info(
            f"Thermal-safe configuration: {physical_cores} physical cores, {logical_cores} logical cores, "
            f"CPU load: {cpu_percent:.1f}%. Using {max_workers} workers for thermal safety."
        )

        return max_workers

    except Exception as e:
        # Very conservative fallback if monitoring fails
        logger.warning(f"Failed to detect system specs with psutil: {e}. Using conservative fallback of 1 worker.")
        return 1


def _get_thermal_limit() -> int:
    """
    Determine thermal limits based on system capabilities.

    Returns:
        Maximum recommended workers considering thermal constraints

    """
    try:
        # Try to get temperature sensors (Linux systems mostly)
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if temps:
                max_temp = 0
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current and entry.current > max_temp:
                            max_temp = entry.current

                # If we can read temps and they're high, be more conservative
                if max_temp > 70:  # Above 70°C, reduce workers
                    return 2
                if max_temp > 60:  # Above 60°C, moderate reduction
                    return 3

        # Check available memory as another thermal/stability indicator
        memory = psutil.virtual_memory()
        if memory.percent > 80:  # High memory usage can indicate thermal stress
            return 2

        # Default thermal-safe limit
        return 4

    except Exception:
        # If we can't monitor anything, be very conservative
        return 2


class AudioProcessor(MediaProcessor):
    """Audio transcoding processor using Opus codec."""

    def __init__(self, config_manager: ConfigManager) -> None:
        super().__init__("AudioProcessor")
        self.config_manager = config_manager
        self.ffmpeg = FFmpegProcessor()

        # Initialize file manager with backup strategy from config
        backup_strategy_str = config_manager.get_value("global_.cleanup_backups", "on_success")
        backup_strategy = BackupStrategy(backup_strategy_str)

        self.file_manager = FileManager(backup_strategy)

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a supported audio format."""
        audio_extensions = self.config_manager.get_value("audio.extensions", set())
        return file_path.suffix.lower() in audio_extensions

    def should_process(self, file_path: Path, **kwargs) -> bool:
        """Check if file should be processed (not already optimized)."""
        try:
            preset = kwargs.get("preset", "music")
            audio_info = FFmpegProbe.get_audio_info(file_path)
            preset_config = self.config_manager.config.get_audio_preset(preset)

            # Check if already Opus - allow conversion if significant bitrate difference
            if audio_info["codec"] == "opus":
                if audio_info["bitrate"]:
                    target_bps = int(preset_config.bitrate.rstrip("k")) * 1000
                    current_bps = int(audio_info["bitrate"])
                    # Allow reprocessing if current bitrate is significantly higher
                    if current_bps > target_bps * 1.2:  # 20% tolerance for Opus→Opus
                        self.logger.info(
                            f"Will reprocess {file_path.name}: Opus {current_bps / 1000:.0f}k → {target_bps / 1000:.0f}k"
                        )
                        return True
                    self.logger.info(
                        f"Skipping {file_path.name}: Already optimal Opus ({current_bps / 1000:.0f}k vs target {target_bps / 1000:.0f}k)"
                    )
                    return False
                # If bitrate is unknown, conservatively skip
                self.logger.info(f"Skipping {file_path.name}: Already Opus format (bitrate unknown)")
                return False

            return True

        except Exception as e:
            self.logger.warning(f"Could not analyze {file_path}: {e}")
            return False

    def process_file(self, file_path: Path, preset: str = "music", **kwargs) -> ProcessingResult:
        """Process a single audio file."""
        start_time = time.time()

        try:
            # Get preset configuration
            preset_config = self.config_manager.config.get_audio_preset(preset)

            # Get file info
            audio_info = FFmpegProbe.get_audio_info(file_path)
            original_size = audio_info["size"]

            # Create temporary output file
            temp_file = file_path.with_suffix(".tmp.opus")

            # Determine effective bitrate using SNR-based intelligent limiting
            folder_snr = kwargs.get("folder_snr")
            effective_bitrate = self._calculate_effective_bitrate(file_path, audio_info, preset_config, folder_snr)

            # Build and run FFmpeg command
            command = self.ffmpeg.build_audio_command(
                input_file=file_path,
                output_file=temp_file,
                codec="libopus",
                bitrate=effective_bitrate,
                application=preset_config.application,
                cutoff=preset_config.cutoff,
                channels=preset_config.channels,
                vbr="on",
                compression_level=10,
            )

            # Execute transcoding
            self.ffmpeg.run_command(command, file_path)
            processing_time = time.time() - start_time

            # Check output file
            if not temp_file.exists():
                msg = f"Output file not created: {temp_file}"
                raise ProcessingError(msg)

            new_size = temp_file.stat().st_size
            size_ratio = self.config_manager.get_value("audio.size_keep_ratio", 0.95)

            # Decide whether to keep the transcoded file
            is_lossless = file_path.suffix.lower() in {".flac", ".wav"}
            size_improvement = new_size < (size_ratio * original_size)

            if size_improvement or is_lossless:
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
                    message=f"Transcoded with {preset} preset",
                    output_file=operation.target_path,
                    original_size=original_size,
                    new_size=new_size,
                    processing_time=processing_time,
                    metadata={
                        "preset": preset,
                        "codec": "opus",
                        "bitrate": preset_config.bitrate,
                        "size_reduction_mb": (original_size - new_size) / (1024 * 1024),
                        "backup_created": operation.backup_path is not None,
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
                    "size_reduction_mb": (original_size - new_size) / (1024 * 1024),
                    "size_improvement": (original_size - new_size) / original_size * 100,
                },
            )

        except Exception as e:
            processing_time = time.time() - start_time

            # Clean up any temporary files
            temp_file = file_path.with_suffix(".tmp.opus")
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

    def process_directory(
        self,
        directory: Path,
        recursive: bool = True,
        **kwargs,
    ) -> list[ProcessingResult]:
        """Process all audio files in directory with multithreading."""
        # Extract parameters from kwargs
        preset = kwargs.pop("preset", "music")  # Remove from kwargs to avoid conflicts
        max_workers = kwargs.pop("max_workers", None)  # Remove from kwargs

        # Get worker count from config if not specified
        if max_workers is None:
            configured_workers = self.config_manager.get_value("global_.default_workers")
            max_workers = get_thermal_safe_worker_count(configured_workers)
        else:
            # Even if specified, ensure it's thermally safe
            max_workers = get_thermal_safe_worker_count(max_workers)

        # Note: Backup cleanup is now handled only at session end for successful operations

        # Get all compatible files with progress feedback
        pattern = "**/*" if recursive else "*"
        self.logger.info(f"Scanning directory: {directory} (recursive: {recursive})")

        # First pass: find all audio files
        all_files = [f for f in directory.glob(pattern) if f.is_file() and self.can_process(f)]
        self.logger.info(f"Found {len(all_files)} audio files to analyze")

        # Second pass: filter files that need processing with multicore analysis
        files = self._analyze_files_parallel(all_files, preset, max_workers, **kwargs)

        # Initialize progress bar with more details
        # Pre-compute folder SNR once per directory so each thread can reuse it
        from collections import defaultdict

        from ..core.audio_analysis import analyze_folder_snr

        folder_map: dict[Path, list[Path]] = defaultdict(list)
        for f in all_files:
            folder_map[f.parent].append(f)
        folder_snr_cache: dict[Path, float] = {}
        for folder, files_in_dir in folder_map.items():
            folder_snr_cache[folder] = analyze_folder_snr(folder, files_in_dir)

        progress_bar = tqdm(
            total=len(files),
            desc="Processing Files",
            unit="file",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        self.logger.info(f"Processing {len(files)} audio files with preset '{preset}'")

        results = []

        try:
            if max_workers == 1:
                # Single-threaded processing
                for file_path in files:
                    progress_bar.set_description(f"Processing {file_path.name}")
                    result = self.process_file(
                        file_path,
                        preset=preset,
                        folder_snr=folder_snr_cache[file_path.parent],
                        **kwargs,
                    )
                    results.append(result)
                    progress_bar.update(1)

                    # Check for thermal issues even in single-threaded mode
                    _check_thermal_throttling()
            else:
                # Multi-threaded processing with thermal monitoring
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_file = {
                        executor.submit(
                            self.process_file,
                            file_path,
                            preset=preset,
                            folder_snr=folder_snr_cache[file_path.parent],
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
                        except Exception as e:
                            self.logger.exception(f"Error processing {file_path}: {e}")
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
                                _check_thermal_throttling()
        finally:
            # Close progress bar
            progress_bar.close()

            # Always clean up session backups
            self.file_manager.cleanup_session_backups()

            # Log summary
            summary = self.file_manager.get_session_summary()
            self.logger.info(
                f"Processing complete: {summary['successful_operations']} successful, "
                f"{summary['failed_operations']} failed, "
                f"{summary['backups_created']} backups created"
            )

        return results

    def _calculate_effective_bitrate(
        self, file_path: Path, audio_info: dict[str, Any], preset_config, folder_snr: float | None = None
    ) -> str:
        """
        Calculate effective bitrate using SNR-based intelligent limiting.

        Args:
            file_path: Path to the audio file
            audio_info: Audio metadata from FFprobe
            preset_config: AudioPreset configuration

        Returns:
            Effective bitrate string (e.g., "128k")

        """
        target_bitrate_bps = int(preset_config.bitrate.rstrip("k")) * 1000

        # Check if SNR scaling is enabled and has a threshold
        if not preset_config.snr_bitrate_scale or preset_config.min_snr_db is None:
            # SNR scaling disabled - only apply input bitrate limit
            if audio_info["bitrate"]:
                input_bitrate_bps = int(audio_info["bitrate"])
                if target_bitrate_bps > input_bitrate_bps:
                    effective_bitrate = f"{input_bitrate_bps // 1000}k"
                    self.logger.info(
                        f"Limiting bitrate for {file_path.name}: {preset_config.bitrate} → {effective_bitrate} (input limit, SNR scaling disabled)"
                    )
                    return effective_bitrate

            self.logger.debug(f"Using preset bitrate for {file_path.name}: {preset_config.bitrate}")
            return preset_config.bitrate

        # SNR scaling enabled - calculate effective bitrate
        estimated_snr = folder_snr if folder_snr is not None else FFmpegProbe.estimate_snr(file_path, audio_info)
        self.logger.debug(f"Estimated SNR for {file_path.name}: {estimated_snr:.1f} dB")

        # Calculate SNR-adjusted bitrate if below threshold
        if estimated_snr < preset_config.min_snr_db:
            # Scale down based on SNR ratio
            snr_ratio = estimated_snr / preset_config.min_snr_db
            snr_adjusted_bps = int(target_bitrate_bps * snr_ratio)

            # Apply minimum floor based on preset type
            is_voice = preset_config.application in ["voip", "voice"]
            min_bitrate_bps = 32000 if is_voice else 64000
            snr_adjusted_bps = max(snr_adjusted_bps, min_bitrate_bps)

            limitation_reason = f"SNR-limited ({estimated_snr:.1f}dB < {preset_config.min_snr_db}dB)"
        else:
            # SNR sufficient for target bitrate
            snr_adjusted_bps = target_bitrate_bps
            limitation_reason = None

        # Apply input bitrate ceiling (don't upsample)
        final_bitrate_bps = snr_adjusted_bps
        if audio_info["bitrate"]:
            input_bitrate_bps = int(audio_info["bitrate"])
            if snr_adjusted_bps > input_bitrate_bps:
                final_bitrate_bps = input_bitrate_bps
                limitation_reason = f"input + {limitation_reason}" if limitation_reason else "input limit"

        effective_bitrate = f"{final_bitrate_bps // 1000}k"

        # Log decision
        if limitation_reason:
            self.logger.info(
                f"Limiting bitrate for {file_path.name}: {preset_config.bitrate} → {effective_bitrate} ({limitation_reason})"
            )
        elif effective_bitrate != preset_config.bitrate:
            self.logger.info(f"Using adjusted bitrate for {file_path.name}: {effective_bitrate}")
        else:
            self.logger.debug(f"Using preset bitrate for {file_path.name}: {preset_config.bitrate}")

        return effective_bitrate

    def _analyze_files_parallel(self, all_files: list[Path], preset: str, max_workers: int, **kwargs) -> list[Path]:
        """
        Analyze files in parallel to determine which ones need processing.

        This parallelizes the should_process check which involves FFmpeg calls.
        """
        files_to_process = []

        if len(all_files) <= 20:
            # For small sets, use sequential processing (overhead isn't worth it)
            for file_path in all_files:
                if self.should_process(file_path, preset=preset, **kwargs):
                    files_to_process.append(file_path)
            self.logger.info(f"Analysis complete: {len(files_to_process)}/{len(all_files)} files need processing")
            return files_to_process

        # For larger sets, use parallel analysis
        analysis_workers = min(max_workers, len(all_files) // 4)  # Don't use too many workers for analysis
        analysis_workers = max(1, analysis_workers)

        self.logger.info(f"Analyzing {len(all_files)} files using {analysis_workers} workers...")

        with tqdm(
            total=len(all_files),
            desc="Analyzing files",
            unit="file",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as progress:
            with ThreadPoolExecutor(max_workers=analysis_workers) as executor:
                # Submit all analysis tasks
                future_to_file = {
                    executor.submit(self.should_process, file_path, preset=preset, **kwargs): file_path
                    for file_path in all_files
                }

                # Collect results as they complete
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        should_process = future.result()
                        if should_process:
                            files_to_process.append(file_path)

                        # Update progress with current file being checked
                        progress.set_description(f"Checked {file_path.stem[:20]}...")
                    except Exception as e:
                        self.logger.warning(f"Failed to analyze {file_path}: {e}")
                    finally:
                        progress.update(1)

        self.logger.info(f"Analysis complete: {len(files_to_process)}/{len(all_files)} files need processing")
        return files_to_process


def _check_thermal_throttling() -> None:
    """
    Check system thermal status and add delays if needed.

    This function monitors CPU usage and temperature (where available)
    and adds cooling delays if the system appears to be under thermal stress.
    """
    try:
        logger = logging.getLogger(__name__)

        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)  # Quick check

        # If CPU is very high, add a cooling delay
        if cpu_percent > 90:
            logger.warning(f"Very high CPU usage ({cpu_percent:.1f}%), adding cooling delay...")
            time.sleep(2.0)  # 2 second cooling delay
        elif cpu_percent > 80:
            logger.info(f"High CPU usage ({cpu_percent:.1f}%), adding brief cooling delay...")
            time.sleep(0.5)  # Brief cooling delay

        # Try to check temperature if available
        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    max_temp = 0
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.current and entry.current > max_temp:
                                max_temp = entry.current

                    if max_temp > 80:
                        logger.warning(f"High CPU temperature ({max_temp:.1f}°C), adding extended cooling delay...")
                        time.sleep(5.0)  # Extended cooling delay
                    elif max_temp > 70:
                        logger.info(f"Elevated CPU temperature ({max_temp:.1f}°C), adding cooling delay...")
                        time.sleep(1.0)  # Moderate cooling delay
        except Exception:
            # Temperature monitoring not available on this system
            pass

    except Exception as e:
        # If monitoring fails, don't crash - just log and continue
        logging.getLogger(__name__).debug(f"Thermal monitoring failed: {e}")
