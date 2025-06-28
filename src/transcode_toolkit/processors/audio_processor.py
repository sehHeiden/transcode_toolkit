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
        from ..core.audio_analysis import analyze_folder_snr
        from ..core.directory_processor import process_directory_unified
        
        # Extract preset from kwargs
        preset = kwargs.pop("preset", "music")
        
        return process_directory_unified(
            processor=self,
            directory=directory,
            recursive=recursive,
            media_type="audio",
            preset=preset,
            folder_analysis_func=analyze_folder_snr,
            **kwargs
        )

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

        with (
            tqdm(
                total=len(all_files),
                desc="Analyzing files",
                unit="file",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ) as progress,
            ThreadPoolExecutor(max_workers=analysis_workers) as executor,
        ):
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
                    for entries in temps.values():
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
