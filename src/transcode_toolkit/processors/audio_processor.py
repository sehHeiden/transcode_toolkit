"""Audio processor implementation using the new core architecture."""

from __future__ import annotations

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

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
from ..core.audio_analysis import analyze_folder_snr
from ..core.directory_processor import process_directory_unified

if TYPE_CHECKING:
    from pathlib import Path

    from ..config.settings import AudioPreset


class AudioProcessor(MediaProcessor):
    """Audio transcoding processor using Opus codec."""

    def __init__(self, config_manager: ConfigManager) -> None:
        """Initialize audio processor with config manager."""
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
        if not isinstance(audio_extensions, set):
            try:
                # Check if it's iterable before converting to set
                if hasattr(audio_extensions, "__iter__") and not isinstance(audio_extensions, (str, bytes)):
                    audio_extensions = set(audio_extensions)
                else:
                    audio_extensions = set()
            except (TypeError, ValueError):
                audio_extensions = set()
        return file_path.suffix.lower() in audio_extensions

    def should_process(self, file_path: Path, **kwargs: object) -> bool:
        """Check if file should be processed (not already optimized)."""
        try:
            preset = str(kwargs.get("preset", "music"))
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
                            "Will reprocess %s: Opus %.0fk → %.0fk",
                            file_path.name,
                            current_bps / 1000,
                            target_bps / 1000,
                        )
                        return True
                    self.logger.info(
                        "Skipping %s: Already optimal Opus (%.0fk vs target %.0fk)",
                        file_path.name,
                        current_bps / 1000,
                        target_bps / 1000,
                    )
                    return False
                # If bitrate is unknown, conservatively skip
                self.logger.info("Skipping %s: Already Opus format (bitrate unknown)", file_path.name)
                return False

        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
            self.logger.warning("Could not analyze %s: %s", file_path, e)
            return False
        else:
            return True

    def process_file(self, file_path: Path, **kwargs: object) -> ProcessingResult:
        """Process a single audio file."""
        start_time = time.time()

        try:
            # Get preset configuration
            preset = str(kwargs.get("preset", "music"))
            preset_config = self.config_manager.config.get_audio_preset(preset)

            # Get file info
            audio_info = FFmpegProbe.get_audio_info(file_path)
            original_size = audio_info["size"]

            # Create temporary output file
            temp_file = file_path.with_suffix(".tmp.opus")

            # Determine effective bitrate using SNR-based intelligent limiting
            folder_snr = kwargs.get("folder_snr")
            folder_snr_value = folder_snr if isinstance(folder_snr, (float, int)) else None
            effective_bitrate = self._calculate_effective_bitrate(
                file_path, audio_info, preset_config, folder_snr_value
            )

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
                create_backups_value = self.config_manager.get_value("global_.create_backups", default=True)
                create_backups = bool(create_backups_value)
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

        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
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
        *,
        recursive: bool = True,
        **kwargs: object,
    ) -> list[ProcessingResult]:
        """Process all audio files in directory with multithreading."""
        # Extract preset from kwargs
        preset = kwargs.pop("preset", "music")

        from ..core.directory_processor import DirectoryProcessingConfig

        config = DirectoryProcessingConfig(
            recursive=recursive,
            media_type="audio",
            preset=str(preset),
            folder_analysis_func=analyze_folder_snr,
        )
        return process_directory_unified(
            processor=self,
            directory=directory,
            config=config,
            **kwargs,
        )

    def _calculate_effective_bitrate(
        self, file_path: Path, audio_info: dict[str, Any], preset_config: AudioPreset, folder_snr: float | None = None
    ) -> str:
        """
        Calculate effective bitrate using SNR-based intelligent limiting.

        Args:
            file_path: Path to the audio file
            audio_info: Audio metadata from FFprobe
            preset_config: AudioPreset configuration
            folder_snr: Optional folder-wide SNR analysis result

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
                        "Limiting bitrate for %s: %s → %s (input limit, SNR scaling disabled)",
                        file_path.name,
                        preset_config.bitrate,
                        effective_bitrate,
                    )
                    return effective_bitrate

            self.logger.debug("Using preset bitrate for %s: %s", file_path.name, preset_config.bitrate)
            return preset_config.bitrate

        # SNR scaling enabled - calculate effective bitrate
        estimated_snr = folder_snr if folder_snr is not None else FFmpegProbe.estimate_snr(file_path, audio_info)
        self.logger.debug("Estimated SNR for %s: %.1f dB", file_path.name, estimated_snr)

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
                "Limiting bitrate for %s: %s → %s (%s)",
                file_path.name,
                preset_config.bitrate,
                effective_bitrate,
                limitation_reason,
            )
        elif effective_bitrate != preset_config.bitrate:
            self.logger.info("Using adjusted bitrate for %s: %s", file_path.name, effective_bitrate)
        else:
            self.logger.debug("Using preset bitrate for %s: %s", file_path.name, preset_config.bitrate)

        return effective_bitrate

    def _analyze_files_parallel(
        self, all_files: list[Path], preset: str, max_workers: int, **kwargs: object
    ) -> list[Path]:
        """
        Analyze files in parallel to determine which ones need processing.

        This parallelizes the should_process check which involves FFmpeg calls.
        """
        files_to_process: list[Path] = []

        # Constants for magic values
        small_batch_threshold = 20

        if len(all_files) <= small_batch_threshold:
            # For small sets, use sequential processing (overhead isn't worth it)
            files_to_process.extend(
                file_path for file_path in all_files if self.should_process(file_path, preset=preset, **kwargs)
            )
            self.logger.info("Analysis complete: %d/%d files need processing", len(files_to_process), len(all_files))
            return files_to_process

        # For larger sets, use parallel analysis
        analysis_workers = min(max_workers, len(all_files) // 4)  # Don't use too many workers for analysis
        analysis_workers = max(1, analysis_workers)

        self.logger.info("Analyzing %d files using %d workers...", len(all_files), analysis_workers)

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
                except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
                    self.logger.warning("Failed to analyze %s: %s", file_path, e)
                finally:
                    progress.update(1)

        self.logger.info("Analysis complete: %d/%d files need processing", len(files_to_process), len(all_files))
        return files_to_process
