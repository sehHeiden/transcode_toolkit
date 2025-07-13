"""Video processor implementation using the new core architecture with SSIM-based quality analysis."""

from __future__ import annotations

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

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
from ..core.directory_processor import process_directory_unified
from ..core.video_analysis import analyze_folder_quality

if TYPE_CHECKING:
    from pathlib import Path


class VideoProcessor(MediaProcessor):
    """Video transcoding processor using H.265/HEVC codec with SSIM-based quality optimization."""

    def __init__(self, config_manager: ConfigManager) -> None:
        """Initialize video processor with config manager."""
        super().__init__("VideoProcessor")
        self.config_manager = config_manager
        # Use much longer timeout for video processing (30 minutes default)
        self.ffmpeg = FFmpegProcessor(timeout=1800)

        # Initialize file manager with backup strategy from config
        backup_strategy_str = config_manager.get_value("global_.cleanup_backups", "on_success")
        backup_strategy = BackupStrategy(backup_strategy_str)

        self.file_manager = FileManager(backup_strategy)

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a supported video format."""
        video_extensions = self.config_manager.get_value("video.extensions", set())
        if not isinstance(video_extensions, set):
            try:
                # Check if it's iterable before converting to set
                if hasattr(video_extensions, "__iter__") and not isinstance(video_extensions, (str, bytes)):
                    video_extensions = set(video_extensions)
                else:
                    video_extensions = set()
            except (TypeError, ValueError):
                video_extensions = set()
        return file_path.suffix.lower() in video_extensions

    def should_process(self, file_path: Path, **_kwargs: object) -> bool:
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
                            "Skipping %s: Already optimal %s (%.1fMbps vs target %.1fMbps)",
                            file_path.name,
                            codec.upper(),
                            current_bps / 1_000_000,
                            threshold / 1_000_000,
                        )
                        return False

                    # Allow reprocessing if significantly higher bitrate
                    self.logger.info(
                        "Will reprocess %s: %s %.1fMbps → %.1fMbps",
                        file_path.name,
                        codec.upper(),
                        current_bps / 1_000_000,
                        threshold / 1_000_000,
                    )
                    return True

                # If bitrate is unknown, conservatively skip
                self.logger.info("Skipping %s: Already %s format (bitrate unknown)", file_path.name, codec.upper())
                return False

        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
            self.logger.warning("Could not analyze %s: %s", file_path, e)
            return False
        else:
            return True

    def process_file(self, file_path: Path, **kwargs: object) -> ProcessingResult:  # noqa: PLR0915
        """Process a single video file with SSIM-guided optimization."""
        start_time = time.time()

        try:
            # Get preset configuration
            preset = str(kwargs.get("preset", "default"))
            preset_config = self.config_manager.config.get_video_preset(preset)

            # Validate codec availability before proceeding
            codec_to_validate = preset_config.codec

            # Override with GPU codec if --gpu flag is used
            if kwargs.get("gpu", False):
                # Try to use a GPU codec instead of the preset codec
                gpu_codec_map = {
                    "libx265": "hevc_nvenc",
                    "libx264": "h264_nvenc",
                    "libaom-av1": "av1_nvenc",  # If available
                }
                codec_to_validate = gpu_codec_map.get(preset_config.codec, "hevc_nvenc")

            is_available, error_msg = self.ffmpeg.validate_codec(codec_to_validate)
            if not is_available:
                return ProcessingResult(
                    source_file=file_path,
                    status=ProcessingStatus.ERROR,
                    message=f"Codec validation failed: {error_msg}",
                    processing_time=0.0,
                    metadata={"preset": preset, "codec_error": error_msg},
                )

            # Get file info and analyze content
            video_info = FFmpegProbe.get_video_info(file_path)
            original_size = video_info["size"]

            # Create temporary output file
            temp_file = file_path.with_suffix(".tmp.mp4")

            # Use config parameters directly for consistency with estimation
            # Only override if explicitly forced via command line
            force_crf = kwargs.get("force_crf")
            force_preset = kwargs.get("force_preset")

            effective_crf = (
                int(force_crf) if force_crf is not None and isinstance(force_crf, (int, str)) else preset_config.crf
            )
            effective_preset = force_preset if force_preset is not None else preset_config.preset

            # Build and run FFmpeg command
            gpu = kwargs.get("gpu", False)
            command = self.ffmpeg.build_video_command(
                input_file=file_path,
                output_file=temp_file,
                codec=preset_config.codec,
                crf=int(effective_crf),
                preset=str(effective_preset),
                preset_config=preset_config,
                gpu=gpu,
            )

            # Calculate dynamic timeout based on file duration
            duration_minutes = video_info.get("duration", 0) / 60
            # Allow 3x realtime for GPU, 5x for CPU encoding, minimum 10 minutes
            timeout_factor = 3 if gpu else 5
            dynamic_timeout = max(600, int(duration_minutes * 60 * timeout_factor))

            # Use dynamic timeout for this specific command
            original_timeout = self.ffmpeg.timeout
            self.ffmpeg.timeout = dynamic_timeout

            try:
                # Execute transcoding
                self.ffmpeg.run_command(command, file_path)
            finally:
                # Restore original timeout
                self.ffmpeg.timeout = original_timeout
            processing_time = time.time() - start_time

            # Check output file
            if not temp_file.exists():
                msg = f"Output file not created: {temp_file}"
                raise ProcessingError(msg)

            new_size = temp_file.stat().st_size
            self.config_manager.get_value("video.size_keep_ratio", 0.95)

            # Calculate actual savings percentage (positive = savings, negative = increase)
            savings_percent = ((original_size - new_size) / original_size * 100) if original_size > 0 else 0
            min_savings_percent = self.config_manager.get_value("video.min_savings_percent", 10.0)

            # Log the actual result for debugging
            if savings_percent < 0:
                self.logger.warning(
                    "File %s INCREASED by %.1f%% (%s → %s bytes)",
                    file_path.name,
                    abs(savings_percent),
                    f"{original_size:,}",
                    f"{new_size:,}",
                )
            else:
                self.logger.info(
                    "File %s savings: %.1f%% (%s → %s bytes)",
                    file_path.name,
                    savings_percent,
                    f"{original_size:,}",
                    f"{new_size:,}",
                )

            # Check for minimum savings requirement (must be positive and >= threshold)
            meets_savings_requirement = (
                savings_percent >= float(min_savings_percent)
                if isinstance(min_savings_percent, (str, int, float))
                else False
            )

            # Always convert if source is not target codec
            source_codec = video_info.get("codec", "").lower()
            target_preset = self.config_manager.config.get_video_preset(preset)
            target_codec = target_preset.codec.lower()
            is_non_target_codec = source_codec != target_codec

            # CRITICAL: Check if file size actually increased (negative savings)
            if savings_percent < 0:
                # File got bigger - always restore backup regardless of codec
                self.logger.error(
                    "File %s increased in size by %.1f%% - this should never happen!",
                    file_path.name,
                    abs(savings_percent),
                )
                temp_file.unlink()  # Remove the bad transcoded file

                return ProcessingResult(
                    source_file=file_path,
                    status=ProcessingStatus.SKIPPED,
                    message=f"File size increased by {abs(savings_percent):.1f}% - transcoding failed",
                    original_size=original_size,
                    new_size=new_size,
                    processing_time=processing_time,
                    metadata={
                        "preset": preset,
                        "crf": effective_crf,
                        "size_reduction_mb": (original_size - new_size) / (1024 * 1024),
                        "size_improvement": savings_percent,
                        "reason": "file_size_increased",
                        "error": "transcoding_made_file_bigger",
                    },
                )

            # Check safeguards BEFORE replacing the original file
            if meets_savings_requirement or is_non_target_codec:
                # Savings requirement met or codec conversion - proceed with replacement
                create_backups = self.config_manager.get_value("global_.create_backups", default=True)
                operation = self.file_manager.atomic_replace(
                    source_path=file_path,
                    temp_path=temp_file,
                    create_backup=bool(create_backups),
                )

                # Get final file size for metadata
                final_size = operation.target_path.stat().st_size if operation.target_path else 0
                final_savings_percent = ((original_size - final_size) / original_size * 100) if original_size > 0 else 0

                return ProcessingResult(
                    source_file=file_path,
                    status=ProcessingStatus.SUCCESS,
                    message=(
                        f"Transcoded with {preset} preset (CRF {effective_crf}, {final_savings_percent:.1f}% savings)"
                    ),
                    output_file=operation.target_path,
                    original_size=original_size,
                    new_size=final_size,
                    processing_time=processing_time,
                    metadata={
                        "preset": preset,
                        "codec": preset_config.codec,
                        "crf": effective_crf,
                        "size_reduction_mb": (original_size - final_size) / (1024 * 1024),
                        "backup_created": operation.backup_path is not None,
                        "savings_percent": final_savings_percent,
                    },
                )

            # SAFEGUARD: Insufficient savings - don't replace original file
            # Remove the transcoded temp file since we're not using it
            temp_file.unlink()

            skip_reason = f"Insufficient savings ({savings_percent:.1f}% < {min_savings_percent}% threshold)"
            self.logger.warning("File %s: %s", file_path.name, skip_reason)

            return ProcessingResult(
                source_file=file_path,
                status=ProcessingStatus.SKIPPED,
                message=skip_reason,
                original_size=original_size,
                new_size=new_size,
                processing_time=processing_time,
                metadata={
                    "preset": preset,
                    "crf": effective_crf,
                    "size_reduction_mb": (original_size - new_size) / (1024 * 1024),
                    "size_improvement": savings_percent,
                    "meets_savings_requirement": False,
                    "reason": "insufficient_savings",
                },
            )

        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
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

    def process_directory(
        self,
        directory: Path,
        *,
        recursive: bool = True,
        **kwargs: object,
    ) -> list[ProcessingResult]:
        """Process all video files in directory with multithreading."""
        # Extract preset from kwargs
        preset = kwargs.pop("preset", "default")

        from ..core.directory_processor import DirectoryProcessingConfig

        config = DirectoryProcessingConfig(
            recursive=recursive,
            media_type="video",
            preset=str(preset),
            folder_analysis_func=analyze_folder_quality,
        )
        results = process_directory_unified(
            processor=self,
            directory=directory,
            config=config,
            **kwargs,
        )

        # Additional summary for video processing
        successful = [r for r in results if r.status.value == "success"]
        skipped = [r for r in results if r.status.value == "skipped"]
        errors = [r for r in results if r.status.value == "error"]

        self.logger.info(
            "Video processing completed: %d successful, %d skipped, %d errors",
            len(successful),
            len(skipped),
            len(errors),
        )

        return results

    def _analyze_files_parallel(
        self, all_files: list[Path], _preset: str, max_workers: int, **kwargs: object
    ) -> list[Path]:
        """Analyze files in parallel to determine which need processing."""
        self.logger.info("Analyzing %d files to determine processing needs...", len(all_files))

        files_to_process: list[Path] = []

        # Use smaller worker count for analysis to avoid overwhelming system
        analysis_workers = min(max_workers, 4)

        # Constants for magic values
        small_batch_threshold = 10

        if analysis_workers == 1 or len(all_files) < small_batch_threshold:
            # Single-threaded analysis for small sets
            files_to_process.extend(file_path for file_path in all_files if self.should_process(file_path, **kwargs))
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
                    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
                        self.logger.warning("Error analyzing %s: %s", file_path, e)

        self.logger.info("Analysis complete: %d files need processing", len(files_to_process))
        return files_to_process
