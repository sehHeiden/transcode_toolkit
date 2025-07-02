"""Video processing CLI commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..failure_table import print_failure_table

if TYPE_CHECKING:
    import argparse

    from ...core import ConfigManager

LOG = logging.getLogger(__name__)


class VideoCommands:
    """Video processing command handlers."""

    def __init__(self, config_manager: ConfigManager) -> None:
        """Initialize video commands handler."""
        self.config_manager = config_manager
        self._working_presets_cache: list[str] | None = None

    def _get_working_presets(self) -> list[str]:
        """Get available presets from config.yaml, filtered by codec availability."""
        if self._working_presets_cache is not None:
            return self._working_presets_cache

        from ...core.ffmpeg import FFmpegProcessor

        # Get all presets from the loaded configuration
        all_presets = list(self.config_manager.config.video.presets.keys())

        # Initialize FFmpeg processor for codec validation
        ffmpeg = FFmpegProcessor()

        # Filter presets based on codec availability
        working_presets = []
        filtered_codecs = set()

        for preset_name in all_presets:
            try:
                preset_config = self.config_manager.config.get_video_preset(preset_name)
                codec = preset_config.codec

                # Check if codec is available
                is_available, error_msg = ffmpeg.validate_codec(codec)

                if is_available:
                    working_presets.append(preset_name)
                else:
                    filtered_codecs.add(f"{codec} ({error_msg})")
                    LOG.debug(f"Filtering preset '{preset_name}': {error_msg}")

            except Exception as e:
                LOG.warning(f"Error checking preset '{preset_name}': {e}")

        # Log filtered codecs for user information
        if filtered_codecs:
            LOG.info(f"Filtered out presets using unavailable codecs: {', '.join(sorted(filtered_codecs))}")

        # Ensure "default" is first if it's available
        if "default" in working_presets:
            working_presets.remove("default")
            working_presets.insert(0, "default")

        self._working_presets_cache = working_presets
        LOG.info(f"Available presets after codec filtering: {len(working_presets)} out of {len(all_presets)} total")
        LOG.debug(f"Working presets: {', '.join(working_presets[:10])}{'...' if len(working_presets) > 10 else ''}")
        return working_presets

    def add_subcommands(self, parser: argparse.ArgumentParser) -> None:
        """Add video subcommands to parser."""
        subparsers = parser.add_subparsers(dest="video_command", help="Video commands")

        # Transcode command
        transcode_parser = subparsers.add_parser("transcode", help="Transcode video files")
        transcode_parser.add_argument("path", type=Path, help="Path to video file or directory")

        # Filter presets to only show working ones
        available_presets = self._get_working_presets()
        transcode_parser.add_argument(
            "--preset",
            "-p",
            choices=available_presets,
            default="default",
            help=f"Encoding preset: {', '.join(available_presets)}",
        )

        # Override options (optional)
        transcode_parser.add_argument("--crf", "-c", type=int, help="Override CRF value from preset")
        transcode_parser.add_argument(
            "--gpu", "-g", action="store_true", help="Use GPU acceleration (overrides preset codec)"
        )

        transcode_parser.add_argument(
            "--recursive",
            "-r",
            action="store_true",
            help="Process directories recursively",
        )
        transcode_parser.add_argument("--no-backups", action="store_true", help="Don't create backup files")

        # Estimate command
        estimate_parser = subparsers.add_parser("estimate", help="Estimate size savings")
        estimate_parser.add_argument("path", type=Path, help="Path to video file or directory")
        estimate_parser.add_argument("--csv", help="Save results to CSV file")
        estimate_parser.add_argument(
            "--save-settings", action="store_true", help="Save optimal settings for future transcoding"
        )

    def handle_command(self, args: argparse.Namespace) -> int:
        """Handle video command execution."""
        if not hasattr(args, "video_command") or args.video_command is None:
            LOG.error("No video command specified")
            return 1

        if args.video_command == "transcode":
            return self._handle_transcode(args)
        if args.video_command == "estimate":
            return self._handle_estimate(args)
        LOG.error("Unknown video command: %s", args.video_command)
        return 1

    def _handle_transcode(self, args: argparse.Namespace) -> int:
        """Handle video transcoding."""
        try:
            from ...core.ffmpeg import FFmpegProcessor
            from ...processors import VideoProcessor

            # Get preset configuration
            preset_name = getattr(args, "preset", "default")
            preset_config = self.config_manager.config.get_video_preset(preset_name)

            # Validate codec availability
            ffmpeg = FFmpegProcessor()
            codec_to_check = preset_config.codec

            # Override with GPU codec if --gpu flag is used
            if getattr(args, "gpu", False):
                codec_to_check = "hevc_nvenc"  # Default GPU codec

            # Validate codec availability (this should rarely fail since we filter presets)
            is_available, error_msg = ffmpeg.validate_codec(codec_to_check)
            if not is_available:
                LOG.error(f"Cannot use preset '{preset_name}': {error_msg}")
                working_presets = self._get_working_presets()
                LOG.info(f"Available presets on your system: {', '.join(working_presets)}")
                return 1

            processor = VideoProcessor(self.config_manager)

            # Prepare kwargs for processing
            process_kwargs = {
                "preset": preset_name,
                "gpu": getattr(args, "gpu", False),
            }

            # Add parameter overrides if specified
            if hasattr(args, "crf") and args.crf is not None:
                process_kwargs["force_crf"] = args.crf

            # Pass preset name for forced preset usage
            process_kwargs["force_preset"] = preset_name

            if args.path.is_file():
                # Extract and cast the values from process_kwargs
                preset_val = process_kwargs.get("preset", "default")
                gpu_val = process_kwargs.get("gpu", False)
                force_crf_val = process_kwargs.get("force_crf")
                force_preset_val = process_kwargs.get("force_preset")

                # Ensure proper types
                preset_str = str(preset_val) if preset_val is not None else "default"
                gpu_bool = bool(gpu_val) if gpu_val is not None else False
                force_crf_int = int(force_crf_val) if force_crf_val is not None else None
                force_preset_str = str(force_preset_val) if force_preset_val is not None else None

                result = processor.process_file(
                    args.path, preset=preset_str, gpu=gpu_bool, force_crf=force_crf_int, force_preset=force_preset_str
                )
                if result.status.value == "success":
                    LOG.info(f"Successfully processed {args.path} with preset '{preset_name}'")
                    return 0
                LOG.error("Failed to process %s: %s", args.path, result.message)
                return 1

            results = processor.process_directory(
                args.path, recursive=getattr(args, "recursive", True), **process_kwargs
            )
            successful = [r for r in results if r.status.value == "success"]
            failed = [r for r in results if r.status.value == "error"]
            skipped = [r for r in results if r.status.value == "skipped"]

            LOG.info(f"Processed {len(successful)}/{len(results)} files successfully with preset '{preset_name}'")

            # Show failure table if there are failures
            if failed:
                print_failure_table(failed, "video")

            return 0 if len(successful) == len(results) else 1

        except Exception:
            LOG.exception("Video transcoding failed")
            return 1

    def _handle_estimate(self, args: argparse.Namespace) -> int:
        """Handle detailed video and audio estimation with per-file analysis."""
        try:
            from ...core.unified_estimate import analyze_directory, print_detailed_summary

            # Use unified estimate with detailed per-file analysis
            save_settings = getattr(args, "save_settings", False)
            csv_path = getattr(args, "csv", None)

            # Check if we're in verbose mode
            verbose_mode = getattr(args, "verbose", 0) > 0

            if not verbose_mode:
                print("📊 Analyzing media files for optimization opportunities...")
                print("💡 Use -v for detailed progress information")

            analyses, optimal_presets = analyze_directory(args.path, save_settings=save_settings)
            print_detailed_summary(analyses, optimal_presets, csv_path=csv_path)
            return 0

        except Exception:
            LOG.exception("Video estimation failed")
            return 1
