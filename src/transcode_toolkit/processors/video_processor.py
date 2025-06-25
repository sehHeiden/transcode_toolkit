"""Video processing implementation."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from transcode_toolkit.core import BackupStrategy, FileManager, MediaProcessor, ProcessingResult, ProcessingStatus
from transcode_toolkit.video import transcode

LOG = logging.getLogger(__name__)


class VideoProcessor(MediaProcessor):
    """Video processor for HEVC transcoding."""

    def __init__(self, file_manager: FileManager | None = None) -> None:
        super().__init__("VideoProcessor")
        self.file_manager = file_manager or FileManager(BackupStrategy.ON_SUCCESS)
        self.video_extensions = {
            ".mp4",
            ".mkv",
            ".avi",
            ".mov",
            ".wmv",
            ".flv",
            ".webm",
            ".m4v",
        }

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a supported video format."""
        return file_path.suffix.lower() in self.video_extensions

    def should_process(self, file_path: Path, **kwargs) -> bool:
        """Check if video should be processed."""
        if not self.can_process(file_path):
            return False

        try:
            import json
            import subprocess

            # Get video metadata
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_entries",
                "stream=codec_name,bit_rate,height",
                "-select_streams",
                "v:0",
                str(file_path),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
                errors="replace",
            )
            data = json.loads(result.stdout)

            if not data.get("streams"):
                return False

            stream = data["streams"][0]
            meta = {
                "codec_name": stream.get("codec_name", ""),
                "bit_rate": int(stream.get("bit_rate", 0)),
                "height": int(stream.get("height", 1080)),
            }

            # Use the transcode module's logic
            return not transcode._should_skip(meta)

        except Exception as e:
            LOG.warning(f"Failed to analyze {file_path}: {e}")
            return False

    def process_file(self, file_path: Path, **kwargs) -> ProcessingResult:
        """Process a single video file."""
        if not self.should_process(file_path, **kwargs):
            return ProcessingResult(
                source_file=file_path,
                status=ProcessingStatus.SKIPPED,
                message="File already optimized or doesn't meet processing criteria",
            )

        original_size = file_path.stat().st_size
        crf = kwargs.get("crf", 24)
        gpu = kwargs.get("gpu", False)

        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                temp_path = Path(tmp.name)

            # Transcode the video
            success = transcode.transcode_video(input_path=file_path, output_path=temp_path, crf=crf, gpu=gpu)

            if not success:
                temp_path.unlink(missing_ok=True)
                return ProcessingResult(
                    source_file=file_path,
                    status=ProcessingStatus.FAILED,
                    message="FFmpeg transcoding failed",
                    original_size=original_size,
                )

            new_size = temp_path.stat().st_size

            # Check if transcoding actually saved space
            size_ratio = new_size / original_size if original_size > 0 else 1.0
            min_ratio = 0.95  # Only keep if at least 5% smaller

            if size_ratio >= min_ratio:
                temp_path.unlink(missing_ok=True)
                return ProcessingResult(
                    source_file=file_path,
                    status=ProcessingStatus.SKIPPED,
                    message=f"No significant size reduction ({size_ratio * 100:.1f}% of original)",
                    original_size=original_size,
                    new_size=new_size,
                )

            # Replace original with transcoded version
            operation = self.file_manager.atomic_replace(file_path, temp_path)

            if operation.success:
                return ProcessingResult(
                    source_file=file_path,
                    status=ProcessingStatus.SUCCESS,
                    message="Successfully transcoded to HEVC",
                    output_file=operation.target_path,
                    original_size=original_size,
                    new_size=new_size,
                )
            return ProcessingResult(
                source_file=file_path,
                status=ProcessingStatus.FAILED,
                message="Failed to replace original file",
                original_size=original_size,
                new_size=new_size,
            )

        except Exception as e:
            LOG.exception(f"Error processing {file_path}: {e}")
            return ProcessingResult(
                source_file=file_path,
                status=ProcessingStatus.ERROR,
                message=str(e),
                original_size=original_size,
            )
