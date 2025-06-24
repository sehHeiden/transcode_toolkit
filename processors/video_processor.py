"""Video processing implementation."""

from __future__ import annotations
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

from core import MediaProcessor, ProcessingResult, ProcessingStatus
from core import FileManager, BackupStrategy
from video import transcode

LOG = logging.getLogger(__name__)


class VideoProcessor(MediaProcessor):
    """Video processor for HEVC transcoding."""

    def __init__(self, file_manager: Optional[FileManager] = None):
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
            import subprocess
            import json

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

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
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
            success = transcode.transcode_video(
                input_path=file_path, output_path=temp_path, crf=crf, gpu=gpu
            )

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
            else:
                return ProcessingResult(
                    source_file=file_path,
                    status=ProcessingStatus.FAILED,
                    message="Failed to replace original file",
                    original_size=original_size,
                    new_size=new_size,
                )

        except Exception as e:
            LOG.error(f"Error processing {file_path}: {e}")
            return ProcessingResult(
                source_file=file_path,
                status=ProcessingStatus.ERROR,
                message=str(e),
                original_size=original_size,
            )


class VideoEstimator:
    """Video size estimation utility."""

    def __init__(self):
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

    def estimate_size(self, file_path: Path, **kwargs) -> int:
        """Estimate size after HEVC transcoding."""
        from video import estimate

        try:
            meta = estimate._probe(file_path)
            return estimate._estimate(meta)
        except Exception as e:
            LOG.warning(f"Failed to estimate size for {file_path}: {e}")
            return file_path.stat().st_size  # Return original size as fallback

    def estimate_directory(self, directory: Path, **kwargs) -> Dict[str, Any]:
        """Estimate sizes for all videos in directory."""
        from video import estimate

        try:
            rows = estimate.analyse(directory)

            current_total = sum(row[1] for row in rows)
            estimated_total = sum(row[2] for row in rows)
            savings = current_total - estimated_total

            return {
                "files": len(rows),
                "current_size": current_total,
                "estimated_size": estimated_total,
                "potential_savings": savings,
                "savings_percent": (savings / current_total * 100)
                if current_total > 0
                else 0,
                "details": rows,
            }
        except Exception as e:
            LOG.error(f"Failed to estimate directory {directory}: {e}")
            return {
                "files": 0,
                "current_size": 0,
                "estimated_size": 0,
                "potential_savings": 0,
                "savings_percent": 0,
                "details": [],
            }
