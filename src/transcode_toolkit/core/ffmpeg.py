"""FFmpeg integration and utilities."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from typing import TYPE_CHECKING, Any

from .base import ProcessingError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

LOG = logging.getLogger(__name__)


def _parse_frame_rate(frame_rate_str: str) -> float:
    """Safely parse frame rate from fraction string like '30/1' or '29.97'."""
    try:
        if "/" in frame_rate_str:
            numerator, denominator = frame_rate_str.split("/", 1)
            return float(numerator) / float(denominator) if float(denominator) != 0 else 0.0
        return float(frame_rate_str)
    except (ValueError, ZeroDivisionError):
        return 0.0


class FFmpegError(ProcessingError):
    """FFmpeg-specific error."""

    def __init__(
        self,
        message: str,
        command: list[str] | None = None,
        return_code: int | None = None,
        stderr: str | None = None,
        stdout: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(message, **kwargs)
        self.command = command
        self.return_code = return_code
        self.stderr = stderr
        self.stdout = stdout


class FFmpegProbe:
    """FFmpeg probe utility for media file analysis."""

    @staticmethod
    def check_availability() -> None:
        """Check if FFmpeg tools are available."""
        required = ["ffmpeg", "ffprobe"]
        missing = [exe for exe in required if not shutil.which(exe)]

        if missing:
            error_msg = f"Missing FFmpeg executables: {', '.join(missing)}"
            LOG.error(error_msg)
            raise FFmpegError(error_msg)

    @staticmethod
    def probe_media(file_path: Path, stream_type: str | None = None) -> dict[str, Any]:
        """Probe media file for metadata."""
        FFmpegProbe.check_availability()

        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
        ]

        if stream_type:
            cmd.extend(["-select_streams", f"{stream_type}:0"])

        cmd.append(str(file_path))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
                encoding="utf-8",
                errors="replace",
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            # Capture both stderr and stdout for detailed error info
            error_details = e.stderr or e.stdout or "No error output"
            msg = f"ffprobe failed for {file_path}: {error_details.strip()}"
            raise FFmpegError(
                msg,
                command=cmd,
                return_code=e.returncode,
                file_path=file_path,
                stderr=e.stderr,
                stdout=e.stdout,
            )
        except subprocess.TimeoutExpired:
            msg = f"ffprobe timed out for {file_path}"
            raise FFmpegError(msg, command=cmd, file_path=file_path)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON from ffprobe for {file_path}: {e}"
            raise FFmpegError(
                msg,
                command=cmd,
                file_path=file_path,
            )

    @staticmethod
    def get_audio_info(file_path: Path) -> dict[str, Any]:
        """Get audio stream information."""
        data = FFmpegProbe.probe_media(file_path, "a")

        if not data.get("streams"):
            msg = f"No audio streams found in {file_path}"
            raise FFmpegError(msg, file_path=file_path)

        stream = data["streams"][0]
        format_info = data.get("format", {})

        return {
            "codec": stream.get("codec_name"),
            "bitrate": stream.get("bit_rate") or format_info.get("bit_rate"),
            "duration": float(format_info.get("duration", 0)),
            "channels": stream.get("channels"),
            "sample_rate": stream.get("sample_rate"),
            "size": int(format_info.get("size", file_path.stat().st_size)),
        }

    @staticmethod
    def estimate_snr(file_path: Path, audio_info: dict[str, Any] | None = None) -> float:
        """
        Estimate Signal-to-Noise Ratio based on codec and bitrate.

        Args:
            file_path: Path to audio file
            audio_info: Pre-computed audio info (optional)

        Returns:
            Estimated SNR in dB

        """
        if audio_info is None:
            try:
                audio_info = FFmpegProbe.get_audio_info(file_path)
            except Exception as e:
                LOG.warning(f"Failed to get audio info for {file_path}: {e}")
                return 60.0  # Conservative default

        codec = audio_info.get("codec", "").lower()
        bitrate = audio_info.get("bitrate")

        # Convert bitrate to int if it's a string
        if bitrate and isinstance(bitrate, str):
            try:
                bitrate = int(bitrate)
            except ValueError:
                bitrate = None
        elif bitrate:
            bitrate = int(bitrate)

        # Estimate SNR based on codec and bitrate
        if codec in ["flac", "alac", "wav", "aiff"]:
            return 96.0  # 16-bit lossless theoretical maximum
        if codec in ["dsd", "dsf", "dff"]:
            return 120.0  # DSD has very high SNR
        if codec == "mp3":
            if bitrate and bitrate >= 320000:
                return 85.0  # High quality MP3
            if bitrate and bitrate >= 192000:
                return 75.0  # Good quality MP3
            if bitrate and bitrate >= 128000:
                return 65.0  # Standard quality MP3
            return 55.0  # Lower quality MP3
        if codec in ["aac", "m4a"]:
            if bitrate and bitrate >= 256000:
                return 80.0  # High quality AAC
            if bitrate and bitrate >= 128000:
                return 70.0  # Good quality AAC
            if bitrate and bitrate >= 96000:
                return 60.0  # Standard quality AAC
            return 50.0  # Lower quality AAC
        if codec in ["opus", "ogg", "vorbis"]:
            if bitrate and bitrate >= 192000:
                return 85.0  # High quality Opus/Ogg
            if bitrate and bitrate >= 128000:
                return 75.0  # Good quality Opus/Ogg
            if bitrate and bitrate >= 96000:
                return 65.0  # Standard quality Opus/Ogg
            return 55.0  # Lower quality Opus/Ogg
        if codec in ["wma", "wmv"]:
            if bitrate and bitrate >= 192000:
                return 75.0  # High quality WMA
            if bitrate and bitrate >= 128000:
                return 65.0  # Good quality WMA
            return 55.0  # Lower quality WMA
        # Unknown codec - conservative estimate
        if bitrate and bitrate >= 256000:
            return 70.0
        if bitrate and bitrate >= 128000:
            return 60.0
        return 50.0

    @staticmethod
    def get_video_info(file_path: Path) -> dict[str, Any]:
        """Get video stream information."""
        data = FFmpegProbe.probe_media(file_path, "v")

        if not data.get("streams"):
            msg = f"No video streams found in {file_path}"
            raise FFmpegError(msg, file_path=file_path)

        stream = data["streams"][0]
        format_info = data.get("format", {})

        return {
            "codec": stream.get("codec_name"),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "bitrate": stream.get("bit_rate") or format_info.get("bit_rate"),
            "duration": float(format_info.get("duration", 0)),
            "fps": _parse_frame_rate(stream.get("r_frame_rate", "0/1")),  # Convert fraction to float safely
            "size": int(format_info.get("size", file_path.stat().st_size)),
        }


class FFmpegProcessor:
    """FFmpeg command executor with enhanced error handling."""

    def __init__(self, timeout: int = 300) -> None:
        self.timeout = timeout

    def run_command(
        self,
        command: list[str],
        file_path: Path | None = None,
        progress_callback: Callable | None = None,
    ) -> subprocess.CompletedProcess:
        """Run FFmpeg command with proper error handling."""
        FFmpegProbe.check_availability()

        LOG.debug(f"Running FFmpeg command: {' '.join(command)}")
        start_time = time.time()

        try:
            # Use stderr=subprocess.PIPE to capture progress info if needed
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,  # We'll handle return code ourselves
                encoding="utf-8",
                errors="replace",
            )

            processing_time = time.time() - start_time
            LOG.debug(f"FFmpeg command completed in {processing_time:.2f}s")

            if result.returncode != 0:
                error_msg = f"FFmpeg failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr.strip()}"

                raise FFmpegError(
                    error_msg,
                    command=command,
                    return_code=result.returncode,
                    file_path=file_path,
                )

            return result

        except subprocess.TimeoutExpired:
            msg = f"FFmpeg command timed out after {self.timeout}s"
            raise FFmpegError(
                msg,
                command=command,
                file_path=file_path,
            )
        except Exception as e:
            if isinstance(e, FFmpegError):
                raise
            msg = f"Unexpected error running FFmpeg: {e}"
            raise FFmpegError(
                msg,
                command=command,
                file_path=file_path,
                cause=e,
            )

    def build_audio_command(
        self,
        input_file: Path,
        output_file: Path,
        codec: str = "libopus",
        bitrate: str = "128k",
        **kwargs,
    ) -> list[str]:
        """Build FFmpeg command for audio transcoding."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_file),
            "-c:a",
            codec,
            "-b:a",
            bitrate,
        ]

        # Add additional audio parameters
        if "application" in kwargs and kwargs["application"] is not None:
            cmd.extend(["-application", kwargs["application"]])
        if "cutoff" in kwargs and kwargs["cutoff"] is not None:
            cmd.extend(["-cutoff", kwargs["cutoff"]])
        if "channels" in kwargs and kwargs["channels"] is not None:
            cmd.extend(["-ac", str(kwargs["channels"])])
        if "vbr" in kwargs and kwargs["vbr"] is not None:
            cmd.extend(["-vbr", kwargs["vbr"]])
        if "compression_level" in kwargs and kwargs["compression_level"] is not None:
            cmd.extend(["-compression_level", str(kwargs["compression_level"])])

        cmd.append(str(output_file))
        return cmd

    def build_video_command(
        self,
        input_file: Path,
        output_file: Path,
        codec: str = "libx265",
        crf: int = 24,
        preset: str = "medium",
        **kwargs,
    ) -> list[str]:
        """Build FFmpeg command for video transcoding."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_file),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
        ]

        # Video encoding parameters
        if kwargs.get("gpu"):
            cmd.extend(["-c:v", "hevc_nvenc", "-preset", "p5", "-cq", str(crf)])
        else:
            cmd.extend(["-c:v", codec, "-preset", preset])
            if codec == "libx265":
                cmd.extend(["-x265-params", f"crf={crf}"])
            else:
                cmd.extend(["-crf", str(crf)])

        # Audio handling (usually copy)
        cmd.extend(["-c:a", kwargs.get("audio_codec", "copy")])

        cmd.append(str(output_file))
        return cmd
