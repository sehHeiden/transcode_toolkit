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
    """FFmpeg probe utility for media file analysis with intelligent caching."""

    # OPTIMIZATION: Class-level cache for probe results
    _probe_cache: dict[tuple[Path, float, str], dict[str, Any]] = {}

    @staticmethod
    def check_availability() -> None:
        """Check if FFmpeg tools are available."""
        required = ["ffmpeg", "ffprobe"]
        missing = [exe for exe in required if not shutil.which(exe)]

        if missing:
            error_msg = f"Missing FFmpeg executables: {', '.join(missing)}"
            LOG.error(error_msg)
            raise FFmpegError(error_msg)

    @classmethod
    def _get_cache_key(cls, file_path: Path, stream_type: str | None) -> tuple[Path, float, str]:
        """Generate cache key based on file path, modification time, and stream type."""
        try:
            mtime = file_path.stat().st_mtime
            stream_key = stream_type or "all"
            return (file_path, mtime, stream_key)
        except OSError:
            # File doesn't exist or other error - return uncacheable key
            return (file_path, -1.0, stream_type or "all")

    @classmethod
    def probe_media(cls, file_path: Path, stream_type: str | None = None) -> dict[str, Any]:
        """Probe media file for metadata with caching."""
        # OPTIMIZATION: Check cache first
        cache_key = cls._get_cache_key(file_path, stream_type)
        if cache_key[1] >= 0 and cache_key in cls._probe_cache:
            return cls._probe_cache[cache_key]

        cls.check_availability()

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

            probe_data = json.loads(result.stdout)

            # OPTIMIZATION: Cache result if valid
            if cache_key[1] >= 0:
                cls._probe_cache[cache_key] = probe_data

            return probe_data
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
        self._available_encoders: dict[str, bool] | None = None

    def get_available_encoders(self) -> dict[str, bool]:
        """Get list of available encoders from FFmpeg."""
        if self._available_encoders is not None:
            return self._available_encoders

        try:
            result = subprocess.run(
                ["ffmpeg", "-encoders"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
                encoding="utf-8",
                errors="replace",
            )

            encoders = {}
            for line in result.stdout.split("\n"):
                # Look for encoder lines (they start with " V" for video or " A" for audio)
                if line.startswith((" V", " A")):  # Video or Audio encoder
                    parts = line.split()
                    if len(parts) >= 2:
                        encoder_name = parts[1]
                        encoders[encoder_name] = True

            self._available_encoders = encoders
            LOG.debug(f"Found {len(encoders)} available encoders")
            return encoders

        except Exception as e:
            LOG.warning(f"Failed to get encoder list: {e}")
            # Fallback: assume common encoders are available
            self._available_encoders = {
                "libx264": True,
                "libx265": True,
                "libopus": True,
                "h264_nvenc": False,
                "hevc_nvenc": False,
                "h264_amf": False,
                "hevc_amf": False,
                "h264_qsv": False,
                "hevc_qsv": False,
            }
            return self._available_encoders

    def is_encoder_available(self, encoder: str) -> bool:
        """Check if a specific encoder is available."""
        return self.get_available_encoders().get(encoder, False)

    def validate_codec(self, codec: str) -> tuple[bool, str]:
        """Validate if codec is available and return appropriate error message."""
        if not self.is_encoder_available(codec):
            return False, f"Encoder '{codec}' not available in your FFmpeg build"

        # For GPU encoders, test if hardware is actually available
        if codec in ["h264_nvenc", "hevc_nvenc", "h264_amf", "hevc_amf", "h264_qsv", "hevc_qsv"]:
            return self._test_gpu_encoder(codec)

        return True, ""

    def _test_gpu_encoder(self, codec: str) -> tuple[bool, str]:
        """Test if GPU encoder actually works by doing a quick encode test."""
        try:
            # Create a minimal test command
            test_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=320x240:rate=1",
                "-c:v",
                codec,
                "-t",
                "1",
                "-f",
                "null",
                "-",
            ]

            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                timeout=10,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

            if result.returncode == 0:
                return True, ""
            # Parse error message for specific hardware issues
            result.stderr.lower()
            if codec in ["h264_nvenc", "hevc_nvenc"]:
                return False, "NVIDIA GPU encoder not available (no compatible GPU or drivers)"
            if codec in ["h264_amf", "hevc_amf"]:
                return False, "AMD GPU encoder not available (no compatible GPU or drivers)"
            if codec in ["h264_qsv", "hevc_qsv"]:
                return False, "Intel QuickSync encoder not available (no compatible GPU or drivers)"
            return False, f"GPU encoder '{codec}' failed hardware test"

        except Exception as e:
            LOG.debug(f"GPU encoder test failed for {codec}: {e}")
            if codec in ["h264_nvenc", "hevc_nvenc"]:
                return False, "NVIDIA GPU encoder not available (no compatible GPU or drivers)"
            if codec in ["h264_amf", "hevc_amf"]:
                return False, "AMD GPU encoder not available (no compatible GPU or drivers)"
            if codec in ["h264_qsv", "hevc_qsv"]:
                return False, "Intel QuickSync encoder not available (no compatible GPU or drivers)"
            return False, f"GPU encoder '{codec}' test failed: {e}"

    def run_command(
        self,
        command: list[str],
        file_path: Path | None = None,
        progress_callback: Callable | None = None,
    ) -> subprocess.CompletedProcess:
        """Run FFmpeg command with proper error handling."""
        FFmpegProbe.check_availability()

        LOG.info(f"Running FFmpeg command: {' '.join(command)}")
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
        preset_config=None,  # VideoPreset configuration object
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

        # Check if GPU encoding should be used - either explicitly requested or codec is GPU-based
        is_gpu_codec = codec in ["h264_nvenc", "hevc_nvenc", "h264_amf", "hevc_amf", "h264_qsv", "hevc_qsv"]
        use_gpu = kwargs.get("gpu") or is_gpu_codec

        # Video encoding parameters
        if use_gpu:
            # Use the actual codec parameter, not hardcoded hevc_nvenc
            gpu_codec = codec if is_gpu_codec else "hevc_nvenc"

            # For GPU encoders, map CPU FFmpeg presets to GPU equivalents
            # First get the actual FFmpeg preset from config if available
            ffmpeg_preset = preset_config.preset if preset_config else preset

            # Map CPU presets to GPU presets
            gpu_preset_map = {
                "ultrafast": "p1",
                "superfast": "p1",
                "veryfast": "p2",
                "faster": "p3",
                "fast": "p4",
                "medium": "p5",
                "slow": "p6",
                "slower": "p7",
                "veryslow": "p7",
            }
            gpu_preset = gpu_preset_map.get(ffmpeg_preset, "p5")

            # NVIDIA NVENC codecs need special rate control parameters
            if "nvenc" in gpu_codec:
                # Use config params if available, otherwise use sensible defaults
                rate_control = getattr(preset_config, "rate_control", None) or "constqp"
                quality_param = getattr(preset_config, "quality_param", None) or "qp"

                cmd.extend(
                    ["-c:v", gpu_codec, "-preset", gpu_preset, "-rc", rate_control, f"-{quality_param}", str(crf)]
                )
            elif "amf" in gpu_codec:
                cmd.extend(["-c:v", gpu_codec, "-preset", gpu_preset, "-qp", str(crf)])
            elif "qsv" in gpu_codec:
                cmd.extend(["-c:v", gpu_codec, "-preset", gpu_preset, "-global_quality", str(crf)])
            else:
                # Fallback for unknown GPU codecs
                cmd.extend(["-c:v", gpu_codec, "-preset", gpu_preset, "-crf", str(crf)])
        else:
            # For CPU encoders, use the actual FFmpeg preset from config, not the preset name
            ffmpeg_preset = preset_config.preset if preset_config else preset
            cmd.extend(["-c:v", codec, "-preset", ffmpeg_preset])

            # Handle different codec quality parameters
            if codec == "libx265":
                cmd.extend(["-x265-params", f"crf={crf}"])
            elif codec in ["libaom-av1", "librav1e", "libsvtav1"]:
                # AV1 codecs use -crf parameter
                cmd.extend(["-crf", str(crf)])
            elif codec == "libvvenc":
                # VVC uses -qp parameter
                cmd.extend(["-qp", str(crf)])
            else:
                # Default codecs use -crf
                cmd.extend(["-crf", str(crf)])

        # Audio handling (usually copy)
        cmd.extend(["-c:a", kwargs.get("audio_codec", "copy")])

        cmd.append(str(output_file))
        return cmd
