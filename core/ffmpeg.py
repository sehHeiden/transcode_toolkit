"""FFmpeg integration and utilities."""

from __future__ import annotations
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from .base import ProcessingError

LOG = logging.getLogger(__name__)


class FFmpegError(ProcessingError):
    """FFmpeg-specific error."""
    
    def __init__(self, message: str, command: Optional[List[str]] = None, return_code: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.command = command
        self.return_code = return_code


class FFmpegProbe:
    """FFmpeg probe utility for media file analysis."""
    
    @staticmethod
    def check_availability() -> None:
        """Check if FFmpeg tools are available."""
        required = ["ffmpeg", "ffprobe"]
        missing = []
        
        for exe in required:
            if not shutil.which(exe):
                missing.append(exe)
        
        if missing:
            error_msg = f"Missing FFmpeg executables: {', '.join(missing)}"
            LOG.error(error_msg)
            raise FFmpegError(error_msg)
    
    @staticmethod
    def probe_media(file_path: Path, stream_type: Optional[str] = None) -> Dict[str, Any]:
        """Probe media file for metadata."""
        FFmpegProbe.check_availability()
        
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
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
                timeout=30
            )
            data = json.loads(result.stdout)
            return data
        except subprocess.CalledProcessError as e:
            raise FFmpegError(
                f"ffprobe failed for {file_path}: {e.stderr or e.stdout}",
                command=cmd,
                return_code=e.returncode,
                file_path=file_path
            )
        except subprocess.TimeoutExpired:
            raise FFmpegError(f"ffprobe timed out for {file_path}", command=cmd, file_path=file_path)
        except json.JSONDecodeError as e:
            raise FFmpegError(f"Invalid JSON from ffprobe for {file_path}: {e}", command=cmd, file_path=file_path)
    
    @staticmethod
    def get_audio_info(file_path: Path) -> Dict[str, Any]:
        """Get audio stream information."""
        data = FFmpegProbe.probe_media(file_path, "a")
        
        if not data.get("streams"):
            raise FFmpegError(f"No audio streams found in {file_path}", file_path=file_path)
        
        stream = data["streams"][0]
        format_info = data.get("format", {})
        
        return {
            "codec": stream.get("codec_name"),
            "bitrate": stream.get("bit_rate"),
            "duration": float(format_info.get("duration", 0)),
            "channels": stream.get("channels"),
            "sample_rate": stream.get("sample_rate"),
            "size": int(format_info.get("size", file_path.stat().st_size)),
        }
    
    @staticmethod
    def get_video_info(file_path: Path) -> Dict[str, Any]:
        """Get video stream information."""
        data = FFmpegProbe.probe_media(file_path, "v")
        
        if not data.get("streams"):
            raise FFmpegError(f"No video streams found in {file_path}", file_path=file_path)
        
        stream = data["streams"][0]
        format_info = data.get("format", {})
        
        return {
            "codec": stream.get("codec_name"),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "bitrate": stream.get("bit_rate") or format_info.get("bit_rate"),
            "duration": float(format_info.get("duration", 0)),
            "fps": eval(stream.get("r_frame_rate", "0/1")),  # Convert fraction to float
            "size": int(format_info.get("size", file_path.stat().st_size)),
        }


class FFmpegProcessor:
    """FFmpeg command executor with enhanced error handling."""
    
    def __init__(self, timeout: int = 300):
        self.timeout = timeout
    
    def run_command(
        self, 
        command: List[str], 
        file_path: Optional[Path] = None,
        progress_callback: Optional[callable] = None
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
                check=False  # We'll handle return code ourselves
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
                    file_path=file_path
                )
            
            return result
            
        except subprocess.TimeoutExpired:
            raise FFmpegError(
                f"FFmpeg command timed out after {self.timeout}s",
                command=command,
                file_path=file_path
            )
        except Exception as e:
            if isinstance(e, FFmpegError):
                raise
            raise FFmpegError(
                f"Unexpected error running FFmpeg: {e}",
                command=command,
                file_path=file_path,
                cause=e
            )
    
    def build_audio_command(
        self,
        input_file: Path,
        output_file: Path,
        codec: str = "libopus",
        bitrate: str = "128k",
        **kwargs
    ) -> List[str]:
        """Build FFmpeg command for audio transcoding."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_file),
            "-c:a", codec,
            "-b:a", bitrate,
        ]
        
        # Add additional audio parameters
        if "application" in kwargs:
            cmd.extend(["-application", kwargs["application"]])
        if "cutoff" in kwargs:
            cmd.extend(["-cutoff", kwargs["cutoff"]])
        if "channels" in kwargs:
            cmd.extend(["-ac", kwargs["channels"]])
        if "vbr" in kwargs:
            cmd.extend(["-vbr", kwargs["vbr"]])
        if "compression_level" in kwargs:
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
        **kwargs
    ) -> List[str]:
        """Build FFmpeg command for video transcoding."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_file),
            "-map", "0:v:0",
            "-map", "0:a?",
        ]
        
        # Video encoding parameters
        if "gpu" in kwargs and kwargs["gpu"]:
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
