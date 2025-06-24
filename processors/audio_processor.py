"""Audio processor implementation using the new core architecture."""

from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from core import (
    MediaProcessor, ProcessingResult, ProcessingStatus, ProcessingError,
    FFmpegProbe, FFmpegProcessor, FFmpegError,
    ConfigManager, ProcessingOptions,
    FileManager, BackupStrategy,
    Estimator
)


class AudioProcessor(MediaProcessor):
    """Audio transcoding processor using Opus codec."""
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__("AudioProcessor")
        self.config_manager = config_manager
        self.ffmpeg = FFmpegProcessor()
        
        # Initialize file manager with backup strategy from config
        backup_strategy_str = config_manager.get_value("global_.cleanup_backups", "on_success")
        backup_strategy = BackupStrategy(backup_strategy_str)
        retention_days = config_manager.get_value("global_.backup_retention_days", 0)
        
        self.file_manager = FileManager(backup_strategy, retention_days)
    
    def can_process(self, file_path: Path) -> bool:
        """Check if file is a supported audio format."""
        audio_extensions = self.config_manager.get_value("audio.extensions", set())
        return file_path.suffix.lower() in audio_extensions
    
    def should_process(self, file_path: Path, preset: str = "music", **kwargs) -> bool:
        """Check if file should be processed (not already optimized)."""
        try:
            audio_info = FFmpegProbe.get_audio_info(file_path)
            preset_config = self.config_manager.config.get_audio_preset(preset)
            
            # Skip if already Opus with similar or lower bitrate
            if audio_info["codec"] == "opus":
                if audio_info["bitrate"]:
                    target_bps = int(preset_config.bitrate.rstrip("k")) * 1000
                    current_bps = int(audio_info["bitrate"])
                    return current_bps > target_bps * 1.1  # 10% tolerance
            
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
            
            # Build and run FFmpeg command
            command = self.ffmpeg.build_audio_command(
                input_file=file_path,
                output_file=temp_file,
                codec="libopus",
                bitrate=preset_config.bitrate,
                application=preset_config.application,
                cutoff=preset_config.cutoff,
                channels=preset_config.channels,
                vbr="on",
                compression_level=10
            )
            
            # Execute transcoding
            result = self.ffmpeg.run_command(command, file_path)
            processing_time = time.time() - start_time
            
            # Check output file
            if not temp_file.exists():
                raise ProcessingError(f"Output file not created: {temp_file}")
            
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
                    create_backup=create_backups
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
                        "backup_created": operation.backup_path is not None
                    }
                )
            else:
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
                        "size_improvement": (original_size - new_size) / original_size * 100
                    }
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Clean up any temporary files
            temp_file = file_path.with_suffix(".tmp.opus")
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
            
            if isinstance(e, (FFmpegError, ProcessingError)):
                error_msg = str(e)
            else:
                error_msg = f"Unexpected error: {e}"
            
            return ProcessingResult(
                source_file=file_path,
                status=ProcessingStatus.ERROR,
                message=error_msg,
                processing_time=processing_time,
                metadata={"preset": preset, "error": str(e)}
            )
    
    def process_directory(
        self,
        directory: Path,
        preset: str = "music",
        max_workers: Optional[int] = None,
        recursive: bool = True,
        **kwargs
    ) -> List[ProcessingResult]:
        """Process all audio files in directory with multithreading."""
        # Get worker count from config if not specified
        if max_workers is None:
            max_workers = self.config_manager.get_value("global_.default_workers")
        
        # Clean up old backups first
        self.file_manager.cleanup_old_backups(directory)
        
        # Get all compatible files
        pattern = "**/*" if recursive else "*"
        files = [
            f for f in directory.glob(pattern) 
            if f.is_file() and self.can_process(f) and self.should_process(f, preset=preset, **kwargs)
        ]
        
        self.logger.info(f"Processing {len(files)} audio files with preset '{preset}'")
        
        results = []
        
        try:
            if max_workers == 1:
                # Single-threaded processing
                for file_path in files:
                    result = self.process_file(file_path, preset=preset, **kwargs)
                    results.append(result)
            else:
                # Multi-threaded processing
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_file = {
                        executor.submit(self.process_file, file_path, preset, **kwargs): file_path
                        for file_path in files
                    }
                    
                    for future in as_completed(future_to_file):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            file_path = future_to_file[future]
                            self.logger.error(f"Error processing {file_path}: {e}")
                            results.append(ProcessingResult(
                                source_file=file_path,
                                status=ProcessingStatus.ERROR,
                                message=f"Threading error: {e}"
                            ))
        finally:
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


class AudioEstimator:
    """Audio size estimation utility."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = __import__("logging").getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def estimate_size(self, file_path: Path, preset: str = "music") -> int:
        """Estimate the size after transcoding with given preset."""
        try:
            audio_info = FFmpegProbe.get_audio_info(file_path)
            preset_config = self.config_manager.config.get_audio_preset(preset)
            
            # Convert bitrate to bits per second
            target_bitrate = int(preset_config.bitrate.rstrip("k")) * 1000
            
            # Estimate: duration * bitrate / 8
            estimated_size = int(audio_info["duration"] * target_bitrate / 8)
            
            return estimated_size
        except Exception as e:
            self.logger.warning(f"Could not estimate size for {file_path}: {e}")
            return file_path.stat().st_size  # Return original size as fallback
    
    def estimate_directory(self, directory: Path, **kwargs) -> Dict[str, any]:
        """Estimate sizes for all audio files in directory."""
        audio_extensions = self.config_manager.get_value("audio.extensions", set())
        
        # Find all audio files
        files = [
            f for f in directory.rglob("*")
            if f.is_file() and f.suffix.lower() in audio_extensions
        ]
        
        results = {}
        
        for preset_name in self.config_manager.config.list_audio_presets():
            current_total = 0
            estimated_total = 0
            file_count = 0
            
            for file_path in files:
                try:
                    current_size = file_path.stat().st_size
                    estimated_size = self.estimate_size(file_path, preset_name)
                    
                    current_total += current_size
                    estimated_total += estimated_size
                    file_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error estimating {file_path}: {e}")
            
            if file_count > 0:
                savings = current_total - estimated_total
                savings_percent = (savings / current_total * 100) if current_total > 0 else 0
                
                results[preset_name] = {
                    "files": file_count,
                    "current_size_mb": current_total / (1024 * 1024),
                    "estimated_size_mb": estimated_total / (1024 * 1024),
                    "savings_mb": savings / (1024 * 1024),
                    "savings_percent": savings_percent,
                }
        
        return results
