"""Configuration management for media toolkit."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

LOG = logging.getLogger(__name__)

# Global config instance
_global_config: MediaToolkitConfig | None = None


@dataclass
class AudioPreset:
    """Audio encoding preset configuration."""

    bitrate: str
    application: str
    cutoff: int | None = None
    channels: str | None = None
    min_snr_db: float | None = None  # Minimum SNR to use this preset's full bitrate
    snr_bitrate_scale: bool = True  # Whether to scale bitrate based on SNR
    description: str = ""

    def model_dump(self) -> dict[str, Any]:
        """Return dictionary representation of the preset."""
        return {
            "bitrate": self.bitrate,
            "application": self.application,
            "cutoff": self.cutoff,
            "channels": self.channels,
            "min_snr_db": self.min_snr_db,
            "snr_bitrate_scale": self.snr_bitrate_scale,
            "description": self.description,
        }


@dataclass
class VideoPreset:
    """Video encoding preset configuration."""

    crf: int
    codec: str
    preset: str
    description: str = ""
    rate_control: str | None = None  # GPU rate control mode (e.g., "constqp" for NVENC)
    quality_param: str | None = None  # Quality parameter type ("crf", "qp", "global_quality")


@dataclass
class AudioConfig:
    """Audio processing configuration."""

    size_keep_ratio: float = 0.95
    extensions: list[str] = field(
        default_factory=lambda: [
            ".flac",
            ".mp3",
            ".wav",
            ".aac",
            ".m4a",
            ".ogg",
            ".wma",
        ]
    )
    presets: dict[str, AudioPreset] = field(default_factory=dict)
    quality_thresholds: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoConfig:
    """Video processing configuration."""

    min_savings_percent: float = 10.0
    size_keep_ratio: float = 0.95
    extensions: list[str] = field(
        default_factory=lambda: [
            ".mp4",
            ".mkv",
            ".avi",
            ".mov",
            ".wmv",
            ".flv",
            ".webm",
            ".m4v",
        ]
    )
    presets: dict[str, VideoPreset] = field(default_factory=dict)


@dataclass
class GlobalConfig:
    """Global settings."""

    default_workers: int | None = None
    log_level: str = "INFO"
    create_backups: bool = True
    cleanup_backups: str = "on_success"


@dataclass
class MediaToolkitConfig:
    """Main configuration class."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    global_: GlobalConfig = field(default_factory=GlobalConfig)

    @classmethod
    def load_from_file(cls, config_path: Path) -> MediaToolkitConfig:
        """Load configuration from YAML file."""
        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            return cls._from_dict(data)
        except Exception as e:
            LOG.warning("Failed to load config from %s: %s", config_path, e)
            return cls()

    def list_audio_presets(self) -> list[str]:
        """Get list of available audio preset names."""
        return list(self.audio.presets.keys())

    def get_audio_preset(self, preset_name: str) -> AudioPreset:
        """Get an audio preset by name."""
        if preset_name not in self.audio.presets:
            available = ", ".join(self.list_audio_presets())
            msg = f"Unknown audio preset '{preset_name}'. Available: {available}"
            raise ValueError(msg)
        return self.audio.presets[preset_name]

    def get_video_preset(self, preset_name: str) -> VideoPreset:
        """Get a video preset by name."""
        if preset_name not in self.video.presets:
            available = ", ".join(self.video.presets.keys())
            msg = f"Unknown video preset '{preset_name}'. Available: {available}"
            raise ValueError(msg)
        return self.video.presets[preset_name]

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> MediaToolkitConfig:
        """Create config from dictionary."""
        # Parse audio config
        audio_data = data.get("audio", {})
        audio_presets = {}
        for name, preset_data in audio_data.get("presets", {}).items():
            try:
                if isinstance(preset_data, dict) and "bitrate" in preset_data and "application" in preset_data:
                    audio_presets[name] = AudioPreset(
                        bitrate=preset_data["bitrate"],
                        application=preset_data["application"],
                        cutoff=preset_data.get("cutoff"),
                        channels=preset_data.get("channels"),
                        min_snr_db=preset_data.get("min_snr_db"),
                        snr_bitrate_scale=preset_data.get("snr_bitrate_scale", True),
                        description=preset_data.get("description", ""),
                    )
                else:
                    LOG.warning("Incomplete audio preset data for '%s': missing required fields", name)
            except Exception as e:
                LOG.warning("Failed to load audio preset '%s': %s", name, e)

        audio_config = AudioConfig(
            size_keep_ratio=audio_data.get("size_keep_ratio", 0.95),
            extensions=audio_data.get("extensions", [".flac", ".mp3", ".wav", ".aac", ".m4a", ".ogg", ".wma"]),
            presets=audio_presets,
            quality_thresholds=audio_data.get("quality_thresholds", {}),
        )

        # Parse video config
        video_data = data.get("video", {})
        video_presets = {}
        for name, preset_data in video_data.get("presets", {}).items():
            try:
                if (
                    isinstance(preset_data, dict)
                    and "crf" in preset_data
                    and "codec" in preset_data
                    and "preset" in preset_data
                ):
                    video_presets[name] = VideoPreset(
                        crf=preset_data["crf"],
                        codec=preset_data["codec"],
                        preset=preset_data["preset"],
                        description=preset_data.get("description", ""),
                        rate_control=preset_data.get("rate_control"),
                        quality_param=preset_data.get("quality_param"),
                    )
                else:
                    LOG.warning(f"Incomplete video preset data for '{name}': missing required fields")
            except Exception as e:
                LOG.warning(f"Failed to load video preset '{name}': {e}")

        # Generate ALL possible CRF/speed combinations automatically
        codecs = video_data.get("codecs", [])
        crf_base = video_data.get("crf_base", {})
        crf_offsets = video_data.get("crf_offsets", [-6, 0, 2, 6])  # Available CRF offsets
        speed_options = video_data.get("speed_options", ["medium", "fast", "slow"])  # Available speeds

        # Codec name mapping for preset names
        codec_names = {
            "libx265": "h265",
            "libaom-av1": "av1_best",
            "libsvtav1": "av1",
            "librav1e": "av1_fast",
            "libvvenc": "vvc",
            "hevc_nvenc": "gpu",
            "hevc_amf": "amd",
            "hevc_qsv": "intel",
        }

        # Generate ALL possible combinations: codec × CRF offset × speed
        for codec in codecs:
            short_name = codec_names.get(codec, codec.replace("lib", ""))
            base_crf = crf_base.get(codec, 24)

            for crf_offset in crf_offsets:
                for speed_preset in speed_options:
                    # Create preset name: codec_crf{offset}_speed{speed}
                    # Examples: h265_crf-12_speedveryslow, av1_crf0_speedmedium
                    preset_name = f"{short_name}_crf{crf_offset}_speed{speed_preset}"

                    # Calculate final CRF
                    final_crf = base_crf + crf_offset

                    # Add VideoPreset with completely independent CRF and speed
                    video_presets[preset_name] = VideoPreset(
                        crf=final_crf,
                        codec=codec,
                        preset=speed_preset,
                        description=f"{codec} CRF {final_crf} (offset {crf_offset:+d}), speed {speed_preset}",
                    )

        # Add default preset that aliases a balanced H.265 preset
        # Look for a preset with CRF offset 0 and medium speed
        default_preset_name = "h265_crf0_speedmedium"
        if default_preset_name in video_presets:
            video_presets["default"] = video_presets[default_preset_name]

        video_config = VideoConfig(
            min_savings_percent=video_data.get("min_savings_percent", 10.0),
            size_keep_ratio=video_data.get("size_keep_ratio", 0.95),
            extensions=video_data.get(
                "extensions",
                [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"],
            ),
            presets=video_presets,
        )

        # Parse global config
        global_data = data.get("global", {})

        # Validate backup strategy
        cleanup_backups = global_data.get("cleanup_backups", "on_success")
        valid_strategies = {"never", "on_success"}
        if cleanup_backups not in valid_strategies:
            LOG.warning(
                f"Invalid backup strategy '{cleanup_backups}'. Using 'on_success'. "
                f"Valid options: {', '.join(valid_strategies)}"
            )
            cleanup_backups = "on_success"

        global_config = GlobalConfig(
            default_workers=global_data.get("default_workers"),
            log_level=global_data.get("log_level", "INFO"),
            create_backups=global_data.get("create_backups", True),
            cleanup_backups=cleanup_backups,
        )

        return cls(audio=audio_config, video=video_config, global_=global_config)


def get_config() -> MediaToolkitConfig:
    """Get the global configuration instance."""
    global _global_config

    if _global_config is None:
        # Try to load from default config file (look in project root)
        config_path = Path.cwd() / "config.yaml"
        if config_path.exists():
            _global_config = MediaToolkitConfig.load_from_file(config_path)
        else:
            _global_config = MediaToolkitConfig()

    # Cast to the expected type for static analysis
    return _global_config  # type: ignore
