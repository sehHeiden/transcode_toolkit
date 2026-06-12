from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AudioPreset(BaseModel):
    model_config = ConfigDict(frozen=True)

    bitrate: str = "128k"
    application: str = "audio"
    cutoff: int | None = None
    channels: int | None = None
    min_snr_db: float = 70.0
    snr_bitrate_scale: bool = True

    @field_validator("min_snr_db")
    @classmethod
    def validate_snr_range(cls, v: float) -> float:
        if v < 0 or v > 100:
            msg = "min_snr_db must be between 0 and 100"
            raise ValueError(msg)
        return v


class AudioConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    size_keep_ratio: float = 0.95
    extensions: list[str] = [".flac", ".mp3", ".wav", ".aac", ".m4a", ".ogg", ".opus", ".wma"]
    presets: dict[str, AudioPreset]


class VideoConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_savings_percent: float = 10.0
    extensions: list[str] = [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm"]


class GlobalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    workers: int | None = None
    create_backups: bool = True


class ToolkitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    audio: AudioConfig
    video: VideoConfig
    global_: GlobalConfig = Field(alias="global")

    @classmethod
    def from_yaml(cls, path: Path = Path("config.yaml")) -> "ToolkitConfig":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)
