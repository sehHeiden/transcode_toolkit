from pathlib import Path

import pytest
from pydantic import ValidationError

from transcode_toolkit.config import AudioPreset, ToolkitConfig


class TestToolkitConfig:
    def test_from_yaml(self, tmp_config: Path):
        config = ToolkitConfig.from_yaml(tmp_config)
        assert "music" in config.audio.presets
        assert config.audio.size_keep_ratio == 0.95
        assert config.video.min_savings_percent == 10.0

    def test_frozen(self, tmp_config: Path):
        config = ToolkitConfig.from_yaml(tmp_config)
        with pytest.raises(ValidationError):
            config.audio.size_keep_ratio = 0.5

    def test_preset_access(self, tmp_config: Path):
        config = ToolkitConfig.from_yaml(tmp_config)
        music = config.audio.presets["music"]
        assert music.bitrate == "128k"
        assert music.min_snr_db == 70.0


class TestAudioPreset:
    def test_defaults(self):
        preset = AudioPreset()
        assert preset.bitrate == "128k"
        assert preset.snr_bitrate_scale is True

    def test_snr_db_range(self):
        with pytest.raises(ValidationError):
            AudioPreset(min_snr_db=150)
