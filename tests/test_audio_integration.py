from __future__ import annotations

from pathlib import Path

from transcode_toolkit.audio import estimate_audio, should_process_audio, transcode_audio
from transcode_toolkit.config import ToolkitConfig
from transcode_toolkit.types import ProcessingStatus


class TestTranscodeAudio:
    def test_speech_wav_to_opus(self, speech_wav: Path, tmp_config: Path) -> None:
        config = ToolkitConfig.from_yaml(tmp_config)
        result = transcode_audio(speech_wav, config.audio.presets["audiobook"], config)
        assert result.status == ProcessingStatus.SUCCESS
        assert result.new_size is not None
        assert result.new_size < result.original_size

    def test_speech_mp3_to_opus(self, speech_mp3: Path, tmp_config: Path) -> None:
        config = ToolkitConfig.from_yaml(tmp_config)
        result = transcode_audio(speech_mp3, config.audio.presets["audiobook"], config)
        assert result.status == ProcessingStatus.SUCCESS
        assert result.new_size is not None
        assert result.new_size < result.original_size

    def test_backup_created(self, speech_wav: Path, tmp_config: Path) -> None:
        config = ToolkitConfig.from_yaml(tmp_config)
        result = transcode_audio(speech_wav, config.audio.presets["music"], config)
        assert result.status == ProcessingStatus.SUCCESS
        backup = speech_wav.with_suffix(".wav.bak")
        if config.global_.create_backups:
            assert backup.exists()

    def test_opus_skipped(self, already_opus: Path) -> None:
        assert not should_process_audio(already_opus)


class TestEstimateAudio:
    def test_estimate_speech_wav(self, speech_wav: Path, tmp_config: Path) -> None:
        config = ToolkitConfig.from_yaml(tmp_config)
        result = estimate_audio(speech_wav, config.audio.presets["audiobook"])
        assert result.status == ProcessingStatus.SUCCESS
        assert result.new_size is not None
        assert result.new_size < result.original_size

    def test_estimate_speech_mp3(self, speech_mp3: Path, tmp_config: Path) -> None:
        config = ToolkitConfig.from_yaml(tmp_config)
        result = estimate_audio(speech_mp3, config.audio.presets["music"])
        assert result.status == ProcessingStatus.SUCCESS
        assert result.new_size is not None

    def test_audiobook_preset_smaller_than_music(self, speech_wav: Path, tmp_config: Path) -> None:
        config = ToolkitConfig.from_yaml(tmp_config)
        ab = estimate_audio(speech_wav, config.audio.presets["audiobook"])
        music = estimate_audio(speech_wav, config.audio.presets["music"])
        assert (ab.new_size or 0) <= (music.new_size or 0)


class TestProbeAudio:
    def test_probe_returns_format(self, speech_mp3: Path) -> None:
        from transcode_toolkit.ffmpeg import probe_media

        info = probe_media(speech_mp3)
        assert "format" in info
        assert float(info["format"].get("duration", 0)) > 0

    def test_measure_snr_returns_float(self, speech_wav: Path) -> None:
        from transcode_toolkit.ffmpeg import measure_snr

        snr = measure_snr(speech_wav)
        assert isinstance(snr, float)
