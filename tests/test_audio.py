from __future__ import annotations

from pathlib import Path

from transcode_toolkit.audio import _parse_bitrate, should_process_audio, transcode_audio
from transcode_toolkit.config import ToolkitConfig
from transcode_toolkit.ffmpeg import validate_duration
from transcode_toolkit.types import ProcessingStatus


class TestParseBitrate:
    def test_parses_correctly(self):
        assert _parse_bitrate("128k") == 128_000
        assert _parse_bitrate("64k") == 64_000
        assert _parse_bitrate("192k") == 192_000
        assert _parse_bitrate("1m") == 1_000_000
        assert _parse_bitrate("96000") == 96_000


class TestShouldProcessAudio:
    def test_opus_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "test.opus"
        f.touch()
        assert not should_process_audio(f)

    def test_mp3_processed(self, tmp_path: Path) -> None:
        f = tmp_path / "test.mp3"
        f.touch()
        assert should_process_audio(f)

    def test_flac_processed(self, tmp_path: Path) -> None:
        f = tmp_path / "test.flac"
        f.touch()
        assert should_process_audio(f)


class TestValidateDuration:
    def test_within_tolerance(self, speech_wav: Path) -> None:
        from transcode_toolkit.ffmpeg import MediaInfo

        duration = MediaInfo.from_path(speech_wav).duration
        assert validate_duration(speech_wav, duration, tolerance=0.5)

    def test_outside_tolerance(self, tmp_path: Path) -> None:
        f = tmp_path / "dummy.wav"
        f.write_bytes(b"\x00" * 100)
        assert not validate_duration(f, 999.0, tolerance=0.5)

    def test_zero_expected_passes(self, tmp_path: Path) -> None:
        f = tmp_path / "dummy.wav"
        f.write_bytes(b"\x00" * 100)
        assert validate_duration(f, 0.0, tolerance=0.5)


class TestTranscodeAudioDurationPreserved:
    def test_output_duration_matches_input(self, speech_wav: Path, tmp_config: Path) -> None:
        from transcode_toolkit.ffmpeg import MediaInfo

        config = ToolkitConfig.from_yaml(tmp_config)
        source_duration = MediaInfo.from_path(speech_wav).duration

        result = transcode_audio(speech_wav, config.audio.presets["audiobook"], config)
        assert result.status == ProcessingStatus.SUCCESS

        output_path = speech_wav.with_suffix(".opus")
        assert output_path.exists()
        output_duration = MediaInfo.from_path(output_path).duration
        assert abs(output_duration - source_duration) < 0.5


class TestTranscodeAudioExtension:
    def test_output_has_opus_extension(self, speech_wav: Path, tmp_config: Path) -> None:
        config = ToolkitConfig.from_yaml(tmp_config)
        result = transcode_audio(speech_wav, config.audio.presets["audiobook"], config)
        assert result.status == ProcessingStatus.SUCCESS
        output_path = speech_wav.with_suffix(".opus")
        assert output_path.exists()
        assert not speech_wav.exists()


class TestChainErrorHandling:
    def test_failed_files_appear_as_errors(self, tmp_path: Path, tmp_config: Path) -> None:
        from transcode_toolkit.audio import transcode_audio_directory

        corrupted = tmp_path / "bad.mp3"
        corrupted.write_bytes(b"\x00\x00\x00\x00")
        config = ToolkitConfig.from_yaml(tmp_config)
        results = transcode_audio_directory(tmp_path, "music", config)
        error_results = [r for r in results if r.status == ProcessingStatus.ERROR]
        assert len(error_results) >= 1
