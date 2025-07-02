from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.fixture
def mock_audio_file(tmp_path):
    """Create a mock audio file for testing."""
    test_file = tmp_path / "test.mp3"
    # Create a file with some actual content
    test_file.write_bytes(b"dummy audio content" * 1000)  # Make it bigger for realistic testing
    return test_file


@pytest.fixture
def mock_video_file(tmp_path):
    """Create a mock video file for testing."""
    test_file = tmp_path / "test.mp4"
    # Create a file with some actual content
    test_file.write_bytes(b"dummy video content" * 1000)  # Make it bigger for realistic testing
    return test_file


@pytest.fixture
def mock_config_manager():
    """Create a mock config manager."""
    config_manager = Mock()
    # Mock the config structure
    config_manager.config = Mock()
    config_manager.config.audio = Mock()
    config_manager.config.audio.extensions = [".mp3", ".flac", ".wav"]
    config_manager.config.audio.presets = {"music": Mock(bitrate="128k", application="audio")}
    config_manager.config.video = Mock()
    config_manager.config.video.extensions = [".mp4", ".mkv", ".avi"]
    config_manager.config.video.presets = {"default": Mock(crf=23, codec="libx265", preset="medium")}

    # Mock get_value to return the correct BackupStrategy value
    def mock_get_value(key, default=None):
        if key == "global_.cleanup_backups":
            return "on_success"
        return default

    config_manager.get_value = Mock(side_effect=mock_get_value)
    config_manager.config.get_audio_preset = Mock(return_value=Mock(bitrate="128k", application="audio"))
    return config_manager


def test_audio_estimate_basic(mock_audio_file):
    """Test basic audio estimation functionality."""
    with patch("transcode_toolkit.audio.estimate.get_config") as mock_get_config:
        # Mock configuration
        mock_config = Mock()
        mock_config.audio.extensions = [".mp3"]
        mock_config.audio.presets = {"music": Mock(bitrate="128k", application="audio", snr_bitrate_scale=False)}
        # Fix the quality_thresholds to return a proper value
        mock_config.audio.quality_thresholds = {"min_saving_percent": 5}
        mock_get_config.return_value = mock_config

        # Mock FFprobe at the correct location - it's imported from ..core
        with patch("transcode_toolkit.core.FFmpegProbe") as mock_probe:
            mock_probe.get_audio_info.return_value = {
                "duration": 180.0,
                "size": 5000000,
                "bitrate": 320000,
                "codec": "mp3",
            }

            from transcode_toolkit.audio.estimate import compare_presets, recommend_preset

            # Action
            results = compare_presets(mock_audio_file.parent)
            recommended = recommend_preset(results)

            # Validate
            assert results is not None
            assert len(results) > 0
            assert recommended is not None
            assert isinstance(recommended, str)


def test_audio_processor_basic(mock_audio_file, mock_config_manager):
    """Test basic audio processor functionality."""
    # Fix the mock config to return a set for extensions
    mock_config_manager.get_value.side_effect = lambda key, default=None: {
        "global_.cleanup_backups": "on_success",
        "audio.extensions": {".mp3", ".flac", ".wav"},
        "global_.create_backups": True,
    }.get(key, default)

    with patch("transcode_toolkit.processors.audio_processor.FFmpegProbe") as mock_probe:
        mock_probe.get_audio_info.return_value = {"duration": 180.0, "size": 5000000, "bitrate": 320000, "codec": "mp3"}

        with patch("transcode_toolkit.processors.audio_processor.FFmpegProcessor") as mock_ffmpeg:
            mock_ffmpeg_instance = MagicMock()
            mock_ffmpeg.return_value = mock_ffmpeg_instance

            from transcode_toolkit.processors.audio_processor import AudioProcessor

            processor = AudioProcessor(mock_config_manager)

            # Test can_process
            assert processor.can_process(mock_audio_file) == True

            # Test should_process
            with patch.object(processor, "_calculate_effective_bitrate", return_value="128k"):
                assert processor.should_process(mock_audio_file, preset="music") == True


def test_video_ffmpeg_command_generation(mock_video_file):
    """Test FFmpeg command generation (this functionality still exists)."""
    from transcode_toolkit.video.transcode import _ffmpeg_cmd

    output_file = mock_video_file.with_suffix(".out.mp4")

    # Test CPU command
    cmd_cpu = _ffmpeg_cmd(mock_video_file, output_file, crf=23, gpu=False)
    assert "libx265" in cmd_cpu
    assert "hevc_nvenc" not in cmd_cpu

    # Test GPU command
    cmd_gpu = _ffmpeg_cmd(mock_video_file, output_file, crf=23, gpu=True)
    assert "hevc_nvenc" in cmd_gpu
    assert "libx265" not in cmd_gpu


def test_video_ffmpeg_command():
    """Test FFmpeg command generation."""
    from transcode_toolkit.video.transcode import _ffmpeg_cmd

    input_path = Path("input.mp4")
    output_path = Path("output.mp4")

    # Test CPU command
    cmd_cpu = _ffmpeg_cmd(input_path, output_path, crf=23, gpu=False)
    assert "libx265" in cmd_cpu
    assert "hevc_nvenc" not in cmd_cpu

    # Test GPU command
    cmd_gpu = _ffmpeg_cmd(input_path, output_path, crf=23, gpu=True)
    assert "hevc_nvenc" in cmd_gpu
    assert "libx265" not in cmd_gpu


def test_unified_estimate_integration(mock_audio_file, mock_video_file):
    """Test the unified estimate functionality integration."""
    with patch("transcode_toolkit.core.unified_estimate.ConfigManager") as mock_cm:
        mock_config = Mock()
        mock_config.video.extensions = [".mp4"]
        mock_config.audio.extensions = [".mp3"]
        mock_cm.return_value.config = mock_config

        with patch("transcode_toolkit.core.unified_estimate.FFmpegProbe") as mock_probe:
            mock_probe.get_video_info.return_value = {"size": 10000000, "duration": 300, "width": 1920, "height": 1080}

            from transcode_toolkit.core.unified_estimate import analyze_directory

            # Action
            analyses, optimal_presets = analyze_directory(mock_video_file.parent)

            # Validate
            assert analyses is not None
            assert optimal_presets is not None
            assert isinstance(analyses, list)
            assert isinstance(optimal_presets, dict)
