"""Test the FFmpeg timeout fix for video estimation."""

import logging
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.transcode_toolkit.core.ffmpeg import FFmpegError
from src.transcode_toolkit.core.unified_estimate import _calculate_ssim_for_preset
from src.transcode_toolkit.core.video_analysis import quick_test_encode


def test_quick_test_encode_uses_fast_timeout() -> None:
    """Test that quick_test_encode uses the correct fast timeout of 10 seconds."""
    # Create a mock video file
    mock_file_path = Path("test_video.mp4")

    with (
        patch("src.transcode_toolkit.core.video_analysis.FFmpegProbe.get_video_info") as mock_get_video_info,
        patch("src.transcode_toolkit.core.video_analysis.FFmpegProbe.get_audio_info") as mock_get_audio_info,
        patch("src.transcode_toolkit.core.video_analysis.FFmpegProcessor") as mock_processor_class,
        patch("src.transcode_toolkit.core.video_analysis.tempfile.NamedTemporaryFile") as mock_temp,
        patch("src.transcode_toolkit.core.video_analysis.validate_quality_fast") as mock_validate,
        patch("src.transcode_toolkit.core.video_analysis.time.time") as mock_time,
    ):
        # Setup mocks for video info
        mock_get_video_info.return_value = {"duration": 60.0, "fps": 30.0, "codec": "h264"}

        # Mock audio info (this was causing the error)
        mock_get_audio_info.return_value = {"codec": "aac", "duration": 60.0}

        # Mock processor
        mock_processor = Mock()
        mock_processor.build_video_command.return_value = ["ffmpeg", "-i", "input.mp4", "-o", "output.mp4"]
        mock_processor.run_command.return_value = None
        mock_processor_class.return_value = mock_processor

        # Mock temp files
        mock_temp_file = Mock()
        mock_temp_file.name = "temp_file.mp4"
        mock_temp.__enter__.return_value = mock_temp_file

        # Mock SSIM validation
        mock_validate.return_value = 0.92

        # Mock time for encode timing
        mock_time.side_effect = [0.0, 5.0]  # 5 second encode time

        # Call the function
        result = quick_test_encode(
            file_path=mock_file_path, test_crf=24, test_duration=30, gpu=False, codec="libx265", speed_preset="fast"
        )

        # Verify that FFmpegProcessor was initialized with timeout=10
        mock_processor_class.assert_called_once_with(timeout=10)

        # Verify result structure
        expected_result_length = 3
        assert len(result) == expected_result_length
        encoded_path, ssim, speed_metrics = result
        assert "fps" in speed_metrics
        assert "processing_time_min" in speed_metrics


def test_calculate_ssim_for_preset_handles_timeout() -> None:
    """Test that _calculate_ssim_for_preset properly handles timeout exceptions."""
    # Create a mock preset config
    mock_preset_config = Mock()
    mock_preset_config.crf = 24
    mock_preset_config.codec = "libx265"
    mock_preset_config.preset = "medium"
    mock_preset_config.name = "test_preset"

    mock_file_path = Path("test_video.mp4")

    # Create a timeout exception with proper chaining
    timeout_error = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=10)
    ffmpeg_error = FFmpegError("Command timed out", command=["ffmpeg"])
    ffmpeg_error.__cause__ = timeout_error

    with patch("src.transcode_toolkit.core.video_analysis.quick_test_encode") as mock_quick_test:
        # Make quick_test_encode raise the timeout error
        mock_quick_test.side_effect = ffmpeg_error

        # Call the function
        result = _calculate_ssim_for_preset(mock_preset_config, mock_file_path)

        # Should return NaN to indicate the preset should be skipped
        import math

        assert math.isnan(result)


def test_calculate_ssim_for_preset_fallback_calculation() -> None:
    """Test that _calculate_ssim_for_preset falls back to formula-based calculation on other errors."""
    # Create a mock preset config
    mock_preset_config = Mock()
    mock_preset_config.crf = 24
    mock_preset_config.codec = "libx265"
    mock_preset_config.preset = "medium"
    mock_preset_config.name = "test_preset"

    mock_file_path = Path("test_video.mp4")

    # Create a non-timeout error
    ffmpeg_error = FFmpegError("Some other error", command=["ffmpeg"])

    with patch("src.transcode_toolkit.core.video_analysis.quick_test_encode") as mock_quick_test:
        # Make quick_test_encode raise a non-timeout error
        mock_quick_test.side_effect = ffmpeg_error

        # Call the function
        result = _calculate_ssim_for_preset(mock_preset_config, mock_file_path)

        # Should return a valid SSIM value (fallback calculation)
        import math

        min_fallback_ssim = 0.70
        max_fallback_ssim = 0.99
        assert min_fallback_ssim <= result <= max_fallback_ssim
        assert not math.isnan(result)


def test_timeout_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Test that timeout events are properly logged."""
    # Create a mock preset config
    mock_preset_config = Mock()
    mock_preset_config.crf = 24
    mock_preset_config.codec = "libx265"
    mock_preset_config.preset = "medium"
    mock_preset_config.name = "test_preset"

    mock_file_path = Path("test_video.mp4")

    # Create a timeout exception with proper chaining
    timeout_error = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=10)
    ffmpeg_error = FFmpegError("Command timed out", command=["ffmpeg"])
    ffmpeg_error.__cause__ = timeout_error

    with patch("src.transcode_toolkit.core.video_analysis.quick_test_encode") as mock_quick_test:
        mock_quick_test.side_effect = ffmpeg_error

        # Set log level to capture warnings
        with caplog.at_level(logging.WARNING):
            _calculate_ssim_for_preset(mock_preset_config, mock_file_path)

        # Check that the timeout warning was logged
        assert len(caplog.records) > 0
        timeout_logged = any("timed out during test encode" in record.message for record in caplog.records)
        assert timeout_logged, f"Expected timeout warning in logs: {[r.message for r in caplog.records]}"
