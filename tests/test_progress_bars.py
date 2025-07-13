"""Test separate progress bars functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.transcode_toolkit.core.unified_estimate import analyze_directory


def test_separate_progress_bars_creation() -> None:
    """Test that separate progress bars are created for video and audio processing."""
    # Create test directory and mock files
    test_dir = Path("test_media")

    with (
        patch("src.transcode_toolkit.core.unified_estimate.list") as mock_list,
        patch("src.transcode_toolkit.core.unified_estimate.FFmpegProbe") as mock_probe,
        patch("src.transcode_toolkit.core.unified_estimate.tqdm") as mock_tqdm,
        patch("src.transcode_toolkit.core.unified_estimate._analyze_video_file") as mock_video_analyze,
        patch("src.transcode_toolkit.core.unified_estimate._analyze_audio_file") as mock_audio_analyze,
        patch("src.transcode_toolkit.core.unified_estimate._analyze_video_audio_track") as mock_video_audio_analyze,
        patch("src.transcode_toolkit.core.unified_estimate.ConfigManager") as mock_config,
    ):
        # Setup mocks
        mock_video_file = Path("test.mp4")
        mock_audio_file = Path("test.mp3")

        # Mock directory listing
        mock_list.return_value = [mock_video_file, mock_audio_file]

        # Mock config
        mock_config_instance = MagicMock()
        mock_config_instance.config.video.extensions = [".mp4", ".mkv"]
        mock_config_instance.config.audio.extensions = [".mp3", ".flac"]
        mock_config.return_value = mock_config_instance

        # Mock FFmpegProbe for video file with audio track
        mock_probe.get_audio_info.return_value = {"duration": 100}

        # Mock analysis functions
        mock_video_analysis = MagicMock()
        mock_video_analysis.file_type = "video"
        mock_video_analysis.best_preset = "h265_balanced"
        mock_video_analysis.savings_percent = 25.0

        mock_audio_analysis = MagicMock()
        mock_audio_analysis.file_type = "audio"
        mock_audio_analysis.best_preset = "music"
        mock_audio_analysis.savings_percent = 30.0

        mock_video_analyze.return_value = mock_video_analysis
        mock_audio_analyze.return_value = mock_audio_analysis
        mock_video_audio_analyze.return_value = mock_audio_analysis

        # Mock progress bars
        mock_video_progress = MagicMock()
        mock_audio_progress = MagicMock()

        # Configure tqdm to return different instances based on desc parameter
        def tqdm_side_effect(*_args: object, **kwargs: dict[str, object]) -> MagicMock:
            if kwargs.get("desc") == "📹 Video analysis":
                return mock_video_progress
            if kwargs.get("desc") == "🔊 Audio analysis":
                return mock_audio_progress
            return MagicMock()

        mock_tqdm.side_effect = tqdm_side_effect

        # Disable logging to avoid verbose mode
        with patch("src.transcode_toolkit.core.unified_estimate.LOG") as mock_log:
            mock_log.isEnabledFor.return_value = False  # Not in verbose mode

            # Call the function
            analyses, optimal_presets = analyze_directory(test_dir)

        # Verify separate progress bars were created
        expected_progress_bars = 2  # One for video and one for audio
        assert mock_tqdm.call_count == expected_progress_bars

        # Check video progress bar was created with correct parameters
        video_calls = [call for call in mock_tqdm.call_args_list if call[1].get("desc") == "📹 Video analysis"]
        assert len(video_calls) == 1
        video_call = video_calls[0]
        assert video_call[1]["total"] == 1  # 1 video file
        assert video_call[1]["unit"] == "file"
        assert video_call[1]["position"] == 0

        # Check audio progress bar was created with correct parameters
        audio_calls = [call for call in mock_tqdm.call_args_list if call[1].get("desc") == "🔊 Audio analysis"]
        assert len(audio_calls) == 1
        audio_call = audio_calls[0]
        expected_audio_files = 2  # 1 audio file + 1 video file with audio track
        assert audio_call[1]["total"] == expected_audio_files
        assert audio_call[1]["unit"] == "file"
        assert audio_call[1]["position"] == 1

        # Verify progress bars were updated correctly
        mock_video_progress.update.assert_called_with(1)
        mock_audio_progress.update.assert_called_with(1)

        # Verify progress bars were closed
        mock_video_progress.close.assert_called_once()
        mock_audio_progress.close.assert_called_once()


def test_no_progress_bars_in_verbose_mode() -> None:
    """Test that no progress bars are created in verbose mode."""
    test_dir = Path("test_media")

    with (
        patch("src.transcode_toolkit.core.unified_estimate.list") as mock_list,
        patch("src.transcode_toolkit.core.unified_estimate.FFmpegProbe") as mock_probe,
        patch("src.transcode_toolkit.core.unified_estimate.tqdm") as mock_tqdm,
        patch("src.transcode_toolkit.core.unified_estimate._analyze_video_file") as mock_video_analyze,
        patch("src.transcode_toolkit.core.unified_estimate._analyze_audio_file") as mock_audio_analyze,
        patch("src.transcode_toolkit.core.unified_estimate.ConfigManager") as mock_config,
    ):
        # Setup mocks
        mock_video_file = Path("test.mp4")
        mock_audio_file = Path("test.mp3")

        mock_list.return_value = [mock_video_file, mock_audio_file]

        # Mock config
        mock_config_instance = MagicMock()
        mock_config_instance.config.video.extensions = [".mp4", ".mkv"]
        mock_config_instance.config.audio.extensions = [".mp3", ".flac"]
        mock_config.return_value = mock_config_instance

        # Mock FFmpegProbe
        mock_probe.get_audio_info.return_value = None  # No audio track in video

        # Mock analysis functions
        mock_video_analysis = MagicMock()
        mock_audio_analysis = MagicMock()
        mock_video_analyze.return_value = mock_video_analysis
        mock_audio_analyze.return_value = mock_audio_analysis

        # Enable verbose mode (INFO level logging)
        with patch("src.transcode_toolkit.core.unified_estimate.LOG") as mock_log:
            mock_log.isEnabledFor.return_value = True  # Verbose mode

            # Call the function
            analyses, optimal_presets = analyze_directory(test_dir)

        # Verify no progress bars were created in verbose mode
        mock_tqdm.assert_not_called()


def test_optimal_audio_settings_shown() -> None:
    """Test that optimal audio settings are included in the output."""
    test_dir = Path("test_media")

    with (
        patch("src.transcode_toolkit.core.unified_estimate.list") as mock_list,
        patch("src.transcode_toolkit.core.unified_estimate.FFmpegProbe"),
        patch("src.transcode_toolkit.core.unified_estimate._analyze_audio_file") as mock_audio_analyze,
        patch("src.transcode_toolkit.core.unified_estimate.ConfigManager") as mock_config,
    ):
        # Setup mocks
        mock_audio_file = Path("test.mp3")
        mock_list.return_value = [mock_audio_file]

        # Mock config
        mock_config_instance = MagicMock()
        mock_config_instance.config.video.extensions = [".mp4"]
        mock_config_instance.config.audio.extensions = [".mp3"]
        mock_config.return_value = mock_config_instance

        # Mock audio analysis with optimal settings
        mock_audio_analysis = MagicMock()
        mock_audio_analysis.file_type = "audio"
        mock_audio_analysis.best_preset = "music"
        mock_audio_analysis.savings_percent = 30.0
        mock_audio_analysis.current_size_mb = 10.0
        mock_audio_analysis.estimated_size_mb = 7.0
        mock_audio_analysis.savings_mb = 3.0

        mock_audio_analyze.return_value = mock_audio_analysis

        # Disable logging to avoid verbose mode
        with patch("src.transcode_toolkit.core.unified_estimate.LOG") as mock_log:
            mock_log.isEnabledFor.return_value = False

            # Call the function
            analyses, optimal_presets = analyze_directory(test_dir)

        # Verify optimal audio preset is returned
        assert optimal_presets["audio_preset"] == "music"
        assert optimal_presets["video_preset"] is None  # No video files

        # Verify analysis contains the audio file
        assert len(analyses) == 1
        assert analyses[0].file_type == "audio"
        assert analyses[0].best_preset == "music"
        expected_savings_percent = 30.0
        assert analyses[0].savings_percent == expected_savings_percent
