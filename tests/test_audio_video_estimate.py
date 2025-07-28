"""Test cases for audio and video estimation functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.transcode_toolkit.cli.main import MediaToolkitCLI


@pytest.fixture
def empty_dir(tmp_path: Path) -> Path:
    """Create an empty directory for testing."""
    return tmp_path / "empty"


@pytest.fixture
def populated_dir(tmp_path: Path) -> Path:
    """Create a directory with dummy audio and video files."""
    d = tmp_path / "populated"
    d.mkdir()
    # Create dummy files
    for i in range(5):
        (d / f"video{i}.mp4").write_bytes(b"dummy video content" * 1000)
        (d / f"audio{i}.mp3").write_bytes(b"dummy audio content" * 1000)
    return d


@patch("src.transcode_toolkit.cli.main.MediaToolkitCLI.run")
def test_audio_estimate_empty_dir(mock_run: MagicMock, empty_dir: Path) -> None:
    """Test audio estimate on an empty directory should not show a table."""
    with patch("builtins.print") as mock_print:
        mock_run.return_value = 0
        cli = MediaToolkitCLI()
        cli.run(["audio", "estimate", str(empty_dir)])

        # Verify print was called but table was not output
        mock_print.assert_not_called()


@patch("src.transcode_toolkit.audio.estimate.print_comparison")
@patch("src.transcode_toolkit.audio.estimate.recommend_preset")
@patch("src.transcode_toolkit.audio.estimate.compare_presets")
def test_audio_estimate_populated_dir(
    mock_compare: MagicMock, mock_recommend: MagicMock, mock_print_comparison: MagicMock, populated_dir: Path
) -> None:
    """Test audio estimate on a populated directory should show analysis results."""
    from src.transcode_toolkit.audio.estimate import EstimationResult
    from src.transcode_toolkit.cli.main import MediaToolkitCLI

    # Mock the estimation results for audio files
    mock_results = [
        EstimationResult(
            preset="audiobook",
            current_size=10 * 1024 * 1024,  # 10MB
            estimated_size=7 * 1024 * 1024,  # 7MB
            saving=3 * 1024 * 1024,  # 3MB
            saving_percent=30.0,
        ),
        EstimationResult(
            preset="music",
            current_size=10 * 1024 * 1024,  # 10MB
            estimated_size=8 * 1024 * 1024,  # 8MB
            saving=2 * 1024 * 1024,  # 2MB
            saving_percent=20.0,
        ),
    ]

    mock_compare.return_value = mock_results
    mock_recommend.return_value = "audiobook"

    # Capture print output
    with patch("builtins.print") as mock_print:
        cli = MediaToolkitCLI()
        exit_code = cli.run(["audio", "estimate", str(populated_dir)])

        # Verify the command succeeded
        assert exit_code == 0

        # Verify the audio functions were called
        mock_compare.assert_called_once_with(populated_dir)
        mock_recommend.assert_called_once_with(mock_results)
        mock_print_comparison.assert_called_once_with(mock_results, "audiobook")

        # Verify expected output messages are printed
        expected_calls = [
            "🎵 Analyzing audio files for optimization opportunities...",
            "💡 Use -v for detailed progress information",
        ]

        for expected_call in expected_calls:
            mock_print.assert_any_call(expected_call)


@patch("src.transcode_toolkit.cli.main.MediaToolkitCLI.run")
def test_video_estimate_empty_dir(mock_run: MagicMock, empty_dir: Path) -> None:
    """Test video estimate on an empty directory should not show a table."""
    with patch("builtins.print") as mock_print:
        mock_run.return_value = 0
        cli = MediaToolkitCLI()
        cli.run(["video", "estimate", str(empty_dir)])

        # Verify print was called but table was not output
        mock_print.assert_not_called()


@patch("src.transcode_toolkit.core.unified_estimate.print_video_comparison")
@patch("src.transcode_toolkit.core.unified_estimate.analyze_directory")
def test_video_estimate_populated_dir(mock_analyze: MagicMock, populated_dir: Path) -> None:
    """Test video estimate on a populated directory should show a table with specific columns."""
    from src.transcode_toolkit.cli.main import MediaToolkitCLI
    from src.transcode_toolkit.core.unified_estimate import FileAnalysis

    # Mock the analysis results
    mock_analyses = [
        FileAnalysis(
            file_path=populated_dir / "video0.mp4",
            file_type="video",
            current_size_mb=100.0,
            best_preset="h265_balanced",
            estimated_size_mb=75.0,
            savings_mb=25.0,
            savings_percent=25.0,
            predicted_ssim=0.95,
            estimated_speed_fps=30.0,
            processing_time_min=5.0,
            alternatives=[],
        )
    ]
    mock_optimal_presets = {"video_preset": "h265_balanced", "audio_preset": None}
    mock_analyze.return_value = (mock_analyses, mock_optimal_presets)

    # Capture print output
    with patch("builtins.print") as mock_print:
        cli = MediaToolkitCLI()
        exit_code = cli.run(["video", "estimate", str(populated_dir)])

        # Verify the command succeeded
        assert exit_code == 0

        # Verify analysis was called
        mock_analyze.assert_called_once()

        # Verify expected output messages are printed
        expected_calls = [
            "📊 Analyzing media files for optimization opportunities...",
            "💡 Use -v for detailed progress information",
        ]

        for expected_call in expected_calls:
            mock_print.assert_any_call(expected_call)


@patch("src.transcode_toolkit.core.unified_estimate._calculate_processing_time")
def test_video_comparison_table_format(mock_calc_time: MagicMock) -> None:
    """Test that video comparison table has the expected column headers."""
    from src.transcode_toolkit.core.unified_estimate import (
        EstimationResult,
        print_video_comparison,
    )

    # Mock the processing time calculation
    mock_calc_time.return_value = 5.0

    # Create mock estimation results using a valid preset name
    mock_results = [
        EstimationResult(
            preset="default",  # Use a valid preset name
            current_size=100 * 1024 * 1024,  # 100MB in bytes
            estimated_size=75 * 1024 * 1024,  # 75MB in bytes
            saving=25 * 1024 * 1024,  # 25MB in bytes
            saving_percent=25.0,
            predicted_ssim=0.95,
        )
    ]

    with patch("builtins.print") as mock_print:
        print_video_comparison(mock_results, "default")

        # Verify table headers are printed with expected columns
        expected_columns = ["Preset", "Current", "Estimated", "Saving", "%", "SSIM", "Time", "Score"]

        # Check that a print call contains all expected column headers
        header_found = False
        for call_args in mock_print.call_args_list:
            call_text = str(call_args[0][0]) if call_args[0] else ""
            if all(col in call_text for col in expected_columns):
                header_found = True
                break

        assert header_found, f"Expected table headers {expected_columns} not found in print calls"
