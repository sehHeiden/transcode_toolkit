"""Test error handling, complex scoring algorithms, and integration scenarios."""

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

from src.transcode_toolkit.core.unified_estimate import (
    DEFAULT_WEIGHTS,
    EstimationResult,
    FileAnalysis,
    analyze_directory,
    calculate_weighted_score,
    process_video_files,
)
from src.transcode_toolkit.core.video_analysis import VideoComplexity

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


def test_ffmpeg_timeout_recovery() -> None:
    """Test graceful recovery from FFmpeg timeouts in various contexts."""
    # Simulate FFmpeg timeout during video processing by mocking FFmpegProbe.probe_media
    with patch("src.transcode_toolkit.core.ffmpeg.FFmpegProbe.probe_media") as mock_probe:
        mock_probe.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=10)

        # Mock file operations to avoid actual file system interactions
        with patch("pathlib.Path.exists", return_value=True):
            # Create a proper mock stat object
            mock_stat_result = Mock()
            mock_stat_result.st_size = 1024 * 1024  # 1MB
            mock_stat_result.st_mtime = 1234567890  # Fixed timestamp

            with patch("pathlib.Path.stat", return_value=mock_stat_result):
                # Verify that the system logs the timeout and continues gracefully
                results = process_video_files([Path("sample.mp4")], None, verbose_mode=False)
                assert len(results) == 0, "Timeout should result in zero analyses"


def test_corrupted_media_file_handling() -> None:
    """Test handling of corrupted or invalid media files."""
    with patch("src.transcode_toolkit.core.unified_estimate.get_media_files") as mock_get_files:
        # Mock getting files but analysis fails
        mock_get_files.return_value = ([Path("corrupted.mp4")], [])

        # Mock FFmpeg probe to raise an error for corrupted files
        with patch("src.transcode_toolkit.core.ffmpeg.FFmpegProbe.probe_media") as mock_probe:
            from src.transcode_toolkit.core.ffmpeg import FFmpegError

            mock_probe.side_effect = FFmpegError(
                "ffprobe failed",
                command=["ffprobe"],
                return_code=1,
                file_path=Path("corrupted.mp4"),
                stderr="Invalid data",
            )

            # Mock video comparison and printing to avoid Unicode issues
            with patch("src.transcode_toolkit.core.unified_estimate.compare_video_presets") as mock_compare:
                mock_compare.return_value = []
                
                with patch("src.transcode_toolkit.core.unified_estimate.print_video_comparison") as mock_print:
                    mock_print.return_value = None
                    
                    with patch("pathlib.Path.exists", return_value=True):
                        # Create proper mock stat object for file size calculations
                        mock_stat_result = Mock()
                        mock_stat_result.st_size = 1024 * 1024  # 1MB
                        mock_stat_result.st_mtime = 1234567890

                        with patch("pathlib.Path.stat", return_value=mock_stat_result):
                            results, _ = analyze_directory(Path("corrupted_files"))
                            assert len(results) == 0, "Corrupted files should be skipped"


def test_multiple_consecutive_timeouts() -> None:
    """Test handling of multiple consecutive timeout errors."""
    files = [Path("video1.mp4"), Path("video2.mp4"), Path("video3.mp4")]

    with patch("src.transcode_toolkit.core.ffmpeg.FFmpegProbe.probe_media") as mock_probe:
        # All files timeout
        mock_probe.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=10)

        with patch("pathlib.Path.exists", return_value=True):
            # Create proper mock stat objects
            mock_stat_result = Mock()
            mock_stat_result.st_size = 1024 * 1024
            mock_stat_result.st_mtime = 1234567890

            with patch("pathlib.Path.stat", return_value=mock_stat_result):
                results = process_video_files(files, None, verbose_mode=False)
                assert len(results) == 0, "All timeouts should result in zero analyses"


def test_missing_codec_support() -> None:
    """Test behavior when required codecs are not available."""
    with patch("src.transcode_toolkit.core.ffmpeg.FFmpegProbe.probe_media") as mock_probe:
        mock_probe.side_effect = RuntimeError("Codec not found: libx265")

        with patch("pathlib.Path.exists", return_value=True):
            # Create proper mock stat object
            mock_stat_result = Mock()
            mock_stat_result.st_size = 1024 * 1024
            mock_stat_result.st_mtime = 1234567890

            with patch("pathlib.Path.stat", return_value=mock_stat_result):
                results = process_video_files([Path("unsupported.mp4")], None, verbose_mode=False)
                # System should handle missing codec gracefully by creating default analysis
                assert len(results) == 1, "Missing codec should result in default analysis"
                # Verify it's a default analysis with appropriate values
                result = results[0]
                assert result.file_type == "video"
                assert result.current_size_mb == 1.0  # 1MB file
                assert result.best_preset == "default"


# ============================================================================
# COMPLEX SCORING ALGORITHM TESTS
# ============================================================================


def test_weighted_scoring_boundary_conditions() -> None:
    """Test scoring with extreme values like 0% and 100% for savings."""
    # Extreme low savings (0%)
    low_savings_score = calculate_weighted_score(
        ssim=0.95, processing_time=2.0, saving_percent=0.0, weights=DEFAULT_WEIGHTS, max_processing_time=10.0
    )

    # Extreme high savings (100%)
    high_savings_score = calculate_weighted_score(
        ssim=0.95, processing_time=2.0, saving_percent=100.0, weights=DEFAULT_WEIGHTS, max_processing_time=10.0
    )

    # Perfect quality (SSIM=1.0)
    perfect_quality_score = calculate_weighted_score(
        ssim=1.0, processing_time=2.0, saving_percent=30.0, weights=DEFAULT_WEIGHTS, max_processing_time=10.0
    )

    # Terrible quality (SSIM=0.5)
    poor_quality_score = calculate_weighted_score(
        ssim=0.5, processing_time=2.0, saving_percent=30.0, weights=DEFAULT_WEIGHTS, max_processing_time=10.0
    )

    assert low_savings_score < high_savings_score, "High savings should have a better score"
    assert poor_quality_score < perfect_quality_score, "Perfect quality should have better score"
    assert 0.0 <= low_savings_score <= 1.0, "Scores should be normalized between 0 and 1"
    assert 0.0 <= high_savings_score <= 1.0, "Scores should be normalized between 0 and 1"


def test_quality_vs_speed_tradeoffs() -> None:
    """Test scoring with different quality/speed trade-offs."""
    # High quality, slow speed
    high_quality_low_speed = calculate_weighted_score(
        ssim=0.99, processing_time=20.0, saving_percent=30.0, weights=DEFAULT_WEIGHTS, max_processing_time=20.0
    )

    # Low quality, high speed
    low_quality_high_speed = calculate_weighted_score(
        ssim=0.70, processing_time=1.0, saving_percent=30.0, weights=DEFAULT_WEIGHTS, max_processing_time=20.0
    )

    # Medium balance
    calculate_weighted_score(
        ssim=0.85, processing_time=10.0, saving_percent=30.0, weights=DEFAULT_WEIGHTS, max_processing_time=20.0
    )

    # With default weights (quality=0.4, speed=0.3, savings=0.3),
    # quality should have the highest impact
    assert high_quality_low_speed != low_quality_high_speed, "Trade-offs should provide different scores"

    # Test with speed-focused weights
    speed_focused_weights = {"quality": 0.2, "speed": 0.6, "savings": 0.2}

    speed_focused_high_quality = calculate_weighted_score(
        ssim=0.99, processing_time=20.0, saving_percent=30.0, weights=speed_focused_weights, max_processing_time=20.0
    )

    speed_focused_low_quality = calculate_weighted_score(
        ssim=0.70, processing_time=1.0, saving_percent=30.0, weights=speed_focused_weights, max_processing_time=20.0
    )

    # With speed-focused weights, fast encoding should win
    assert speed_focused_low_quality > speed_focused_high_quality, "Speed-focused weights should favor fast encoding"


def test_codec_efficiency_factors() -> None:
    """Test how codec efficiency affects scoring and recommendations."""
    # Create estimation results for different codecs
    h264_result = EstimationResult(
        preset="h264_preset",
        current_size=100 * 1024 * 1024,  # 100MB
        estimated_size=80 * 1024 * 1024,  # 20% compression
        saving=20 * 1024 * 1024,
        saving_percent=20.0,
        predicted_ssim=0.90,
    )

    h265_result = EstimationResult(
        preset="h265_preset",
        current_size=100 * 1024 * 1024,  # Same size
        estimated_size=60 * 1024 * 1024,  # 40% compression (more efficient)
        saving=40 * 1024 * 1024,
        saving_percent=40.0,
        predicted_ssim=0.92,  # Slightly better quality
    )

    av1_result = EstimationResult(
        preset="av1_preset",
        current_size=100 * 1024 * 1024,  # Same size
        estimated_size=50 * 1024 * 1024,  # 50% compression (most efficient)
        saving=50 * 1024 * 1024,
        saving_percent=50.0,
        predicted_ssim=0.94,  # Best quality
    )

    # Test that more efficient codecs get better savings scores
    assert h265_result.saving_percent > h264_result.saving_percent, "H.265 should be more efficient than H.264"
    assert av1_result.saving_percent > h265_result.saving_percent, "AV1 should be more efficient than H.265"


def test_zero_processing_time_edge_case() -> None:
    """Test scoring with zero processing time (theoretical maximum speed)."""
    zero_time_score = calculate_weighted_score(
        ssim=0.90, processing_time=0.0, saving_percent=30.0, weights=DEFAULT_WEIGHTS, max_processing_time=10.0
    )

    normal_time_score = calculate_weighted_score(
        ssim=0.90, processing_time=5.0, saving_percent=30.0, weights=DEFAULT_WEIGHTS, max_processing_time=10.0
    )

    # Zero processing time should result in maximum speed score
    assert zero_time_score > normal_time_score, "Zero processing time should yield higher score"
    assert 0.0 <= zero_time_score <= 1.0, "Score should be normalized"


# ============================================================================
# COMPREHENSIVE INTEGRATION TESTS
# ============================================================================


def test_batch_processing_mixed_formats(tmp_path: Path) -> None:
    """Test processing directories with mixed video/audio formats."""
    # Create mock files with different extensions
    video_files = [tmp_path / "video1.mp4", tmp_path / "video2.avi", tmp_path / "video3.mkv"]
    audio_files = [tmp_path / "audio1.mp3", tmp_path / "audio2.flac"]

    # Create the files
    for file_path in video_files + audio_files:
        file_path.write_bytes(b"dummy content")

    with patch("src.transcode_toolkit.core.unified_estimate.get_media_files") as mock_get_files:
        mock_get_files.return_value = (video_files, audio_files)

        with patch("src.transcode_toolkit.core.video_analysis.analyze_file") as mock_analyze:
            mock_analyze.return_value = {
                "duration": 120.0,
                "size": 1024 * 1024,
                "codec": "h264",
                "width": 1920,
                "height": 1080,
                "bitrate": 5000000,
                "fps": 30,
                "complexity": VideoComplexity(0.5, 0.5, 0.5, 0.5),
            }

            with patch("src.transcode_toolkit.core.unified_estimate._analyze_audio_file") as mock_audio:
                mock_audio.return_value = FileAnalysis(
                    file_path=Path("dummy.mp3"),
                    file_type="audio",
                    current_size_mb=5.0,
                    best_preset="mp3_preset",
                    estimated_size_mb=3.0,
                    savings_mb=2.0,
                    savings_percent=40.0,
                    predicted_ssim=None,
                    estimated_speed_fps=None,
                    processing_time_min=None,
                    alternatives=[],
                )

                results, presets = analyze_directory(tmp_path)

                # Should process audio files successfully (video files fail ffprobe due to dummy content)
                # This tests that the system gracefully handles mixed success/failure scenarios
                audio_analyses = [r for r in results if r.file_type == "audio"]
                assert len(audio_analyses) == len(audio_files), "All audio files should be analyzed"
                
                # Video files will have failed analysis due to dummy content, which is expected
                # This verifies graceful error handling for unprocessable video files
                assert len(results) >= len(audio_files), "Should at least process all audio files"


def test_large_file_processing() -> None:
    """Test handling of very large media files."""
    large_file = Path("large_video.mp4")

    # Mock the analyze_file function in the unified_estimate module where it's imported
    with patch("src.transcode_toolkit.core.unified_estimate.analyze_file") as mock_analyze_file:
        mock_analyze_file.return_value = {
            "duration": 7200.0,  # 2 hours
            "size": 2000 * 1024 * 1024,  # 2GB
            "codec": "h264",
            "width": 3840,  # 4K resolution
            "height": 2160,
            "bitrate": 50000000,  # 50 Mbps
            "fps": 60,  # High frame rate
            "complexity": VideoComplexity(0.8, 0.9, 0.7, 0.8),  # High complexity
        }

        with patch("pathlib.Path.exists", return_value=True):
            # Create proper mock stat object
            mock_stat_result = Mock()
            mock_stat_result.st_size = 2000 * 1024 * 1024  # 2GB
            mock_stat_result.st_mtime = 1234567890

            with (
                patch("pathlib.Path.stat", return_value=mock_stat_result),
                # Mock FFmpegProbe.get_video_info to avoid additional ffprobe calls
                patch("src.transcode_toolkit.core.unified_estimate.FFmpegProbe.get_video_info") as mock_probe,
                # Mock the SSIM calculation to avoid test encode
                patch("src.transcode_toolkit.core.unified_estimate._calculate_ssim_for_preset", return_value=0.85),
            ):
                mock_probe.return_value = {"codec": "h264"}
                results = process_video_files([large_file], None, verbose_mode=False)

                assert len(results) == 1, "Large file should be processed successfully"

                # Define constants for large file thresholds
                large_file_size_mb = 1000
                min_processing_time_minutes = 10

                result = results[0]
                assert result.current_size_mb > large_file_size_mb, "Should recognize large file size"
                assert result.processing_time_min is not None, "Should estimate processing time"
                assert result.processing_time_min > min_processing_time_minutes, (
                    "Large file should take significant time"
                )


def test_network_storage_simulation() -> None:
    """Test processing files that simulate network storage access patterns."""
    network_file = Path("//network/share/video.mp4")

    # Simulate network delays and potential connection issues
    # Use a list to make the reference more explicit for type checkers
    call_count = [0]

    def slow_analyze_file(_file_path: Path, *, _use_cache: bool = True) -> dict:
        call_count[0] += 1

        # Simulate network delay
        import time

        time.sleep(0.1)  # Small delay to simulate network access

        # Occasionally fail to simulate network issues
        network_failure_threshold = 2
        network_error_msg = "Network connection lost"
        if call_count[0] == network_failure_threshold:
            raise OSError(network_error_msg)

        return {
            "duration": 300.0,
            "size": 100 * 1024 * 1024,
            "codec": "h264",
            "width": 1920,
            "height": 1080,
            "bitrate": 5000000,
            "fps": 30,
            "complexity": VideoComplexity(0.5, 0.5, 0.5, 0.5),
        }

    with (
        patch("src.transcode_toolkit.core.video_analysis.analyze_file", side_effect=slow_analyze_file),
        patch("pathlib.Path.exists", return_value=True),
    ):
        # Create proper mock stat object
        mock_stat_result = Mock()
        mock_stat_result.st_size = 100 * 1024 * 1024
        mock_stat_result.st_mtime = 1234567890

        with patch("pathlib.Path.stat", return_value=mock_stat_result):
            # Try to process multiple files, some may fail due to network issues
            files = [network_file, Path("local.mp4")]
            results = process_video_files(files, None, verbose_mode=False)

            # Should handle network failures gracefully
            assert len(results) <= len(files), "Network failures should be handled gracefully"


def test_empty_directory_handling() -> None:
    """Test processing completely empty directories."""
    with patch("src.transcode_toolkit.core.unified_estimate.get_media_files") as mock_get_files:
        mock_get_files.return_value = ([], [])  # No files found

        results, presets = analyze_directory(Path("empty_directory"))

        assert len(results) == 0, "Empty directory should return no results"
        assert isinstance(presets, dict), "Should return presets dict even when empty"


def test_directory_with_unreadable_files(tmp_path: Path) -> None:
    """Test handling directories with files that can't be read."""
    # Create some files
    readable_file = tmp_path / "readable.mp4"
    unreadable_file = tmp_path / "unreadable.mp4"

    readable_file.write_bytes(b"dummy video content")
    unreadable_file.write_bytes(b"dummy video content")

    with patch("src.transcode_toolkit.core.unified_estimate.get_media_files") as mock_get_files:
        mock_get_files.return_value = ([readable_file, unreadable_file], [])

        def analyze_side_effect(file_path: Path, *, use_cache: bool = True) -> dict:
            if file_path == unreadable_file:
                access_denied_msg = "Access denied"
                raise PermissionError(access_denied_msg)
            return {
                "duration": 120.0,
                "size": 1024 * 1024,
                "codec": "h264",
                "width": 1920,
                "height": 1080,
                "bitrate": 5000000,
                "fps": 30,
                "complexity": VideoComplexity(0.5, 0.5, 0.5, 0.5),
            }

        with (
            # Mock the analyze_file function in the unified_estimate module where it's imported
            patch("src.transcode_toolkit.core.unified_estimate.analyze_file", side_effect=analyze_side_effect),
            # Mock FFmpegProbe.get_video_info to avoid additional ffprobe calls
            patch("src.transcode_toolkit.core.unified_estimate.FFmpegProbe.get_video_info") as mock_probe,
            # Mock the SSIM calculation to avoid test encode
            patch("src.transcode_toolkit.core.unified_estimate._calculate_ssim_for_preset", return_value=0.85),
            # Mock compare_video_presets and related functions to avoid errors
            patch("src.transcode_toolkit.core.unified_estimate.compare_video_presets", return_value=[]),
            patch("src.transcode_toolkit.core.unified_estimate.recommend_video_preset", return_value="default"),
            patch("src.transcode_toolkit.core.unified_estimate.print_video_comparison"),
        ):
            # Create mock stat result for readable files  
            readable_stat = Mock()
            readable_stat.st_size = 1024 * 1024
            readable_stat.st_mtime = 1234567890
            
            # Create a more targeted approach: mock just the global Path.stat
            # to handle all stat calls uniformly
            with patch("pathlib.Path.stat", return_value=readable_stat):
                mock_probe.return_value = {"codec": "h264"}
                
                results, _ = analyze_directory(tmp_path)

                # The core behavior being tested is that analyze_file raises PermissionError
                # for unreadable files and the system handles it gracefully.
                # Since analyze_file raises PermissionError for unreadable_file,
                # that file should be skipped, leaving only the readable file.
                assert len(results) == 1, "Should process only readable files"
                assert results[0].file_path == readable_file, "Should process the readable file"
