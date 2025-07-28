"""Test that higher speed (lower processing time) results in better weighted scores."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.transcode_toolkit.core.unified_estimate import (
    DEFAULT_WEIGHTS,
    EstimationResult,
    calculate_weighted_score,
    recommend_video_preset,
)


@pytest.fixture
def white_noise_video_file(tmp_path: Path) -> Path:
    """Create a mock white noise video file for testing."""
    test_file = tmp_path / "white_noise.mp4"
    # Create a file with random-like content to simulate white noise
    import secrets

    # Use secrets for non-cryptographic purposes as ruff suggests
    random_data = secrets.token_bytes(10000)
    test_file.write_bytes(random_data)
    return test_file


def test_higher_speed_better_score() -> None:
    """Test that higher encoding speed (lower processing time) results in better weighted scores."""
    # Test data: Same quality and savings, but different processing times
    # Lower processing time should result in higher score

    # Fast preset (2 minutes processing time)
    fast_preset_ssim = 0.92
    fast_preset_time = 2.0  # minutes
    fast_preset_savings = 30.0  # percent

    # Slow preset (10 minutes processing time)
    slow_preset_ssim = 0.92  # Same quality
    slow_preset_time = 10.0  # minutes - much slower
    slow_preset_savings = 30.0  # Same savings

    # Calculate scores with maximum normalization
    max_time = max(fast_preset_time, slow_preset_time)

    fast_score = calculate_weighted_score(
        ssim=fast_preset_ssim,
        processing_time=fast_preset_time,
        saving_percent=fast_preset_savings,
        weights=DEFAULT_WEIGHTS,
        max_processing_time=max_time,
    )

    slow_score = calculate_weighted_score(
        ssim=slow_preset_ssim,
        processing_time=slow_preset_time,
        saving_percent=slow_preset_savings,
        weights=DEFAULT_WEIGHTS,
        max_processing_time=max_time,
    )

    # Fast preset should have higher score due to better speed
    assert fast_score > slow_score, (
        f"Fast preset score ({fast_score:.3f}) should be higher than slow preset score ({slow_score:.3f})"
    )

    # Verify the difference is significant (at least minimum threshold)
    min_score_difference = 0.1
    score_difference = fast_score - slow_score
    assert score_difference >= min_score_difference, (
        f"Score difference ({score_difference:.3f}) should be at least {min_score_difference}"
    )


def test_speed_weight_impact() -> None:
    """Test that increasing the speed weight increases the impact of processing time on scores."""
    # Test with different weight configurations
    quality_focused_weights = {"quality": 0.7, "speed": 0.1, "savings": 0.2}
    speed_focused_weights = {"quality": 0.2, "speed": 0.7, "savings": 0.1}

    fast_time = 2.0
    slow_time = 10.0
    max_time = slow_time
    ssim = 0.92
    savings = 30.0

    # Calculate scores with quality-focused weights
    fast_score_quality_focused = calculate_weighted_score(
        ssim=ssim,
        processing_time=fast_time,
        saving_percent=savings,
        weights=quality_focused_weights,
        max_processing_time=max_time,
    )
    slow_score_quality_focused = calculate_weighted_score(
        ssim=ssim,
        processing_time=slow_time,
        saving_percent=savings,
        weights=quality_focused_weights,
        max_processing_time=max_time,
    )

    # Calculate scores with speed-focused weights
    fast_score_speed_focused = calculate_weighted_score(
        ssim=ssim,
        processing_time=fast_time,
        saving_percent=savings,
        weights=speed_focused_weights,
        max_processing_time=max_time,
    )
    slow_score_speed_focused = calculate_weighted_score(
        ssim=ssim,
        processing_time=slow_time,
        saving_percent=savings,
        weights=speed_focused_weights,
        max_processing_time=max_time,
    )

    quality_focused_diff = fast_score_quality_focused - slow_score_quality_focused
    speed_focused_diff = fast_score_speed_focused - slow_score_speed_focused

    # Speed-focused weighting should create larger difference between fast and slow
    assert speed_focused_diff > quality_focused_diff, (
        f"Speed-focused difference ({speed_focused_diff:.3f}) should be "
        f"larger than quality-focused difference ({quality_focused_diff:.3f})"
    )


def test_preset_recommendation_favors_speed() -> None:
    """Test that preset recommendation system favors faster presets when quality/savings are similar."""
    # Create mock estimation results with similar quality/savings but different speeds
    fast_preset_result = EstimationResult(
        preset="fast_preset",
        current_size=100 * 1024 * 1024,  # 100MB
        estimated_size=70 * 1024 * 1024,  # 70MB
        saving=30 * 1024 * 1024,  # 30MB saved
        saving_percent=30.0,
        predicted_ssim=0.92,
    )

    slow_preset_result = EstimationResult(
        preset="slow_preset",
        current_size=100 * 1024 * 1024,  # Same size
        estimated_size=70 * 1024 * 1024,  # Same compression
        saving=30 * 1024 * 1024,  # Same savings
        saving_percent=30.0,  # Same savings percent
        predicted_ssim=0.92,  # Same quality
    )

    results = [fast_preset_result, slow_preset_result]

    # Mock the processing time calculation to return different speeds
    with (
        patch("src.transcode_toolkit.core.unified_estimate._calculate_processing_time") as mock_calc_time,
        patch("src.transcode_toolkit.core.unified_estimate.ConfigManager") as mock_config_manager,
    ):
        # Mock config manager
        mock_config = Mock()
        mock_config.config.get_video_preset.return_value = Mock(crf=23, codec="libx265", preset="medium")
        mock_config_manager.return_value = mock_config

        # Set up processing times: fast_preset = 2min, slow_preset = 10min
        def mock_processing_time(_size_mb: float, _preset_config: Mock) -> float:
            # The preset_name is available but we'll use the side_effect for control
            return 2.0  # Will be overridden by side_effect

        mock_calc_time.side_effect = [2.0, 10.0]  # fast_preset=2min, slow_preset=10min

        # Get recommendation
        recommended = recommend_video_preset(results)

        # Fast preset should be recommended due to better speed
        assert recommended == "fast_preset", f"Expected 'fast_preset' to be recommended, got '{recommended}'"


def test_white_noise_video_processing_speed_impact(white_noise_video_file: Path) -> None:
    """Test speed impact using a simulated white noise video file."""
    # This test uses the white noise file fixture to simulate a more realistic scenario
    # We'll mock the actual video processing but use the file for path operations

    assert white_noise_video_file.exists(), "White noise video file should exist"
    assert white_noise_video_file.stat().st_size > 0, "White noise video file should have content"

    # Create estimation results for different speed presets
    gpu_preset = EstimationResult(
        preset="hevc_nvenc_fast",  # GPU preset - should be fastest
        current_size=white_noise_video_file.stat().st_size,
        estimated_size=int(white_noise_video_file.stat().st_size * 0.7),  # 30% compression
        saving=int(white_noise_video_file.stat().st_size * 0.3),
        saving_percent=30.0,
        predicted_ssim=0.90,  # Slightly lower quality due to GPU
    )

    cpu_fast_preset = EstimationResult(
        preset="libx265_fast",  # CPU fast preset
        current_size=white_noise_video_file.stat().st_size,
        estimated_size=int(white_noise_video_file.stat().st_size * 0.65),  # Better compression
        saving=int(white_noise_video_file.stat().st_size * 0.35),
        saving_percent=35.0,  # Better savings
        predicted_ssim=0.93,  # Better quality
    )

    cpu_slow_preset = EstimationResult(
        preset="libx265_slower",  # CPU slow preset
        current_size=white_noise_video_file.stat().st_size,
        estimated_size=int(white_noise_video_file.stat().st_size * 0.60),  # Best compression
        saving=int(white_noise_video_file.stat().st_size * 0.40),
        saving_percent=40.0,  # Best savings
        predicted_ssim=0.95,  # Best quality
    )

    results = [gpu_preset, cpu_fast_preset, cpu_slow_preset]

    # Mock processing times: GPU=1min, CPU_fast=5min, CPU_slow=20min
    with (
        patch("src.transcode_toolkit.core.unified_estimate._calculate_processing_time") as mock_calc_time,
        patch("src.transcode_toolkit.core.unified_estimate.ConfigManager") as mock_config_manager,
    ):
        mock_config = Mock()
        mock_config.config.get_video_preset.return_value = Mock()
        mock_config_manager.return_value = mock_config

        # Set processing times based on preset type
        def get_processing_time(_size_mb: float, _preset_config: Mock) -> float:
            # This will be called for each result in order
            return 1.0  # Will be controlled by side_effect

        mock_calc_time.side_effect = [1.0, 5.0, 20.0]  # GPU, CPU_fast, CPU_slow

        # Test with speed-focused weights
        speed_focused_weights = {"quality": 0.2, "speed": 0.6, "savings": 0.2}
        recommended = recommend_video_preset(results, weights=speed_focused_weights)

        # With speed-focused weights, GPU preset should win despite lower quality
        assert recommended == "hevc_nvenc_fast", (
            f"With speed-focused weights, expected 'hevc_nvenc_fast', got '{recommended}'"
        )

        # Reset mock for next test
        mock_calc_time.side_effect = [1.0, 5.0, 20.0]

        # Test with quality-focused weights
        quality_focused_weights = {"quality": 0.7, "speed": 0.1, "savings": 0.2}
        recommended_quality = recommend_video_preset(results, weights=quality_focused_weights)

        # With quality-focused weights, slower preset might win due to better quality
        # (This depends on the exact scoring, but it should be different from speed-focused)
        assert recommended_quality != recommended or recommended_quality in ["libx265_slower", "libx265_fast"], (
            f"Quality-focused recommendation should prioritize quality: got '{recommended_quality}'"
        )


def test_speed_score_calculation_edge_cases() -> None:
    """Test edge cases in speed score calculation."""
    # Test with zero processing time (theoretical maximum speed)
    zero_time_score = calculate_weighted_score(
        ssim=0.90, processing_time=0.0, saving_percent=30.0, weights=DEFAULT_WEIGHTS, max_processing_time=10.0
    )

    # Test with maximum processing time (slowest)
    max_time_score = calculate_weighted_score(
        ssim=0.90, processing_time=10.0, saving_percent=30.0, weights=DEFAULT_WEIGHTS, max_processing_time=10.0
    )

    # Zero time should result in maximum speed score contribution
    assert zero_time_score > max_time_score, "Zero processing time should result in higher score"

    # Test with no max_processing_time provided (fallback formula)
    fallback_score = calculate_weighted_score(
        ssim=0.90, processing_time=5.0, saving_percent=30.0, weights=DEFAULT_WEIGHTS, max_processing_time=None
    )

    # Should still return a valid score
    assert 0.0 <= fallback_score <= 1.0, f"Fallback score should be between 0 and 1, got {fallback_score}"
