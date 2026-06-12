from __future__ import annotations

from pathlib import Path

import pytest

from transcode_toolkit.config import ToolkitConfig
from transcode_toolkit.ffmpeg import available_encoders, has_vmaf_support, measure_vmaf, probe
from transcode_toolkit.types import ProcessingStatus
from transcode_toolkit.video import estimate_video, should_process_video, transcode_video


class TestTranscodeVideo:
    def test_noisy_video_compresses_well(self, noisy_video: Path, tmp_config: Path) -> None:
        available_encoders.cache_clear()
        config = ToolkitConfig.from_yaml(tmp_config)
        result = transcode_video(noisy_video, codec="libx265", crf=28, speed="fast", config=config)
        assert result.status == ProcessingStatus.SUCCESS
        assert result.new_size is not None
        assert result.new_size < result.original_size

    def test_compressed_video_skipped(self, compressed_video: Path, tmp_config: Path) -> None:
        available_encoders.cache_clear()
        config = ToolkitConfig.from_yaml(tmp_config)
        result = transcode_video(compressed_video, codec="libx265", crf=28, speed="fast", config=config)
        assert result.status in (ProcessingStatus.SKIPPED, ProcessingStatus.SUCCESS)

    def test_tiny_video_filtered(self, tiny_video: Path) -> None:
        assert not should_process_video(tiny_video)

    def test_large_video_passes_filter(self, noisy_video: Path) -> None:
        assert should_process_video(noisy_video)


class TestEstimateVideo:
    def test_estimate_returns_presets(self, noisy_video: Path, tmp_config: Path) -> None:
        available_encoders.cache_clear()
        config = ToolkitConfig.from_yaml(tmp_config)
        results = estimate_video(noisy_video, config)
        assert len(results) > 0
        for r in results:
            assert "size_ratio" in r
            assert r["size_ratio"] > 0

    def test_estimate_sorted_by_size_ratio(self, noisy_video: Path, tmp_config: Path) -> None:
        available_encoders.cache_clear()
        config = ToolkitConfig.from_yaml(tmp_config)
        results = estimate_video(noisy_video, config)
        ratios = [r["size_ratio"] for r in results]
        assert ratios == sorted(ratios)


@pytest.mark.skipif(not has_vmaf_support(), reason="libvmaf not available")
class TestVmaf:
    def test_same_file_quality(self, noisy_video: Path) -> None:
        score = measure_vmaf(noisy_video, noisy_video)
        assert score >= 99.0

    def test_probe_video_duration(self, noisy_video: Path) -> None:
        info = probe(str(noisy_video), noisy_video.stat().st_mtime)
        duration = float(info.get("format", {}).get("duration", 0))
        assert duration > 0


class TestAvailableEncoders:
    def test_libx265_available(self) -> None:
        available_encoders.cache_clear()
        encoders = available_encoders()
        assert "libx265" in encoders

    def test_libopus_available(self) -> None:
        available_encoders.cache_clear()
        encoders = available_encoders()
        assert "libopus" in encoders
