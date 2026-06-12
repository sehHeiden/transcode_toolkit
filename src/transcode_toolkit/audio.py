from __future__ import annotations

import shutil
from functools import partial
from pathlib import Path

from .chain import Chain
from .config import AudioPreset, ToolkitConfig
from .ffmpeg import MediaInfo, build_audio_cmd, cleanup, measure_snr, run_ffmpeg, validate_duration
from .types import ProcessingResult, ProcessingStatus


def transcode_audio(path: Path, preset: AudioPreset, config: ToolkitConfig) -> ProcessingResult:
    source_size = path.stat().st_size
    try:
        info = MediaInfo.from_path(path)
    except Exception:
        return ProcessingResult(source=path, status=ProcessingStatus.ERROR, original_size=source_size)

    snr = measure_snr(path)
    bitrate = scale_bitrate(snr, preset, info.bitrate)
    timeout = max(300, info.duration * 10)

    output = path.with_suffix(".opus")
    cmd = build_audio_cmd(path, output, bitrate, preset)
    try:
        run_ffmpeg(cmd, timeout=timeout)
    except Exception:
        cleanup(output)
        return ProcessingResult(source=path, status=ProcessingStatus.ERROR, original_size=source_size)

    if output.stat().st_size >= source_size * config.audio.size_keep_ratio:
        cleanup(output)
        return ProcessingResult(source=path, status=ProcessingStatus.SKIPPED, original_size=source_size)

    if not validate_duration(output, info.duration):
        cleanup(output)
        return ProcessingResult(source=path, status=ProcessingStatus.ERROR, original_size=source_size)

    if config.global_.create_backups and path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))

    if path.suffix.lower() != ".opus":
        path.unlink()

    return ProcessingResult(
        source=path,
        status=ProcessingStatus.SUCCESS,
        original_size=source_size,
        new_size=output.stat().st_size,
    )


def scale_bitrate(snr: float, preset: AudioPreset, source_bitrate: int) -> int:
    target = _parse_bitrate(preset.bitrate)
    if not preset.snr_bitrate_scale or snr >= preset.min_snr_db:
        return min(target, source_bitrate)
    ratio = snr / preset.min_snr_db
    return min(target, max(int(source_bitrate * ratio), 16_000))


def should_process_audio(path: Path) -> bool:
    return path.suffix.lower() != ".opus"


def estimate_audio(path: Path, preset: AudioPreset) -> ProcessingResult:
    info = MediaInfo.from_path(path)
    source_size = path.stat().st_size
    snr = measure_snr(path)
    bitrate = scale_bitrate(snr, preset, info.bitrate)
    estimated_size = int(bitrate * info.duration / 8)
    return ProcessingResult(
        source=path,
        status=ProcessingStatus.SUCCESS,
        original_size=source_size,
        new_size=estimated_size,
    )


def transcode_audio_directory(path: Path, preset_name: str, config: ToolkitConfig) -> list[ProcessingResult]:
    preset = config.audio.presets[preset_name]
    processor = partial(transcode_audio, preset=preset, config=config)
    return (
        Chain(path)
        .discover(extensions=config.audio.extensions)
        .filter(should_process_audio)
        .transcode(processor, workers=config.global_.workers or 2)
    )


def _parse_bitrate(s: str) -> int:
    multipliers = {"k": 1000, "m": 1_000_000}
    suffix = s[-1].lower()
    return int(float(s.rstrip("kKmM")) * multipliers.get(suffix, 1))
