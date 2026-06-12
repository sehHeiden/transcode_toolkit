from __future__ import annotations

import shutil
from functools import partial
from pathlib import Path

from .chain import Chain
from .config import ToolkitConfig
from .ffmpeg import (
    MediaInfo,
    available_encoders,
    build_video_cmd,
    cleanup,
    has_vmaf_support,
    measure_vmaf,
    run_ffmpeg,
    validate_duration,
)
from .types import ProcessingResult, ProcessingStatus


def transcode_video(
    path: Path,
    *,
    codec: str = "libx265",
    crf: int = 24,
    speed: str = "medium",
    config: ToolkitConfig,
) -> ProcessingResult:
    source_size = path.stat().st_size

    if codec not in available_encoders():
        return ProcessingResult(source=path, status=ProcessingStatus.ERROR, original_size=source_size)

    try:
        info = MediaInfo.from_path(path)
    except Exception:
        return ProcessingResult(source=path, status=ProcessingStatus.ERROR, original_size=source_size)

    timeout = max(600, info.duration * 300)
    output = path.with_suffix(f".tmp{path.suffix}")
    cmd = build_video_cmd(path, output, codec=codec, crf=crf, speed=speed)

    try:
        run_ffmpeg(cmd, timeout=timeout)
    except Exception:
        cleanup(output)
        return ProcessingResult(source=path, status=ProcessingStatus.ERROR, original_size=source_size)

    output_size = output.stat().st_size
    if (1 - output_size / source_size) * 100 < config.video.min_savings_percent:
        cleanup(output)
        return ProcessingResult(source=path, status=ProcessingStatus.SKIPPED, original_size=source_size)

    if not validate_duration(output, info.duration):
        cleanup(output)
        return ProcessingResult(source=path, status=ProcessingStatus.ERROR, original_size=source_size)

    if config.global_.create_backups:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    shutil.move(str(output), str(path))

    return ProcessingResult(
        source=path,
        status=ProcessingStatus.SUCCESS,
        original_size=source_size,
        new_size=path.stat().st_size,
    )


def should_process_video(path: Path) -> bool:
    return path.stat().st_size > 1024 * 1024


def estimate_video(path: Path, config: ToolkitConfig) -> list[dict]:
    encoders = available_encoders()
    vmaf_available = has_vmaf_support()
    results = []
    for p in _PRESETS:
        if p["codec"] not in encoders:
            continue
        output = path.with_suffix(f".test{path.suffix}")
        cmd = build_video_cmd(path, output, codec=p["codec"], crf=p["crf"], speed=p["speed"])
        try:
            run_ffmpeg(cmd, timeout=60)
            vmaf = measure_vmaf(path, output) if vmaf_available else 0.0
            ratio = output.stat().st_size / path.stat().st_size
            results.append({**p, "vmaf": vmaf, "size_ratio": ratio})
        except Exception:
            pass
        finally:
            cleanup(output)
    return sorted(results, key=lambda r: r["size_ratio"])


def transcode_video_directory(
    path: Path, *, codec: str, crf: int, speed: str, config: ToolkitConfig
) -> list[ProcessingResult]:
    processor = partial(transcode_video, codec=codec, crf=crf, speed=speed, config=config)
    return (
        Chain(path)
        .discover(extensions=config.video.extensions)
        .filter(should_process_video)
        .transcode(processor, workers=config.global_.workers or 2)
    )


_PRESETS: list[dict] = (
    [
        {"codec": "libx265", "crf": c, "speed": s, "label": f"x265_crf{c}_{s}"}
        for c in (24, 28)
        for s in ("medium", "fast")
    ]
    + [
        {"codec": "libsvtav1", "crf": c, "speed": s, "label": f"svtav1_crf{c}_{s}"}
        for c in (30, 34)
        for s in ("medium", "fast")
    ]
    + [
        {"codec": "hevc_nvenc", "crf": c, "speed": s, "label": f"nvenc_crf{c}_{s}"}
        for c in (22, 26)
        for s in ("medium", "fast")
    ]
)
