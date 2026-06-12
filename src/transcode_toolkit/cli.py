from __future__ import annotations

import shutil
from pathlib import Path

import typer

from .config import ToolkitConfig
from .types import ProcessingResult, ProcessingStatus

app = typer.Typer()
audio_app = typer.Typer()
video_app = typer.Typer()
utils_app = typer.Typer()
app.add_typer(audio_app, name="audio")
app.add_typer(video_app, name="video")
app.add_typer(utils_app, name="utils")


@app.command()
def tui(config_path: Path | None = typer.Option(None, "--config")) -> None:
    from .tui import run_tui

    run_tui(config_path)


def _config(path: Path | None) -> ToolkitConfig:
    return ToolkitConfig.from_yaml(path or Path("config.yaml"))


@audio_app.command("estimate")
def audio_estimate(
    path: Path = typer.Argument(exists=True),
    preset: str | None = typer.Option(None),
    csv_path: Path | None = typer.Option(None, "--csv"),
    config_path: Path | None = typer.Option(None, "--config"),
) -> None:
    import csv as csv_mod

    from .audio import estimate_audio

    config = _config(config_path)
    presets = {preset: config.audio.presets[preset]} if preset else config.audio.presets
    files = [f for f in path.rglob("*") if f.suffix.lower() in config.audio.extensions and f.is_file()]
    for name, p in presets.items():
        results = [estimate_audio(f, p) for f in files]
        total_src = sum(r.original_size for r in results)
        total_est = sum(r.new_size or 0 for r in results)
        savings_pct = (1 - total_est / total_src) * 100 if total_src else 0
        print(f"{name:20s} {total_src / 1e6:.1f} MB -> {total_est / 1e6:.1f} MB ({savings_pct:.1f}% saved)")
        if csv_path:
            with csv_path.open("a", newline="") as f:
                writer = csv_mod.writer(f)
                for r in results:
                    writer.writerow([name, str(r.source), r.original_size, r.new_size])


@audio_app.command("transcode")
def audio_transcode(
    path: Path = typer.Argument(exists=True),
    preset: str = typer.Option("music"),
    config_path: Path | None = typer.Option(None, "--config"),
) -> None:
    from .audio import transcode_audio_directory

    results = transcode_audio_directory(path, preset, _config(config_path))
    _print_summary(results)


@video_app.command("estimate")
def video_estimate(
    path: Path = typer.Argument(exists=True),
    config_path: Path | None = typer.Option(None, "--config"),
) -> None:
    from .video import estimate_video

    results = estimate_video(path, _config(config_path))
    print(f"{'Preset':25s} {'VMAF':>6s} {'Ratio':>6s}")
    print("-" * 40)
    for r in results:
        print(f"{r['label']:25s} {r['vmaf']:6.1f} {r['size_ratio']:6.2f}")


@video_app.command("transcode")
def video_transcode(
    path: Path = typer.Argument(exists=True),
    gpu: bool = typer.Option(False, "--gpu"),
    crf: int = typer.Option(24, min=0, max=51),
    speed: str = typer.Option("medium"),
    config_path: Path | None = typer.Option(None, "--config"),
) -> None:
    from .video import transcode_video_directory

    codec = "hevc_nvenc" if gpu else "libx265"
    results = transcode_video_directory(path, codec=codec, crf=crf, speed=speed, config=_config(config_path))
    _print_summary(results)


@utils_app.command()
def duplicates(
    path: Path = typer.Argument(exists=True),
    extensions: list[str] = typer.Option([".mp3", ".flac", ".m4a"]),
) -> None:
    import xxhash

    files = [f for f in path.rglob("*") if f.suffix.lower() in extensions and f.is_file()]
    by_size: dict[int, list[Path]] = {}
    for f in files:
        by_size.setdefault(f.stat().st_size, []).append(f)
    candidates = {s: fs for s, fs in by_size.items() if len(fs) > 1}
    hashes: dict[str, list[Path]] = {}
    for fs in candidates.values():
        for f in fs:
            h = xxhash.xxh128(f.read_bytes()).hexdigest()
            hashes.setdefault(h, []).append(f)
    dupes = {h: fs for h, fs in hashes.items() if len(fs) > 1}
    for fs in dupes.values():
        print(f"\nDuplicate ({fs[0].stat().st_size / 1e6:.1f} MB each):")
        for f in fs:
            print(f"  {f}")


@utils_app.command()
def info() -> None:
    print(f"ffmpeg:  {'found' if shutil.which('ffmpeg') else 'NOT FOUND'}")
    print(f"ffprobe: {'found' if shutil.which('ffprobe') else 'NOT FOUND'}")


def _print_summary(results: list[ProcessingResult]) -> None:
    by_status = {s: [r for r in results if r.status == s] for s in ProcessingStatus}
    for status, items in by_status.items():
        if not items:
            continue
        extra = ""
        if status == ProcessingStatus.SUCCESS:
            total_saved = sum(r.original_size - (r.new_size or r.original_size) for r in items)
            extra = f" ({total_saved / 1e6:.1f} MB saved)" if total_saved else ""
        print(f"{status}: {len(items)}{extra}")
