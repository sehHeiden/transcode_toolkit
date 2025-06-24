"""audio.estimate – Opus re-encode size estimator."""

import csv
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, NamedTuple, cast
from config import get_config


class EstimationResult(NamedTuple):
    preset: str
    current_size: int
    estimated_size: int
    saving: int
    saving_percent: float


LOG = logging.getLogger("audio-est")


def _check_executables() -> None:
    """Check if required executables are available in PATH."""
    required = ["ffprobe"]
    missing = []

    for exe in required:
        if not shutil.which(exe):
            missing.append(exe)

    if missing:
        print("❌ Missing required executables:")
        for exe in missing:
            print(f"   - {exe}")
        print("\n💡 Install FFmpeg to get these tools:")
        print("   • Windows: https://ffmpeg.org/download.html#build-windows")
        print("   • winget install Gyan.FFmpeg")
        print("   • choco install ffmpeg")
        print("   • Or download from: https://www.gyan.dev/ffmpeg/builds/")
        print("\n   Make sure to add FFmpeg to your system PATH.")
        raise SystemExit(1)


def _probe(path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_entries",
        "format=duration,bit_rate",
        str(path),
    ]
    fmt = json.loads(
        subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    )["format"]
    return {"duration": float(fmt.get("duration", 0)), "size": path.stat().st_size}


def _estimate(meta: Dict[str, Any], br: int) -> int:
    return int(meta["duration"] * br / 8)


def analyse(root: Path, target_br: int) -> List[Tuple[Path, int, int]]:
    config = get_config()
    audio_exts = config.audio.extensions
    
    rows = []
    for p in root.rglob("*"):
        if p.suffix.lower() in audio_exts:
            try:
                meta = _probe(p)
            except subprocess.CalledProcessError as exc:
                LOG.warning("ffprobe failed for %s: %s", p, exc)
                continue
            rows.append((p, meta["size"], _estimate(meta, target_br)))
    return rows


def compare_presets(root: Path) -> List[EstimationResult]:
    """Compare all presets and return estimation results for each."""
    config = get_config()
    audio_exts = config.audio.extensions
    
    files = []
    for p in root.rglob("*"):
        if p.suffix.lower() in audio_exts:
            try:
                meta = _probe(p)
                files.append((p, meta))
            except subprocess.CalledProcessError as exc:
                LOG.warning("ffprobe failed for %s: %s", p, exc)
                continue

    results = []
    for preset_name, preset_config in config.audio.presets.items():
        # Convert bitrate string to bits per second
        bitrate = int(preset_config.bitrate.rstrip("k")) * 1000
        
        current_size = sum(meta["size"] for _, meta in files)
        estimated_size = sum(_estimate(meta, bitrate) for _, meta in files)
        saving = current_size - estimated_size
        saving_percent = 0 if current_size == 0 else 100 * saving / current_size

        results.append(
            EstimationResult(
                preset=preset_name,
                current_size=current_size,
                estimated_size=estimated_size,
                saving=saving,
                saving_percent=saving_percent,
            )
        )

    return results


def recommend_preset(results: List[EstimationResult]) -> str:
    """Recommend the best preset based on analysis."""
    if not results:
        return "music"  # default fallback

    # Sort by saving percentage (descending)
    sorted_results = sorted(results, key=lambda x: x.saving_percent, reverse=True)

    best = sorted_results[0]

    # If best saving is less than 5%, recommend keeping original
    if best.saving_percent < 5:
        return "no_conversion"  # Special case

    # For audiobooks, prefer audiobook presets if they're in top 2
    audiobook_presets = [r for r in sorted_results[:2] if "audiobook" in r.preset]
    if audiobook_presets:
        return audiobook_presets[0].preset

    return best.preset


def print_comparison(results: List[EstimationResult], recommended: str) -> None:
    """Print a formatted comparison of all presets."""
    print("\n=== PRESET COMPARISON ===")
    print(f"{'Preset':<20} {'Current':>10} {'Estimated':>10} {'Saving':>10} {'%':>8}")
    print("-" * 60)

    for result in sorted(results, key=lambda x: x.saving_percent, reverse=True):
        marker = " ★" if result.preset == recommended else "  "
        print(
            f"{result.preset + marker:<20} "
            f"{result.current_size / 2**20:8.1f} MB "
            f"{result.estimated_size / 2**20:8.1f} MB "
            f"{result.saving / 2**20:8.1f} MB "
            f"{result.saving_percent:6.1f}%"
        )

    print(f"\n★ RECOMMENDED: {recommended}")

    if recommended == "no_conversion":
        print("  → Files are already well compressed. Conversion may not be worth it.")
    else:
        config = get_config()
        preset_config = config.audio.presets.get(recommended)
        if preset_config:
            print(f"  → Bitrate: {preset_config.bitrate}")
            print(f"  → Application: {preset_config.application}")
            print(f"  → Description: {preset_config.description}")
            if preset_config.cutoff:
                print(f"  → Frequency cutoff: {preset_config.cutoff} Hz")
            if preset_config.channels:
                print(f"  → Channels: {preset_config.channels}")


def print_summary(rows, *, preset: str, csv_path: str | None = None) -> None:
    cur = sum(r[1] for r in rows)
    new = sum(r[2] for r in rows)
    diff = cur - new
    pct = 0 if cur == 0 else 100 * diff / cur
    print(f"{preset} files analysed : {len(rows):,}")
    print(f"Current size            : {cur / 2**30:.2f} GiB")
    print(f"Estimated Opus          : {new / 2**30:.2f} GiB")
    print(f"Potential saving        : {diff / 2**20:.1f} MiB ({pct:.1f} %)")

    if csv_path:
        csv_path_obj = Path(csv_path)
        csv_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with csv_path_obj.open("w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["file", "current", "estimated", "saving"])
            for p, c, e in rows:
                wr.writerow([p, c, e, c - e])
            wr.writerow(["TOTAL", cur, new, diff])
        print("CSV report →", csv_path_obj)


def cli(
    path: Path,
    *,
    preset: str | None = None,
    csv_path: str | None = None,
    json_path: str | None = None,
    verbose: bool = False,
    compare: bool = False,
) -> None:
    """CLI entry point for audio estimation.

    Args:
        path: Directory to analyze
        preset: Specific preset to analyze (if None, compares all)
        csv_path: Path to save CSV report
        json_path: Path to save JSON report
        verbose: Enable verbose logging
        compare: Force comparison mode even if preset is specified
    """
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Check if required executables are available
    _check_executables()

    if compare or preset is None:
        # Compare all presets and show recommendations
        results = compare_presets(path)
        recommended = recommend_preset(results)
        print_comparison(results, recommended)

        if json_path:
            config = get_config()
            output_data = {
                "recommended_preset": recommended,
                "available_presets": config.list_audio_presets(),
                "comparisons": [
                    {
                        "preset": r.preset,
                        "current_bytes": r.current_size,
                        "estimated_bytes": r.estimated_size,
                        "saving_bytes": r.saving,
                        "saving_percent": r.saving_percent,
                        "preset_config": config.audio.presets[r.preset].model_dump(),
                    }
                    for r in results
                ],
            }
            Path(json_path).write_text(json.dumps(output_data, indent=2))
            print(f"\nJSON report → {json_path}")

        print(
            f"\n💡 To convert with recommended settings:\n   python -m audio.transcode /path/to/audio --preset {recommended}"
        )
        return

    # Original single preset analysis
    config = get_config()
    if preset not in config.audio.presets:
        available = ", ".join(config.list_audio_presets())
        raise ValueError(f"Invalid preset: {preset!r}. Available: {available}")

    # Convert bitrate string to bits per second
    preset_config = config.audio.presets[preset]
    target_bitrate = int(preset_config.bitrate.rstrip("k")) * 1000
    
    rows = analyse(path, target_bitrate)
    print_summary(rows, preset=preset, csv_path=csv_path)

    if json_path:
        cur = sum(r[1] for r in rows)
        new = sum(r[2] for r in rows)
        diff = cur - new
        pct = 0 if cur == 0 else 100 * diff / cur
        json_output_data: Dict[str, Any] = {
            "preset_used": preset,
            "preset_config": preset_config.model_dump(),
            "files": len(rows),
            "current_bytes": cur,
            "estimated_bytes": new,
            "saving_bytes": diff,
            "saving_percent": pct,
        }
        Path(json_path).write_text(json.dumps(json_output_data, indent=2))
        print(f"\nJSON report → {json_path}")
