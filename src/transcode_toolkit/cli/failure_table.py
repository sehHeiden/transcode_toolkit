"""Shared failure table display utility for CLI commands."""


def print_failure_table(failed_results, media_type="media"):
    """
    Print a simple table showing conversion failures.

    Args:
        failed_results: List of ProcessingResult objects with error status
        media_type: Type of media being processed ("audio", "video", or "media")

    """
    if not failed_results:
        return

    print("\n" + "=" * 80)
    print(f"{'CONVERSION FAILURES':^80}")
    print("=" * 80)
    print(f"Total failed: {len(failed_results)} files\n")

    # Simple table header
    print(f"{'FILE':<40} | {'ERROR':<35}")
    print("-" * 80)

    for result in failed_results:
        # Truncate long file names
        filename = result.source_file.name
        if len(filename) > 37:
            filename = filename[:34] + "..."

        # Truncate long error messages
        error_msg = result.message or "Unknown error"
        if len(error_msg) > 32:
            error_msg = error_msg[:29] + "..."

        print(f"{filename:<40} | {error_msg:<35}")

    # Media-specific tips
    tips = {
        "video": "💡 TIP: Check GPU drivers, disk space, or try different presets",
        "audio": "💡 TIP: Check FFmpeg codecs, file permissions, or try different presets",
        "media": "💡 TIP: Check FFmpeg installation, drivers, or try different presets",
    }

    print(f"\n{tips.get(media_type, tips['media'])}\n")
