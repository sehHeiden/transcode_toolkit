"""Shared failure table display utility for CLI commands."""

# Constants for table formatting
MAX_FILENAME_LENGTH = 37
FILENAME_TRUNCATE_LENGTH = 34
MAX_ERROR_MSG_LENGTH = 32
ERROR_MSG_TRUNCATE_LENGTH = 29


def print_failure_table(failed_results: list, media_type: str = "media") -> None:
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
        if len(filename) > MAX_FILENAME_LENGTH:
            filename = filename[:FILENAME_TRUNCATE_LENGTH] + "..."

        # Truncate long error messages
        error_msg = result.message or "Unknown error"
        if len(error_msg) > MAX_ERROR_MSG_LENGTH:
            error_msg = error_msg[:ERROR_MSG_TRUNCATE_LENGTH] + "..."

        print(f"{filename:<40} | {error_msg:<35}")

    # Media-specific tips
    tips = {
        "video": "💡 TIP: Check GPU drivers, disk space, or try different presets",
        "audio": "💡 TIP: Check FFmpeg codecs, file permissions, or try different presets",
        "media": "💡 TIP: Check FFmpeg installation, drivers, or try different presets",
    }

    print(f"\n{tips.get(media_type, tips['media'])}\n")
