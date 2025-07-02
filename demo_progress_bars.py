#!/usr/bin/env python3
"""Demo script to show separate progress bars for video and audio analysis."""

import time

from tqdm import tqdm


def demo_separate_progress_bars():
    """Demonstrate separate progress bars for video and audio processing."""
    print("🎯 DEMO: Separate Progress Bars for Video and Audio Analysis")
    print("=" * 65)
    print()

    # Simulate some files
    video_files = ["movie1.mp4", "movie2.mkv", "movie3.avi"]
    audio_files = ["song1.mp3", "song2.flac", "audiobook.m4a", "podcast.wav"]

    print(f"Found {len(video_files)} video files and {len(audio_files)} audio files")
    print()

    # Create separate progress bars
    video_progress = tqdm(total=len(video_files), desc="Analyzing video", unit="file", position=0)

    audio_progress = tqdm(total=len(audio_files), desc="Analyzing audio", unit="file", position=1)

    try:
        # Process video files
        for video_file in video_files:
            time.sleep(0.5)  # Simulate processing time
            video_progress.set_postfix_str(f"Processing {video_file}")
            video_progress.update(1)

        # Process audio files
        for audio_file in audio_files:
            time.sleep(0.3)  # Simulate processing time
            audio_progress.set_postfix_str(f"Processing {audio_file}")
            audio_progress.update(1)

    finally:
        # Clean up progress bars
        video_progress.close()
        audio_progress.close()

    print()
    print("✅ Analysis complete!")
    print()
    print("OPTIMAL AUDIO SETTINGS:")
    print("   Best audio preset: music")
    print("   Potential savings: 25-35%")
    print()
    print("OPTIMAL VIDEO SETTINGS:")
    print("   Best video preset: h265_balanced")
    print("   Potential savings: 40-60%")
    print()
    print("🚀 RECOMMENDED COMMANDS:")
    print("   Video: transcode-toolkit video transcode --preset h265_balanced .")
    print("   Audio: transcode-toolkit audio transcode --preset music .")


if __name__ == "__main__":
    demo_separate_progress_bars()
