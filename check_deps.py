#!/usr/bin/env python3
"""check_deps.py - Check and help install FFmpeg dependencies."""

import shutil
import subprocess
import sys


def check_ffmpeg_installation() -> bool:
    """Check if FFmpeg tools are available and suggest installation methods."""
    required = ["ffmpeg", "ffprobe"]
    missing = []

    print("🔍 Checking FFmpeg installation...")

    for exe in required:
        if shutil.which(exe):
            try:
                result = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=5, check=False)
                version_line = result.stdout.split("\n")[0]
                print(f"✅ {exe}: {version_line}")
            except subprocess.TimeoutExpired:
                print(f"⚠️  {exe}: Found but version check timed out")
            except Exception as e:
                print(f"⚠️  {exe}: Found but error checking version: {e}")
        else:
            missing.append(exe)
            print(f"❌ {exe}: Not found in PATH")

    if missing:
        print(f"\n💡 Missing executables: {', '.join(missing)}")
        print("\n📦 Installation options for Windows:")
        print("   1. Using winget (recommended):")
        print("      winget install Gyan.FFmpeg")
        print("\n   2. Using Chocolatey:")
        print("      choco install ffmpeg")
        print("\n   3. Manual download:")
        print("      • Visit: https://www.gyan.dev/ffmpeg/builds/")
        print("      • Download 'ffmpeg-release-essentials.zip'")
        print("      • Extract to C:\\ffmpeg")
        print("      • Add C:\\ffmpeg\\bin to your system PATH")
        print("\n   4. Using conda/mamba:")
        print("      conda install -c conda-forge ffmpeg")

        print("\n🔧 After installation:")
        print("   • Restart your terminal/PowerShell")
        print("   • Run 'ffmpeg -version' to verify")
        print("   • Or run this script again to check")

        return False
    print("\n✅ All FFmpeg tools are available!")
    return True


def check_python_environment() -> None:
    """Check if we're in the right Python environment."""
    print("\n🐍 Python environment:")
    print(f"   Version: {sys.version}")
    print(f"   Executable: {sys.executable}")

    # Check if we're in a virtual environment
    if hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix):
        print("   Environment: Virtual environment ✅")
    else:
        print("   Environment: System Python ⚠️")
        print("   Consider using a virtual environment for development")


def main() -> int:
    """Main function to check all dependencies."""
    print("🚀 Transcode Toolkit Dependency Checker")
    print("=" * 50)

    # Check Python environment
    check_python_environment()

    # Check FFmpeg
    ffmpeg_ok = check_ffmpeg_installation()

    print("\n" + "=" * 50)
    if ffmpeg_ok:
        print("🎉 All dependencies are ready!")
        print("\nYou can now use:")
        print("   uv run python media_toolkit.py audio estimate /path/to/audio")
        print("   uv run python media_toolkit.py audio transcode /path/to/audio")
    else:
        print("❌ Some dependencies are missing. Please install them first.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
