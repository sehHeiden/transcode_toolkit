"""
Entry point for the TUI.
"""

import sys
from tui import TranscodeApp


def main() -> int:
    """Run TranscodeApp TUI."""
    app = TranscodeApp()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
