#!/usr/bin/env python3
"""Main entry point for the refactored media toolkit."""

import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from cli.main import main

if __name__ == "__main__":
    sys.exit(main())
