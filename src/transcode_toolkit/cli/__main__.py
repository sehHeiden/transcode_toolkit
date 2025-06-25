"""Entry point when running the CLI module with python -m."""

import sys

from .main import main

if __name__ == "__main__":
    sys.exit(main())
