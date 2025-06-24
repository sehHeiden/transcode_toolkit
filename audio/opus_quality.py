"""audio.opus_quality – (placeholder) analyse audio and propose Opus params."""

import json
import logging
from pathlib import Path


def cli(path: Path, *, json_path: str | None, verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    info = {"files": 0, "note": "audio quality analysis not yet implemented"}
    print(info)
    if json_path:
        Path(json_path).write_text(json.dumps(info))
