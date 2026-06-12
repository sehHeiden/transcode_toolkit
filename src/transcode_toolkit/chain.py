from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .types import ProcessingResult, ProcessingStatus


class Chain:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._files: list[Path] = []

    def discover(self, *, extensions: list[str]) -> Chain:
        self._files = [f for f in self._path.rglob("*") if f.suffix.lower() in extensions and f.is_file()]
        return self

    def filter(self, predicate: Callable[[Path], bool]) -> Chain:
        self._files = [f for f in self._files if predicate(f)]
        return self

    def transcode(self, processor: Callable[[Path], ProcessingResult], *, workers: int = 2) -> list[ProcessingResult]:
        results: list[ProcessingResult] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(processor, f): f for f in self._files}
            for future in as_completed(futures):
                exc = future.exception()
                if exc is None:
                    results.append(future.result())
                else:
                    source = futures[future]
                    try:
                        source_size = source.stat().st_size
                    except OSError:
                        source_size = 0
                    results.append(
                        ProcessingResult(source=source, status=ProcessingStatus.ERROR, original_size=source_size)
                    )
        return results
