"""Base classes and interfaces for media processing."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

LOG = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Status of a processing operation."""

    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class ProcessingResult:
    """Result of a media processing operation."""

    source_file: Path
    status: ProcessingStatus
    message: str = ""
    output_file: Path | None = None
    original_size: int | None = None
    new_size: int | None = None
    processing_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ProcessingError(Exception):
    """Base exception for media processing errors."""

    def __init__(
        self,
        message: str,
        file_path: Path | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.file_path = file_path
        self.cause = cause


class MediaProcessor(ABC):
    """Abstract base class for media processors."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    @abstractmethod
    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle the given file."""

    @abstractmethod
    def should_process(self, file_path: Path, **kwargs) -> bool:
        """Check if the file should be processed (not already optimized)."""

    @abstractmethod
    def process_file(self, file_path: Path, **kwargs) -> ProcessingResult:
        """Process a single file."""

    def process_directory(self, directory: Path, recursive: bool = True, **kwargs) -> list[ProcessingResult]:
        """Process all compatible files in a directory."""
        results = []

        if not directory.exists():
            msg = f"Directory does not exist: {directory}"
            raise ProcessingError(msg)

        pattern = "**/*" if recursive else "*"
        files = [f for f in directory.glob(pattern) if f.is_file() and self.can_process(f)]

        self.logger.info(f"Found {len(files)} files to process in {directory}")

        for file_path in files:
            try:
                if self.should_process(file_path, **kwargs):
                    result = self.process_file(file_path, **kwargs)
                    results.append(result)
                else:
                    results.append(
                        ProcessingResult(
                            source_file=file_path,
                            status=ProcessingStatus.SKIPPED,
                            message="File already optimized or doesn't meet processing criteria",
                        )
                    )
            except Exception as e:
                self.logger.exception(f"Error processing {file_path}: {e}")
                results.append(
                    ProcessingResult(
                        source_file=file_path,
                        status=ProcessingStatus.ERROR,
                        message=str(e),
                    )
                )

        return results
