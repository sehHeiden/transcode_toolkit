from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ProcessingStatus(StrEnum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    ERROR = "error"


class ProcessingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: Path
    status: ProcessingStatus
    original_size: int
    new_size: int | None = None
