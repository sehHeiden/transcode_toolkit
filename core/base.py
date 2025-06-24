"""Base classes and interfaces for media processing."""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

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
    output_file: Optional[Path] = None
    original_size: Optional[int] = None
    new_size: Optional[int] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def size_reduction(self) -> Optional[float]:
        """Calculate size reduction ratio (0.0 to 1.0)."""
        if self.original_size and self.new_size:
            return (self.original_size - self.new_size) / self.original_size
        return None
    
    @property
    def size_reduction_percent(self) -> Optional[float]:
        """Calculate size reduction as percentage."""
        reduction = self.size_reduction
        return reduction * 100 if reduction is not None else None


class ProcessingError(Exception):
    """Base exception for media processing errors."""
    
    def __init__(self, message: str, file_path: Optional[Path] = None, cause: Optional[Exception] = None):
        super().__init__(message)
        self.file_path = file_path
        self.cause = cause


class MediaProcessor(ABC):
    """Abstract base class for media processors."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    @abstractmethod
    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle the given file."""
        pass
    
    @abstractmethod
    def should_process(self, file_path: Path, **kwargs) -> bool:
        """Check if the file should be processed (not already optimized)."""
        pass
    
    @abstractmethod
    def process_file(self, file_path: Path, **kwargs) -> ProcessingResult:
        """Process a single file."""
        pass
    
    def process_directory(
        self, 
        directory: Path, 
        recursive: bool = True,
        **kwargs
    ) -> List[ProcessingResult]:
        """Process all compatible files in a directory."""
        results = []
        
        if not directory.exists():
            raise ProcessingError(f"Directory does not exist: {directory}")
        
        pattern = "**/*" if recursive else "*"
        files = [f for f in directory.glob(pattern) if f.is_file() and self.can_process(f)]
        
        self.logger.info(f"Found {len(files)} files to process in {directory}")
        
        for file_path in files:
            try:
                if self.should_process(file_path, **kwargs):
                    result = self.process_file(file_path, **kwargs)
                    results.append(result)
                else:
                    results.append(ProcessingResult(
                        source_file=file_path,
                        status=ProcessingStatus.SKIPPED,
                        message="File already optimized or doesn't meet processing criteria"
                    ))
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                results.append(ProcessingResult(
                    source_file=file_path,
                    status=ProcessingStatus.ERROR,
                    message=str(e)
                ))
        
        return results


class Estimator(Protocol):
    """Protocol for size estimation."""
    
    def estimate_size(self, file_path: Path, **kwargs) -> int:
        """Estimate the size after processing."""
        ...
    
    def estimate_directory(self, directory: Path, **kwargs) -> Dict[str, Any]:
        """Estimate sizes for all files in directory."""
        ...
