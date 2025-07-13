"""Enhanced file and backup management."""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from .base import ProcessingError

if TYPE_CHECKING:
    from pathlib import Path

LOG = logging.getLogger(__name__)


class BackupStrategy(Enum):
    """Backup creation strategies."""

    NEVER = "never"
    ON_SUCCESS = "on_success"


@dataclass
class FileOperation:
    """Represents a file operation that can be rolled back."""

    operation_type: str
    source_path: Path
    backup_path: Path | None = None
    target_path: Path | None = None
    success: bool = False
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class FileManager:
    """Enhanced file manager with atomic operations and backup management."""

    def __init__(
        self,
        backup_strategy: BackupStrategy = BackupStrategy.ON_SUCCESS,
    ) -> None:
        """Initialize file manager with backup strategy."""
        self.backup_strategy = backup_strategy
        self.session_operations: list[FileOperation] = []
        self.session_backups: set[Path] = set()

    def create_backup(self, file_path: Path) -> Path | None:
        """Create a backup of the file."""
        if self.backup_strategy == BackupStrategy.NEVER:
            return None

        backup_path = file_path.with_suffix(file_path.suffix + ".bak")

        try:
            # Ensure backup directory exists
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            # Create backup with metadata preservation
            shutil.copy2(file_path, backup_path)

            # Track the backup
            self.session_backups.add(backup_path)

            operation = FileOperation(
                operation_type="backup_create",
                source_path=file_path,
                backup_path=backup_path,
                success=True,
            )
            self.session_operations.append(operation)
            LOG.debug("Created backup: %s", backup_path)
        except Exception as e:
            LOG.exception("Failed to create backup for %s", file_path)
            msg = f"Backup creation failed: {e}"
            raise ProcessingError(msg, file_path=file_path, cause=e) from e
        else:
            return backup_path

    def atomic_replace(self, source_path: Path, temp_path: Path, *, create_backup: bool = True) -> FileOperation:
        """Atomically replace a file with proper backup handling."""
        backup_path = None

        try:
            # Create backup if requested and strategy allows
            if create_backup and self.backup_strategy != BackupStrategy.NEVER:
                backup_path = self.create_backup(source_path)

            # Ensure target directory exists
            source_path.parent.mkdir(parents=True, exist_ok=True)

            # Atomic move operation
            if source_path.exists():
                source_path.unlink()

            # Move temp file to final location
            # Keep original extension for video files, or use .opus for audio files
            if temp_path.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}:
                # Video file - keep original extension
                final_path = source_path
            elif source_path.suffix.lower() not in {".opus"}:
                # Audio file - change to .opus
                final_path = source_path.with_suffix(".opus")
            else:
                # Already correct extension
                final_path = source_path

            shutil.move(temp_path, final_path)

            operation = FileOperation(
                operation_type="file_replace",
                source_path=source_path,
                backup_path=backup_path,
                target_path=final_path,
                success=True,
            )
            self.session_operations.append(operation)

            # Backup cleanup is now handled only at session end for ON_SUCCESS strategy
            LOG.debug("Atomically replaced %s -> %s", source_path, final_path)
        except (OSError, PermissionError, shutil.Error) as e:
            # Rollback on failure
            if backup_path and backup_path.exists():
                try:
                    if not source_path.exists():
                        shutil.move(backup_path, source_path)
                        LOG.info("Restored backup after failed replacement: %s", source_path)
                except (OSError, PermissionError, shutil.Error):
                    LOG.exception("Failed to restore backup")

            operation = FileOperation(
                operation_type="file_replace",
                source_path=source_path,
                backup_path=backup_path,
                success=False,
            )
            self.session_operations.append(operation)

            msg = f"Atomic file replacement failed: {e}"
            raise ProcessingError(msg, file_path=source_path, cause=e) from e
        else:
            return operation

    def _cleanup_backup(self, backup_path: Path) -> None:
        """Remove a backup file."""
        try:
            if backup_path.exists():
                backup_path.unlink()
                self.session_backups.discard(backup_path)
                LOG.debug("Cleaned up backup: %s", backup_path)
        except (OSError, PermissionError) as e:
            LOG.warning("Failed to cleanup backup %s: %s", backup_path, e)

    def cleanup_session_backups(self) -> int:
        """Clean up all backups from this session based on strategy."""
        if self.backup_strategy != BackupStrategy.ON_SUCCESS:
            return 0

        cleaned_count = 0
        successful_operations = [op for op in self.session_operations if op.success and op.backup_path]

        for operation in successful_operations:
            if operation.backup_path and operation.backup_path in self.session_backups:
                self._cleanup_backup(operation.backup_path)
                cleaned_count += 1

        LOG.info("Session cleanup: removed %d backup files", cleaned_count)
        return cleaned_count

    def get_session_summary(self) -> dict[str, Any]:
        """Get summary of file operations in this session."""
        successful_ops = [op for op in self.session_operations if op.success]
        failed_ops = [op for op in self.session_operations if not op.success]

        return {
            "total_operations": len(self.session_operations),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "backups_created": len(self.session_backups),
            "backup_strategy": self.backup_strategy.value,
            "operations": self.session_operations,
        }
