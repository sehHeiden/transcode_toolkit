"""Enhanced file and backup management."""

from __future__ import annotations
import logging
import shutil
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from .base import ProcessingError

LOG = logging.getLogger(__name__)


class BackupStrategy(Enum):
    """Backup creation strategies."""

    NEVER = "never"
    IMMEDIATE = "immediate"
    SESSION_END = "session_end"
    ON_SUCCESS = "on_success"


@dataclass
class FileOperation:
    """Represents a file operation that can be rolled back."""

    operation_type: str
    source_path: Path
    backup_path: Optional[Path] = None
    target_path: Optional[Path] = None
    success: bool = False
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class FileManager:
    """Enhanced file manager with atomic operations and backup management."""

    def __init__(
        self,
        backup_strategy: BackupStrategy = BackupStrategy.ON_SUCCESS,
        retention_days: int = 0,
    ):
        self.backup_strategy = backup_strategy
        self.retention_days = retention_days
        self.session_operations: List[FileOperation] = []
        self.session_backups: Set[Path] = set()

    def create_backup(self, file_path: Path) -> Optional[Path]:
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

            LOG.debug(f"Created backup: {backup_path}")
            return backup_path

        except Exception as e:
            LOG.error(f"Failed to create backup for {file_path}: {e}")
            raise ProcessingError(
                f"Backup creation failed: {e}", file_path=file_path, cause=e
            )

    def atomic_replace(
        self, source_path: Path, temp_path: Path, create_backup: bool = True
    ) -> FileOperation:
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
            final_path = (
                source_path.with_suffix(".opus")
                if source_path.suffix.lower() not in {".opus"}
                else source_path
            )
            shutil.move(temp_path, final_path)

            operation = FileOperation(
                operation_type="file_replace",
                source_path=source_path,
                backup_path=backup_path,
                target_path=final_path,
                success=True,
            )
            self.session_operations.append(operation)

            # Handle immediate backup cleanup
            if backup_path and self.backup_strategy in {
                BackupStrategy.IMMEDIATE,
                BackupStrategy.ON_SUCCESS,
            }:
                self._cleanup_backup(backup_path)

            LOG.debug(f"Atomically replaced {source_path} -> {final_path}")
            return operation

        except Exception as e:
            # Rollback on failure
            if backup_path and backup_path.exists():
                try:
                    if not source_path.exists():
                        shutil.move(backup_path, source_path)
                        LOG.info(
                            f"Restored backup after failed replacement: {source_path}"
                        )
                except Exception as restore_error:
                    LOG.error(f"Failed to restore backup: {restore_error}")

            operation = FileOperation(
                operation_type="file_replace",
                source_path=source_path,
                backup_path=backup_path,
                success=False,
            )
            self.session_operations.append(operation)

            raise ProcessingError(
                f"Atomic file replacement failed: {e}", file_path=source_path, cause=e
            )

    def _cleanup_backup(self, backup_path: Path) -> None:
        """Remove a backup file."""
        try:
            if backup_path.exists():
                backup_path.unlink()
                self.session_backups.discard(backup_path)
                LOG.debug(f"Cleaned up backup: {backup_path}")
        except Exception as e:
            LOG.warning(f"Failed to cleanup backup {backup_path}: {e}")

    def cleanup_session_backups(self) -> int:
        """Clean up all backups from this session based on strategy."""
        if self.backup_strategy not in {
            BackupStrategy.SESSION_END,
            BackupStrategy.ON_SUCCESS,
        }:
            return 0

        cleaned_count = 0
        successful_operations = [
            op for op in self.session_operations if op.success and op.backup_path
        ]

        for operation in successful_operations:
            if operation.backup_path and operation.backup_path in self.session_backups:
                self._cleanup_backup(operation.backup_path)
                cleaned_count += 1

        LOG.info(f"Session cleanup: removed {cleaned_count} backup files")
        return cleaned_count

    def cleanup_old_backups(self, directory: Path) -> int:
        """Clean up old backup files based on retention policy."""
        if self.retention_days <= 0:
            return 0

        retention_seconds = self.retention_days * 24 * 60 * 60
        current_time = time.time()
        cleaned_count = 0

        # Find all .bak files in the directory tree
        backup_files = list(directory.rglob("*.bak"))

        for backup_file in backup_files:
            try:
                file_age = current_time - backup_file.stat().st_mtime
                if file_age > retention_seconds:
                    backup_file.unlink()
                    cleaned_count += 1
                    LOG.info(
                        f"Removed old backup (age: {file_age / 86400:.1f} days): {backup_file}"
                    )
            except Exception as e:
                LOG.warning(f"Failed to check/remove old backup {backup_file}: {e}")

        return cleaned_count

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of file operations in this session."""
        successful_ops = [op for op in self.session_operations if op.success]
        failed_ops = [op for op in self.session_operations if not op.success]

        return {
            "total_operations": len(self.session_operations),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "backups_created": len(self.session_backups),
            "backup_strategy": self.backup_strategy.value,
            "retention_days": self.retention_days,
            "operations": self.session_operations,
        }

    def force_cleanup_all_backups(self, directory: Path) -> int:
        """Force removal of all .bak files in directory (emergency cleanup)."""
        backup_files = list(directory.rglob("*.bak"))
        cleaned_count = 0

        for backup_file in backup_files:
            try:
                backup_file.unlink()
                cleaned_count += 1
                LOG.info(f"Force removed backup: {backup_file}")
            except Exception as e:
                LOG.warning(f"Failed to force remove backup {backup_file}: {e}")

        return cleaned_count
