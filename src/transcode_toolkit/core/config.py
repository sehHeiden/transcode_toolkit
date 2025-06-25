"""Enhanced configuration management."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# Import the existing config system and extend it
from transcode_toolkit.config import MediaToolkitConfig
from transcode_toolkit.config import get_config as _get_global_config

if TYPE_CHECKING:
    from pathlib import Path

LOG = logging.getLogger(__name__)


@dataclass
class ProcessingOptions:
    """Processing options that can override configuration."""

    create_backups: bool | None = None
    cleanup_backups: str | None = None
    workers: int | None = None
    timeout: int | None = None
    dry_run: bool = False
    verbose: bool = False


class ConfigManager:
    """Enhanced configuration manager with context support."""

    def __init__(self, config_path: Path | None = None) -> None:
        self._config = MediaToolkitConfig.load_from_file(config_path) if config_path else _get_global_config()
        self._overrides: dict[str, Any] = {}
        self._context_stack: list[dict[str, Any]] = []

    @property
    def config(self) -> MediaToolkitConfig:
        """Get the base configuration."""
        return self._config

    def get_value(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value with override support."""
        # Check overrides first
        if key_path in self._overrides:
            return self._overrides[key_path]

        # Navigate through config using dot notation
        parts = key_path.split(".")
        value = self._config

        try:
            for part in parts:
                value = getattr(value, part)
            return value
        except AttributeError:
            return default

    def set_override(self, key_path: str, value: Any) -> None:
        """Set a temporary configuration override."""
        self._overrides[key_path] = value

    def push_context(self, overrides: dict[str, Any]) -> None:
        """Push a new configuration context."""
        self._context_stack.append(self._overrides.copy())
        self._overrides.update(overrides)

    def pop_context(self) -> None:
        """Pop the current configuration context."""
        if self._context_stack:
            self._overrides = self._context_stack.pop()

    def apply_processing_options(self, options: ProcessingOptions) -> None:
        """Apply processing options as configuration overrides."""
        overrides: dict[str, Any] = {}

        if options.create_backups is not None:
            overrides["global_.create_backups"] = options.create_backups
        if options.cleanup_backups is not None:
            overrides["global_.cleanup_backups"] = options.cleanup_backups
        if options.workers is not None:
            overrides["global_.default_workers"] = options.workers

        for key, value in overrides.items():
            self.set_override(key, value)


class ConfigContext:
    """Context manager for temporary configuration changes."""

    def __init__(self, config_manager: ConfigManager, overrides: dict[str, Any]) -> None:
        self.config_manager = config_manager
        self.overrides = overrides

    def __enter__(self) -> ConfigManager:
        self.config_manager.push_context(self.overrides)
        return self.config_manager

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.config_manager.pop_context()
        return False


def with_config_overrides(config_manager: ConfigManager, **overrides) -> ConfigContext:
    """Create a context with configuration overrides."""
    return ConfigContext(config_manager, overrides)
