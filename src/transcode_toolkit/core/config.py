"""Enhanced configuration management."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Import the existing config system and extend it
from ..config import MediaToolkitConfig
from ..config import get_config as _get_global_config

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
        """
        Initialize enhanced configuration manager.

        Args:
            config_path: Optional path to config file

        """
        self._config = MediaToolkitConfig.load_from_file(config_path) if config_path else _get_global_config()
        self._overrides: dict[str, Any] = {}
        self._context_stack: list[dict[str, Any]] = []

        # Load raw config for dynamic preset building
        self._raw_config = self._load_raw_config(config_path)

    def _load_raw_config(self, config_path: Path | None = None) -> dict[str, Any]:
        """Load raw YAML config data."""
        if config_path is None:
            # Use default config path
            config_path = Path("config.yaml")

        try:
            with config_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            LOG.warning("Failed to load raw config from %s: %s", config_path, e)
            return {}

    @property
    def config(self) -> MediaToolkitConfig:
        """Get the base configuration."""
        return self._config

    @property
    def raw_config(self) -> dict[str, Any]:
        """Get the raw YAML configuration data."""
        return self._raw_config

    def get_value(self, key_path: str, default: object = None) -> object:
        """Get configuration value with override support."""
        # Check overrides first
        if key_path in self._overrides:
            return self._overrides[key_path]

        # Try to get from underlying config
        try:
            value = self._config
            parts = key_path.split(".")
            for part in parts:
                value = getattr(value, part)
        except AttributeError:
            return default
        else:
            return value

    def set_override(self, key_path: str, value: object) -> None:
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
        """
        Initialize configuration context.

        Args:
            config_manager: Configuration manager instance
            overrides: Configuration overrides to apply

        """
        self.config_manager = config_manager
        self.overrides = overrides

    def __enter__(self) -> ConfigManager:
        """Enter the configuration context."""
        self.config_manager.push_context(self.overrides)
        return self.config_manager

    def __exit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
        """Exit the configuration context."""
        self.config_manager.pop_context()


def with_config_overrides(config_manager: ConfigManager, **overrides: object) -> ConfigContext:
    """Create a context with configuration overrides."""
    return ConfigContext(config_manager, overrides)
