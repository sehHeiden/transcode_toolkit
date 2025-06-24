"""Enhanced configuration management."""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Generic
from pydantic import BaseModel, Field, ValidationError

# Import the existing config system and extend it
from config import MediaToolkitConfig, get_config as _get_global_config

LOG = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ProcessingOptions:
    """Processing options that can override configuration."""
    create_backups: Optional[bool] = None
    cleanup_backups: Optional[str] = None
    workers: Optional[int] = None
    timeout: Optional[int] = None
    dry_run: bool = False
    verbose: bool = False


class ConfigManager:
    """Enhanced configuration manager with context support."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self._config = MediaToolkitConfig.load_from_file(config_path) if config_path else _get_global_config()
        self._overrides: Dict[str, Any] = {}
        self._context_stack: list[Dict[str, Any]] = []
    
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
        parts = key_path.split('.')
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
    
    def clear_overrides(self) -> None:
        """Clear all configuration overrides."""
        self._overrides.clear()
    
    def push_context(self, overrides: Dict[str, Any]) -> None:
        """Push a new configuration context."""
        self._context_stack.append(self._overrides.copy())
        self._overrides.update(overrides)
    
    def pop_context(self) -> None:
        """Pop the current configuration context."""
        if self._context_stack:
            self._overrides = self._context_stack.pop()
    
    def apply_processing_options(self, options: ProcessingOptions) -> None:
        """Apply processing options as configuration overrides."""
        overrides = {}
        
        if options.create_backups is not None:
            overrides['global_.create_backups'] = options.create_backups
        if options.cleanup_backups is not None:
            overrides['global_.cleanup_backups'] = options.cleanup_backups
        if options.workers is not None:
            overrides['global_.default_workers'] = options.workers
        
        for key, value in overrides.items():
            self.set_override(key, value)


class ConfigContext:
    """Context manager for temporary configuration changes."""
    
    def __init__(self, config_manager: ConfigManager, overrides: Dict[str, Any]):
        self.config_manager = config_manager
        self.overrides = overrides
    
    def __enter__(self) -> ConfigManager:
        self.config_manager.push_context(self.overrides)
        return self.config_manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.config_manager.pop_context()


# Extend the existing config system
def get_config_manager(config_path: Optional[Path] = None) -> ConfigManager:
    """Get a configuration manager instance."""
    return ConfigManager(config_path)


def with_config_overrides(config_manager: ConfigManager, **overrides) -> ConfigContext:
    """Create a context with configuration overrides."""
    return ConfigContext(config_manager, overrides)
