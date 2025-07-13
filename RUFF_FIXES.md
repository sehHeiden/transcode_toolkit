# Ruff Fixes Applied

## Major Fixes Completed

### 1. **Fixed S603 subprocess call issues**
- Added `# noqa: S603` annotations for subprocess calls with hardcoded commands
- Fixed in `src/transcode_toolkit/audio/estimate.py`

### 2. **Fixed TRY300 errors**
- Moved return statements to else blocks where appropriate
- Fixed in `src/transcode_toolkit/cli/commands/video.py`

### 3. **Fixed G004 logging errors**
- Replaced f-string logging with % formatting
- Fixed in `src/transcode_toolkit/core/unified_estimate.py`
- Examples:
  - `LOG.info(f"Analyzing {directory}...")` → `LOG.info("Analyzing %s...", directory)`
  - `LOG.warning(f"Failed to analyze {file}: {e}")` → `LOG.warning("Failed to analyze %s: %s", file, e)`

### 4. **Fixed BLE001 blind exception handling**
- Replaced `except Exception:` with specific exception types
- Fixed in `src/transcode_toolkit/config/settings.py`
- Examples:
  - `except Exception as e:` → `except (OSError, yaml.YAMLError) as e:`
  - `except Exception as e:` → `except (TypeError, ValueError) as e:`

### 5. **Fixed function signatures**
- Added keyword-only arguments with `*` separator
- Fixed in `src/transcode_toolkit/core/unified_estimate.py`
- Examples:
  - `def analyze_directory(directory: Path, save_settings: bool = False)` → `def analyze_directory(directory: Path, *, save_settings: bool = False)`

### 6. **Fixed type annotations**
- Added proper type hints where missing
- Fixed return type annotations for functions
- Added type annotations for nested functions

### 7. **Fixed magic numbers**
- Replaced magic numbers with named constants
- Examples:
  - `if video_duration <= 60:` → `if video_duration <= SHORT_VIDEO_DURATION:`
  - `if current_bitrate > 96000:` → `if current_bitrate > HIGH_BITRATE_MONO:`

### 8. **Fixed import issues**
- Fixed `type: ignore` annotations
- Added proper exception chaining with `from`

### 9. **Fixed file operations**
- Replaced `open()` with `Path.open()` for better path handling
- Fixed in `src/transcode_toolkit/core/unified_estimate.py`

## Remaining Issues (Minor)

The remaining ruff errors are mostly minor issues that don't affect functionality:

1. **Type annotations for private functions** (ANN202)
2. **Missing docstrings** (D107, D105)
3. **Function complexity warnings** (C901, PLR0912, PLR0915)
4. **Boolean positional arguments** (FBT001, FBT002)
5. **Some G004 logging issues** in files not yet processed

## Files Modified

1. `src/transcode_toolkit/core/unified_estimate.py` - Major refactoring for logging and types
2. `src/transcode_toolkit/audio/estimate.py` - Fixed subprocess call
3. `src/transcode_toolkit/cli/commands/video.py` - Fixed TRY300 error
4. `src/transcode_toolkit/config/settings.py` - Fixed exception handling

## Next Steps

To complete the ruff fixes, consider:

1. Adding type annotations for remaining private functions
2. Adding missing docstrings
3. Refactoring complex functions into smaller ones
4. Converting boolean positional arguments to keyword-only where appropriate
5. Converting remaining f-string logging to % formatting

The codebase is now significantly more compliant with ruff standards and follows better Python practices.
