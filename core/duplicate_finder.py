"""Parallel duplicate file detection based on hash comparison."""

from __future__ import annotations

import hashlib
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

LOG = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Information about a file for duplicate detection."""
    
    path: Path
    size: int
    hash: str | None = None
    
    def __post_init__(self):
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")


class DuplicateFinder:
    """Parallel duplicate file finder using hash-based comparison."""
    
    def __init__(self, max_workers: int = None, chunk_size: int = 8192):
        """Initialize the duplicate finder.
        
        Args:
            max_workers: Maximum number of threads for parallel processing
            chunk_size: Size of chunks to read when calculating hash
        """
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.chunk_size = chunk_size
        self._size_groups: Dict[int, List[FileInfo]] = defaultdict(list)
        self._hash_groups: Dict[str, List[FileInfo]] = defaultdict(list)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        hash_obj = hashlib.sha256()
        
        try:
            with file_path.open('rb') as f:
                while chunk := f.read(self.chunk_size):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            LOG.warning(f"Failed to calculate hash for {file_path}: {e}")
            raise
    
    def _get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes."""
        try:
            return file_path.stat().st_size
        except Exception as e:
            LOG.warning(f"Failed to get size for {file_path}: {e}")
            raise
    
    def _collect_files(self, paths: List[Path], extensions: Set[str] = None) -> List[FileInfo]:
        """Collect all files from given paths with optional extension filtering."""
        files = []
        
        for path in paths:
            if path.is_file():
                if extensions is None or path.suffix.lower() in extensions:
                    try:
                        size = self._get_file_size(path)
                        files.append(FileInfo(path=path, size=size))
                    except Exception as e:
                        LOG.warning(f"Skipping {path}: {e}")
            elif path.is_dir():
                for file_path in path.rglob('*'):
                    if file_path.is_file():
                        if extensions is None or file_path.suffix.lower() in extensions:
                            try:
                                size = self._get_file_size(file_path)
                                files.append(FileInfo(path=file_path, size=size))
                            except Exception as e:
                                LOG.warning(f"Skipping {file_path}: {e}")
        
        return files
    
    def _group_by_size(self, files: List[FileInfo]) -> Dict[int, List[FileInfo]]:
        """Group files by size - first optimization step."""
        size_groups = defaultdict(list)
        
        for file_info in files:
            size_groups[file_info.size].append(file_info)
        
        # Only return groups with more than one file (potential duplicates)
        return {size: files for size, files in size_groups.items() if len(files) > 1}
    
    def _calculate_hashes_parallel(self, files: List[FileInfo]) -> List[FileInfo]:
        """Calculate hashes for files in parallel."""
        LOG.info(f"Calculating hashes for {len(files)} files using {self.max_workers} workers...")
        
        def calculate_hash_wrapper(file_info: FileInfo) -> FileInfo:
            try:
                file_info.hash = self._calculate_file_hash(file_info.path)
                return file_info
            except Exception as e:
                LOG.error(f"Hash calculation failed for {file_info.path}: {e}")
                return file_info  # Return with hash=None
        
        processed_files = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all hash calculation tasks
            future_to_file = {
                executor.submit(calculate_hash_wrapper, file_info): file_info 
                for file_info in files
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                try:
                    result = future.result()
                    if result.hash is not None:
                        processed_files.append(result)
                    
                    # Progress logging
                    if len(processed_files) % max(1, len(files) // 10) == 0:
                        progress = len(processed_files) / len(files) * 100
                        LOG.info(f"Hash calculation progress: {progress:.1f}%")
                        
                except Exception as e:
                    file_info = future_to_file[future]
                    LOG.error(f"Hash calculation error for {file_info.path}: {e}")
        
        LOG.info(f"Successfully calculated hashes for {len(processed_files)} files")
        return processed_files
    
    def _group_by_hash(self, files: List[FileInfo]) -> Dict[str, List[FileInfo]]:
        """Group files by hash - final step to identify duplicates."""
        hash_groups = defaultdict(list)
        
        for file_info in files:
            if file_info.hash:
                hash_groups[file_info.hash].append(file_info)
        
        # Only return groups with more than one file (actual duplicates)
        return {hash_val: files for hash_val, files in hash_groups.items() if len(files) > 1}
    
    def find_duplicates(
        self, 
        paths: List[Path], 
        extensions: Set[str] = None,
        progress_callback=None
    ) -> Dict[str, List[Path]]:
        """Find duplicate files in the given paths.
        
        Args:
            paths: List of paths to search (files or directories)
            extensions: Set of file extensions to include (e.g., {'.mp3', '.flac'})
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary mapping hash to list of duplicate file paths
        """
        LOG.info(f"Starting duplicate search in {len(paths)} paths...")
        
        if progress_callback:
            progress_callback("Collecting files...")
        
        # Step 1: Collect all files
        all_files = self._collect_files(paths, extensions)
        LOG.info(f"Found {len(all_files)} files to analyze")
        
        if len(all_files) < 2:
            LOG.info("Not enough files to find duplicates")
            return {}
        
        if progress_callback:
            progress_callback(f"Analyzing {len(all_files)} files...")
        
        # Step 2: Group by size (quick pre-filter)
        size_groups = self._group_by_size(all_files)
        LOG.info(f"Found {len(size_groups)} size groups with potential duplicates")
        
        if not size_groups:
            LOG.info("No files with matching sizes found")
            return {}
        
        # Step 3: Calculate hashes only for files with matching sizes
        files_to_hash = []
        for size, files in size_groups.items():
            files_to_hash.extend(files)
        
        if progress_callback:
            progress_callback(f"Calculating hashes for {len(files_to_hash)} files...")
        
        hashed_files = self._calculate_hashes_parallel(files_to_hash)
        
        # Step 4: Group by hash to find actual duplicates
        if progress_callback:
            progress_callback("Identifying duplicates...")
        
        hash_groups = self._group_by_hash(hashed_files)
        
        # Convert to result format
        duplicates = {}
        total_duplicate_files = 0
        
        for hash_val, files in hash_groups.items():
            file_paths = [file_info.path for file_info in files]
            duplicates[hash_val] = file_paths
            total_duplicate_files += len(file_paths)
        
        LOG.info(f"Found {len(duplicates)} duplicate groups containing {total_duplicate_files} files")
        
        if progress_callback:
            progress_callback(f"Found {len(duplicates)} duplicate groups")
        
        return duplicates
    
    def get_duplicate_summary(self, duplicates: Dict[str, List[Path]]) -> Dict:
        """Get summary statistics about found duplicates."""
        if not duplicates:
            return {
                'total_groups': 0,
                'total_files': 0,
                'wasted_space': 0,
                'groups': []
            }
        
        total_files = sum(len(files) for files in duplicates.values())
        total_groups = len(duplicates)
        wasted_space = 0
        group_details = []
        
        for hash_val, file_paths in duplicates.items():
            if file_paths:
                try:
                    file_size = file_paths[0].stat().st_size
                    group_wasted = file_size * (len(file_paths) - 1)  # All but one are "wasted"
                    wasted_space += group_wasted
                    
                    group_details.append({
                        'hash': hash_val[:16] + '...',  # Truncated hash for display
                        'count': len(file_paths),
                        'size_each': file_size,
                        'wasted_space': group_wasted,
                        'files': [str(p) for p in file_paths]
                    })
                except Exception as e:
                    LOG.warning(f"Error calculating size for duplicate group: {e}")
        
        # Sort groups by wasted space (descending)
        group_details.sort(key=lambda x: x.get('wasted_space', 0), reverse=True)
        
        return {
            'total_groups': total_groups,
            'total_files': total_files,
            'wasted_space': wasted_space,
            'groups': group_details
        }
