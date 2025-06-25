#!/usr/bin/env python3
"""Simple test script for duplicate file detection."""

import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.duplicate_finder import DuplicateFinder


def create_test_files():
    """Create some test files for duplicate detection."""
    test_dir = Path(tempfile.mkdtemp(prefix="duplicate_test_"))
    print(f"Creating test files in: {test_dir}")
    
    # Create some identical files
    content1 = b"This is test file content number one."
    content2 = b"This is test file content number two - different from one."
    content3 = b"This is test file content number one."  # Same as content1
    
    # Create subdirectories
    (test_dir / "subdir1").mkdir()
    (test_dir / "subdir2").mkdir()
    
    # Create files
    files = [
        (test_dir / "file1.txt", content1),
        (test_dir / "file2.txt", content2),
        (test_dir / "file3.txt", content3),  # Duplicate of file1
        (test_dir / "subdir1" / "file4.txt", content1),  # Another duplicate
        (test_dir / "subdir2" / "file5.txt", content2),  # Duplicate of file2
        (test_dir / "unique.txt", b"This is unique content."),
    ]
    
    for file_path, content in files:
        file_path.write_bytes(content)
        print(f"Created: {file_path}")
    
    return test_dir


def test_duplicate_detection():
    """Test the duplicate detection functionality."""
    print("=== Testing Duplicate Detection ===\n")
    
    # Create test files
    test_dir = create_test_files()
    
    try:
        # Initialize duplicate finder
        finder = DuplicateFinder(max_workers=2)
        
        # Find duplicates
        print("\nSearching for duplicates...")
        duplicates = finder.find_duplicates([test_dir])
        
        # Display results
        print(f"\nFound {len(duplicates)} duplicate groups:")
        
        for i, (hash_val, file_paths) in enumerate(duplicates.items(), 1):
            print(f"\nGroup {i} (Hash: {hash_val[:16]}...):")
            for path in file_paths:
                print(f"  {path}")
        
        # Get summary
        summary = finder.get_duplicate_summary(duplicates)
        print(f"\nSummary:")
        print(f"  Total groups: {summary['total_groups']}")
        print(f"  Total files: {summary['total_files']}")
        print(f"  Wasted space: {summary['wasted_space']} bytes")
        
        print("\n=== Test completed successfully! ===")
        
    finally:
        # Cleanup test files
        import shutil
        print(f"\nCleaning up test directory: {test_dir}")
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    test_duplicate_detection()
