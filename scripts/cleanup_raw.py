
#!/usr/bin/env python3
import os
import time
from pathlib import Path

def cleanup_raw_files(max_files: int = 500):
    """Keep only the most recent raw response files"""
    raw_dir = Path("raw")
    if not raw_dir.exists():
        return
    
    # Get all .resp.txt files with their timestamps
    files = []
    for file_path in raw_dir.glob("*.resp.txt"):
        try:
            # Extract timestamp from filename
            timestamp = int(file_path.stem.split('_')[0])
            files.append((timestamp, file_path))
        except (ValueError, IndexError):
            # If we can't parse timestamp, use file mtime
            files.append((file_path.stat().st_mtime, file_path))
    
    # Sort by timestamp (newest first)
    files.sort(reverse=True)
    
    # Remove excess files
    if len(files) > max_files:
        for _, file_path in files[max_files:]:
            try:
                file_path.unlink()
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")
        
        print(f"Cleaned up {len(files) - max_files} old raw response files")
    else:
        print(f"Only {len(files)} raw files, no cleanup needed")

if __name__ == "__main__":
    cleanup_raw_files()
