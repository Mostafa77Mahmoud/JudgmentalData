
#!/usr/bin/env python3
import os
import glob
from pathlib import Path

def cleanup_errors():
    """Clean up error files and reset for fresh testing"""
    
    # Clean up raw error files
    raw_dir = Path("raw")
    if raw_dir.exists():
        error_files = list(raw_dir.glob("*error*")) + list(raw_dir.glob("*MAX_TOKENS*"))
        for file in error_files:
            print(f"Removing {file}")
            file.unlink()
    
    # Clean up manual review files
    manual_dir = Path("manual_review")
    if manual_dir.exists():
        for file in manual_dir.glob("*.json"):
            print(f"Removing {file}")
            file.unlink()
    
    # Clear invalid excerpts
    invalid_file = Path("data/invalid_excerpts.jsonl")
    if invalid_file.exists():
        invalid_file.write_text("")
        print("Cleared invalid_excerpts.jsonl")
    
    print("Cleanup completed!")

if __name__ == "__main__":
    cleanup_errors()
