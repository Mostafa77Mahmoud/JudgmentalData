
#!/usr/bin/env python3
"""Analyze raw API responses to debug truncation issues"""

import glob
import json
import os

def analyze_raw_responses():
    """Analyze all raw response files"""
    raw_files = glob.glob("raw/*resp.txt") + glob.glob("raw/*.txt")
    
    if not raw_files:
        print("No raw response files found in raw/ directory")
        return
    
    print(f"Found {len(raw_files)} raw response files\n")
    
    for i, filepath in enumerate(raw_files[-5:]):  # Last 5 files
        print(f"=== File {i+1}: {os.path.basename(filepath)} ===")
        
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
            
            # Try to decode as UTF-8
            try:
                text_content = content.decode('utf-8')
            except UnicodeDecodeError:
                text_content = content.decode('utf-8', errors='replace')
            
            print(f"Size: {len(content)} bytes")
            print(f"First 1000 chars:\n{text_content[:1000]}")
            
            # Try to parse as JSON
            try:
                data = json.loads(text_content)
                print(f"JSON keys: {list(data.keys())}")
                
                # Check for finish_reason
                if "candidates" in data:
                    for j, candidate in enumerate(data["candidates"]):
                        finish_reason = candidate.get("finishReason")
                        if finish_reason:
                            print(f"Candidate {j} finish_reason: {finish_reason}")
                
            except json.JSONDecodeError:
                print("Not valid JSON")
            
            print("-" * 50)
        
        except Exception as e:
            print(f"Error reading file: {e}")
            print("-" * 50)

if __name__ == "__main__":
    analyze_raw_responses()
