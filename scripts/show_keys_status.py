
#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gemini_config import API_KEYS, MODELS

def main():
    print(f"=== Gemini Configuration Status ===")
    print(f"API Keys loaded: {len(API_KEYS)}")
    print(f"Models configured: {len(MODELS)}")
    print(f"Models: {MODELS}")
    
    # Check if keys look valid (basic format check)
    valid_keys = 0
    for i, key in enumerate(API_KEYS):
        if key and len(key) > 30 and key.startswith("AIza"):
            valid_keys += 1
        else:
            print(f"Warning: Key {i} appears invalid")
    
    print(f"Valid-looking keys: {valid_keys}/{len(API_KEYS)}")

if __name__ == "__main__":
    main()
