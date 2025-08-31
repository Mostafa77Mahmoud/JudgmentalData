
#!/usr/bin/env python3
"""
Manual Review Summary Script
Displays candidates that need human review due to fabrication or parsing failures.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def load_manual_review_items():
    """Load all manual review items"""
    review_dir = Path("manual_review")
    if not review_dir.exists():
        return []
    
    items = []
    for review_file in review_dir.glob("*.json"):
        try:
            with open(review_file, 'r', encoding='utf-8') as f:
                item = json.load(f)
                item['review_file'] = str(review_file)
                items.append(item)
        except Exception as e:
            print(f"Error loading {review_file}: {e}")
    
    return items

def print_summary(items):
    """Print human-readable summary"""
    if not items:
        print("No items in manual review queue.")
        return
    
    print(f"\n=== MANUAL REVIEW QUEUE ({len(items)} items) ===")
    print()
    
    # Group by error type
    by_error = {}
    for item in items:
        error_type = item.get("error", "Unknown")[:50]
        if error_type not in by_error:
            by_error[error_type] = []
        by_error[error_type].append(item)
    
    for error_type, error_items in by_error.items():
        print(f"Error Type: {error_type} ({len(error_items)} items)")
        print("-" * 60)
        
        for item in error_items[:3]:  # Show first 3 of each type
            print(f"ID: {item.get('candidate_id', 'Unknown')}")
            print(f"Claim: {item.get('claim', '')[:100]}...")
            print(f"Context: {item.get('chunk_excerpt', '')[:100]}...")
            print(f"Model: {item.get('model_used', 'Unknown')} (Key {item.get('key_index', '?')})")
            print(f"Raw File: {item.get('raw_response_path', 'None')}")
            print(f"Timestamp: {datetime.fromtimestamp(item.get('timestamp', 0))}")
            print()
        
        if len(error_items) > 3:
            print(f"... and {len(error_items) - 3} more items of this type")
        print()

def main():
    items = load_manual_review_items()
    print_summary(items)
    
    # Save summary to file
    summary_file = "manual_review/summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_items": len(items),
            "by_error_type": {
                error: len([i for i in items if error in i.get("error", "")])
                for error in set(i.get("error", "")[:50] for i in items)
            },
            "generated_at": datetime.now().isoformat(),
            "items": items
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    main()
