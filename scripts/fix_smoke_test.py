
#!/usr/bin/env python3
"""
Script to analyze smoke test failures and provide corrected verification
"""
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gemini_client import batch_verify
from src.gemini_config import MAX_FABRICATION_RATE

def analyze_smoke_test_file(file_path: str):
    """Analyze a smoke test file and report issues"""
    print(f"\n=== Analyzing {file_path} ===")
    
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return
    
    # Load and analyze the data
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    
    print(f"Total items: {len(items)}")
    
    # Check fabrication rate
    fabricated = sum(1 for item in items if item.get('suspected_fabrication', False))
    fabrication_rate = fabricated / len(items) if items else 0
    
    print(f"Fabricated items: {fabricated}")
    print(f"Fabrication rate: {fabrication_rate:.2%}")
    print(f"Max allowed rate: {MAX_FABRICATION_RATE:.1%}")
    print(f"Test status: {'❌ FAILED' if fabrication_rate > MAX_FABRICATION_RATE else '✅ PASSED'}")
    
    # Show verification errors
    verification_errors = [item for item in items if 'Verification failed' in item.get('explanation', '')]
    if verification_errors:
        print(f"\nVerification errors: {len(verification_errors)}")
        for item in verification_errors[:3]:  # Show first 3
            print(f"  - {item['id']}: {item['explanation']}")
    
    # Show sample claims
    print("\nSample claims:")
    for i, item in enumerate(items[:3]):
        print(f"  {i+1}. {item['claim'][:100]}...")
        print(f"     Verdict: {item['verdict']} | Fabricated: {item.get('suspected_fabrication', 'unknown')}")

def re_verify_failed_items(file_path: str, output_path: str = None):
    """Re-verify failed items with better prompting"""
    if not output_path:
        output_path = file_path.replace('.jsonl', '_reverified.jsonl')
    
    print(f"\n=== Re-verifying items from {file_path} ===")
    
    # Load items
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    
    # Prepare items for verification (only the fields needed)
    verify_items = []
    for item in items:
        verify_item = {
            "id": item["id"],
            "language": item["language"], 
            "claim": item["claim"],
            "context_chunk_id": item["context_chunk_id"],
            "context_excerpt": item["context_excerpt"]
        }
        verify_items.append(verify_item)
    
    print(f"Re-verifying {len(verify_items)} items...")
    
    # Batch verify in smaller groups to avoid quota issues
    batch_size = 2  # Smaller batches
    results = []
    
    for i in range(0, len(verify_items), batch_size):
        batch = verify_items[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(verify_items) + batch_size - 1)//batch_size}")
        
        try:
            batch_results = batch_verify(batch)
            results.extend(batch_results)
        except Exception as e:
            print(f"Batch failed: {e}")
            # Add failed results
            for item in batch:
                failed_result = {
                    "id": item["id"],
                    "language": item["language"],
                    "claim": item["claim"],
                    "context_chunk_id": item["context_chunk_id"],
                    "context_excerpt": item["context_excerpt"],
                    "verdict": "False",
                    "explanation": f"Re-verification failed: {str(e)[:50]}",
                    "reference": "UNKNOWN",
                    "suspected_fabrication": True,
                    "generator_model": "local",
                    "raw_response_path": "",
                    "meta": {"confidence": 0.0, "re_verification_failed": True}
                }
                results.append(failed_result)
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Report new fabrication rate
    fabricated = sum(1 for item in results if item.get('suspected_fabrication', False))
    fabrication_rate = fabricated / len(results) if results else 0
    
    print(f"\n=== Re-verification Results ===")
    print(f"Total items: {len(results)}")
    print(f"Fabricated items: {fabricated}")
    print(f"New fabrication rate: {fabrication_rate:.2%}")
    print(f"Test status: {'❌ STILL FAILED' if fabrication_rate > MAX_FABRICATION_RATE else '✅ NOW PASSED'}")
    print(f"Results saved to: {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/fix_smoke_test.py <command> [file_path]")
        print("Commands:")
        print("  analyze <file_path>    - Analyze smoke test file")
        print("  re-verify <file_path>  - Re-verify failed items")
        print("  check-all             - Check all smoke test files")
        return
    
    command = sys.argv[1]
    
    if command == "analyze" and len(sys.argv) >= 3:
        analyze_smoke_test_file(sys.argv[2])
    elif command == "re-verify" and len(sys.argv) >= 3:
        re_verify_failed_items(sys.argv[2])
    elif command == "check-all":
        smoke_files = list(Path("data/generation_stage_B").rglob("smoke_test_*.jsonl"))
        for file_path in smoke_files:
            analyze_smoke_test_file(str(file_path))
    else:
        print("Invalid command or missing file path")

if __name__ == "__main__":
    main()
