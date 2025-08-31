
#!/usr/bin/env python3
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List

def validate_smoke_test(lang: str, count: int) -> Dict:
    """Validate smoke test results"""
    
    # Check for smoke test output file
    output_files = list(Path(f"data/generation_stage_B/{lang}").glob(f"smoke_test_{lang}_*.jsonl"))
    if not output_files:
        return {"success": False, "error": "No smoke test output files found"}
        
    latest_file = max(output_files, key=lambda x: x.stat().st_mtime)
    
    try:
        examples = []
        with open(latest_file, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line.strip()))
                
        # Compute metrics
        total = len(examples)
        true_count = sum(1 for ex in examples if ex.get("verdict") == "True")
        false_count = sum(1 for ex in examples if ex.get("verdict") == "False")
        fabrication_count = sum(1 for ex in examples if ex.get("suspected_fabrication") == True)
        
        # Schema validation
        required_fields = [
            "id", "language", "claim", "context_chunk_id", "context_excerpt",
            "verdict", "explanation", "reference", "suspected_fabrication",
            "generator_model", "meta"
        ]
        
        valid_count = 0
        for ex in examples:
            if all(field in ex for field in required_fields):
                valid_count += 1
                
        fabrication_rate = fabrication_count / total if total > 0 else 0.0
        schema_valid_rate = valid_count / total if total > 0 else 0.0
        
        # Success criteria
        success = (
            total >= count * 0.8 and 
            schema_valid_rate >= 0.9 and 
            fabrication_rate <= 0.05
        )
        
        return {
            "success": success,
            "total_examples": total,
            "true_count": true_count,
            "false_count": false_count,
            "fabrication_count": fabrication_count,
            "fabrication_rate": fabrication_rate,
            "schema_valid_rate": schema_valid_rate,
            "output_file": str(latest_file),
            "sample_examples": examples[:3]
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Validate smoke test results")
    parser.add_argument("--lang", choices=["ar", "en"], required=True, help="Language")
    parser.add_argument("--count", type=int, default=20, help="Expected count")
    
    args = parser.parse_args()
    
    results = validate_smoke_test(args.lang, args.count)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    if not results.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    main()
