#!/usr/bin/env python3
"""
Smoke test validation script for AAOIFI dataset generation
"""
import argparse
import json
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_generator import DatasetGenerator

def main():
    parser = argparse.ArgumentParser(description="Run AAOIFI smoke test")
    parser.add_argument("--lang", choices=["ar", "en"], required=True, help="Language to test")
    parser.add_argument("--count", type=int, default=20, help="Number of examples to generate")

    args = parser.parse_args()

    # Check for API keys
    api_keys = [
        os.getenv("GEMINI_KEY_1"),
        os.getenv("GEMINI_KEY_2"),
        os.getenv("GEMINI_KEY_3"),
        os.getenv("GEMINI_KEY_4")
    ]

    valid_keys = [key for key in api_keys if key]
    if not valid_keys:
        print("ERROR: No valid API keys found. Set GEMINI_KEY_1 through GEMINI_KEY_4 in Secrets.")
        return 1

    print(f"Found {len(valid_keys)} valid API keys")

    try:
        # Initialize generator
        generator = DatasetGenerator(valid_keys)

        # Run smoke test
        results = generator.run_smoke_test(args.lang, args.count)

        # Print results as JSON
        print("\n" + "="*50)
        print("SMOKE TEST RESULTS")
        print("="*50)
        print(json.dumps(results, indent=2, ensure_ascii=False))

        # Print summary
        if results["success"]:
            print(f"\n✅ SMOKE TEST PASSED for {args.lang}")
            stats = results["stats"]
            print(f"Valid rate: {stats['valid_rate']:.1%}")
            print(f"Fabrication rate: {stats['fabrication_rate']:.1%}")
            print(f"Generated: {stats['valid_count']} valid examples")
            return 0
        else:
            print(f"\n❌ SMOKE TEST FAILED for {args.lang}")
            print(f"Error: {results.get('error', 'Unknown error')}")

            if "failure_reasons" in results:
                print("\nTop failure reasons:")
                for reason, count in list(results["failure_reasons"].items())[:5]:
                    print(f"  {reason}: {count}")

            return 1

    except Exception as e:
        print(f"EXCEPTION during smoke test: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())