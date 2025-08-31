import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.dataset_generator import DatasetGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate AAOIFI judgmental dataset using Gemini models only")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test")
    parser.add_argument("--full", action="store_true", help="Run full generation")
    parser.add_argument("--lang", choices=["ar", "en"], required=True, help="Language")
    parser.add_argument("--target", type=int, default=2000, help="Target examples for full generation")
    parser.add_argument("--count", type=int, default=15, help="Examples for smoke test")

    args = parser.parse_args()

    try:
        print("Initializing Gemini-only dataset generator...")
        generator = DatasetGenerator()

        if args.smoke:
            print(f"Running smoke test for {args.lang} with {args.count} examples...")
            results = generator.run_smoke_test(args.lang, args.count)
            print(json.dumps(results, indent=2, ensure_ascii=False))
        elif args.full:
            print(f"Running full generation for {args.lang} with target {args.target} examples...")
            results = generator.generate_full_dataset(args.lang, args.target)
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print("Please specify --smoke or --full")
            parser.print_help()

    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()