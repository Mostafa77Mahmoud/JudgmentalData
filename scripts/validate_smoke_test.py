#!/usr/bin/env python3
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
#!/usr/bin/env python3
import logging
import time
from src.dataset_generator import DatasetGenerator
from dataset_generator import generate_candidates_from_seeds
from src.data_processor import DataProcessor
from src.gemini_client import batch_verify
from src.gemini_config import CONTEXT_MAX_CHARS
from src.parse_utils import validate_example_schema


def validate_smoke_test(lang: str, count: int) -> Dict:
    """Validate smoke test results"""

    # Check for smoke test output file
    output_files = list(
        Path(f"data/generation_stage_B/{lang}").glob(
            f"smoke_test_{lang}_*.jsonl"))
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
        fabrication_count = sum(1 for ex in examples
                                if ex.get("suspected_fabrication"))

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
        success = (total >= count * 0.8 and schema_valid_rate >= 0.9
                   and fabrication_rate <= 0.05)

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


def run_validator():
    parser = argparse.ArgumentParser(description="Validate smoke test results")
    parser.add_argument("--lang",
                        choices=["ar", "en"],
                        required=True,
                        help="Language")
    parser.add_argument("--count", type=int, default=20, help="Expected count")

    args = parser.parse_args()

    results = validate_smoke_test(args.lang, args.count)
    print(json.dumps(results, indent=2, ensure_ascii=False))

    if not results.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    run_validator()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


def ensure_chunks_exist(logger) -> bool:
    """Ensure arabic_chunks.json exists, create from cleaned text if needed"""
    chunks_file = Path("inputs/arabic_chunks.json")
    cleaned_file = Path("inputs/arabic_cleaned.txt")

    if chunks_file.exists():
        logger.info(f"Using existing chunks file: {chunks_file}")
        return True

    if not cleaned_file.exists():
        logger.error("Neither chunks nor cleaned text file exists")
        return False

    logger.info(f"Creating chunks from {cleaned_file}")

    try:
        with open(cleaned_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split into logical paragraphs/chunks of ~1000 chars
        chunks = []
        chunk_size = 1000

        # Try to split at sentence boundaries
        sentences = text.split('.')
        current_chunk = ""
        chunk_id = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + "."
            else:
                if current_chunk.strip():
                    chunks.append({
                        "id": chunk_id,
                        "text": current_chunk.strip(),
                        "word_count": len(current_chunk.split()),
                        "language": "arabic"
                    })
                    chunk_id += 1
                current_chunk = sentence + "."

        # Add the last chunk
        if current_chunk.strip():
            chunks.append({
                "id": chunk_id,
                "text": current_chunk.strip(),
                "word_count": len(current_chunk.split()),
                "language": "arabic"
            })

        # Save chunks
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        logger.info(f"Created {len(chunks)} chunks and saved to {chunks_file}")
        return True

    except Exception as e:
        logger.error(f"Failed to create chunks: {e}")
        return False


def load_seeds(language: str, count: int, logger) -> List[Dict]:
    """Load QA seeds for generation"""
    if language == "ar":
        qa_files = [
            "inputs/arabic_qa_pairs (2000).json", "inputs/arabic_qa_pairs.json"
        ]
    else:
        qa_files = [
            "inputs/english_qa_pairs (2000).json",
            "inputs/english_qa_pairs.json"
        ]

    for qa_file in qa_files:
        if Path(qa_file).exists():
            with open(qa_file, 'r', encoding='utf-8') as f:
                seeds = json.load(f)
            logger.info(f"Loaded {len(seeds)} seeds from {qa_file}")
            return seeds[:count]

    logger.error(f"No QA pairs file found for language {language}")
    return []


def run_smoke_test(language: str, count: int, logger) -> Dict:
    """Run smoke test with specified parameters"""
    logger.info(f"Starting smoke test for {language} with {count} examples")

    # Ensure chunks exist
    if not ensure_chunks_exist(logger):
        return {"success": False, "error": "Failed to load/create chunks"}

    # Load seeds
    seeds = load_seeds(language, count, logger)
    if not seeds:
        return {"success": False, "error": "No seeds loaded"}

    # Initialize generator
    try:
        generator = DatasetGenerator()
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        return {"success": False, "error": f"Generator init failed: {e}"}

    # Generate candidates from seeds (using public method)
    try:
        candidates = generator.generate_candidates_from_seeds(seeds, language)
        logger.info(f"Generated {len(candidates)} candidates")
    except Exception as e:
        logger.error(f"Failed to generate candidates: {e}")
        return {"success": False, "error": f"Candidate generation failed: {e}"}

    # Skip local pre-verification → go straight to model verification
    model_verified = []
    try:
        model_verified = generator.batch_verify_with_model(
            candidates, language)
        logger.info(f"Model verified {len(model_verified)} examples")
    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        return {"success": False, "error": f"Model verification failed: {e}"}

    # Combine verified examples
    all_examples = model_verified

    # Filter valid examples
    valid_examples = []
    failed_parses = 0
    for ex in all_examples:
        is_valid, reason = validate_example_schema(ex,
                                                   generator.required_fields)
        if is_valid:
            valid_examples.append(ex)
        else:
            failed_parses += 1
            logger.warning(f"Invalid example: {reason}")

    # Compute metrics
    stats = generator._compute_stats(valid_examples)

    # Save results
    output_file = f"data/generation_stage_B/{language}/smoke_test_{language}_{len(valid_examples)}.jsonl"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for example in valid_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    # Write summary log
    summary = {
        "timestamp": time.time(),
        "language": language,
        "target_count": count,
        "generated_count": len(valid_examples),
        "model_verified": len(model_verified),
        "failed_parses_count": failed_parses,
        "fabrication_rate": stats["fabrication_rate"],
        "output_file": output_file,
        "stats": stats
    }

    with open("logs/smoke_test_summary.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    logger.info(
        f"Smoke test completed: {len(valid_examples)} examples, fabrication rate: {stats['fabrication_rate']:.2%}"
    )

    return {
        "success": len(valid_examples) >= count * 0.8,
        "generated_count": len(valid_examples),
        "model_verified": len(model_verified),
        "failed_parses_count": failed_parses,
        "fabrication_rate": stats["fabrication_rate"],
        "output_file": output_file,
        "stats": stats
    }


def main():
    parser = argparse.ArgumentParser(description="Validate smoke test")
    parser.add_argument("--lang",
                        choices=["ar", "en"],
                        default="ar",
                        help="Language")
    parser.add_argument("--count",
                        type=int,
                        default=15,
                        help="Number of examples")
    parser.add_argument("--run",
                        action="store_true",
                        help="Run full smoke test")

    args = parser.parse_args()
    logger = setup_logging()

    if args.run:
        result = run_smoke_test(args.lang, args.count, logger)

        # Print results
        print(f"\n=== SMOKE TEST RESULTS ({args.lang.upper()}) ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Exit with appropriate code
        if result["success"]:
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        print("Use --run to execute smoke test")


if __name__ == "__main__":
    main()
