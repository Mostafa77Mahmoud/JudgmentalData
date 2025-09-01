import json
import uuid
import time
import random
import re
import argparse
import sys
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from src.gemini_client import GeminiClient, batch_verify, robust_parse_json_array, batch_verify_single
from src.gemini_config import BATCH_SIZE, CONTEXT_MAX_CHARS, MAX_FABRICATION_RATE, MAX_OUTPUT_TOKENS, MAX_INPUT_TOKENS
from src.parse_utils import parse_json_loose, compute_token_overlap, validate_example_schema, find_exact_substring
from src.data_processor import DataProcessor

# Add the project root to Python path if not already there
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set up logger
logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Production-ready dataset generator using only Gemini models"""

    def __init__(self, config_path: str = "config/keys.json"):
        """Initialize the dataset generator with configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self._create_directories()

        self.gemini_client = GeminiClient(config_path)
        self.processor = DataProcessor()
        self._load_data_sources()
        self._load_seeds()

        # Load context max chars from config
        self.context_max_chars = CONTEXT_MAX_CHARS

        self.required_fields = [
            "id", "language", "claim", "context_chunk_id", "context_excerpt",
            "verdict", "explanation", "reference", "suspected_fabrication",
            "generator_model", "meta"
        ]

    def _create_directories(self):
        """Create necessary output directories"""
        directories = [
            "data/generation_stage_B/ar", "data/generation_stage_B/en",
            "output/alpaca", "raw", "logs", "progress", "manual_review"
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _load_data_sources(self):
        """Load data sources with priority: chunks.json > cleaned.txt"""
        try:
            self.processor.load_data()
            self.logger.info("Loaded data sources successfully")
        except Exception as e:
            self.logger.error(f"Failed to load data sources: {e}")
            raise

    def generate_candidates_from_seeds(self, seeds: List[Dict],
                                       language: str) -> List[Dict]:
        """Public wrapper to generate candidates directly from seeds"""
        try:
            # تقدر تبني prompt وتستخدم الموديل زي ما بتعمل في _generate_with_model
            return self._generate_with_model(seeds,
                                             language,
                                             target_count=len(seeds))
        except Exception as e:
            self.logger.error(f"Error generating candidates from seeds: {e}")
            return []

    def batch_verify_with_model(self, candidates: List[Dict],
                                language: str) -> List[Dict]:
        """Public wrapper to verify candidates with model in batch"""
        try:
            return self.batch_verify_with_model(candidates, language)
        except Exception as e:
            self.logger.error(f"Error in batch verification: {e}")
            return []

    def _load_seeds(self):
        """Load QA pairs as generation seeds"""
        try:
            # Try both possible filenames
            ar_files = [
                "inputs/arabic_qa_pairs (2000).json",
                "inputs/arabic_qa_pairs.json"
            ]
            en_files = [
                "inputs/english_qa_pairs (2000).json",
                "inputs/english_qa_pairs.json"
            ]

            self.arabic_seeds = None
            for ar_file in ar_files:
                if Path(ar_file).exists():
                    with open(ar_file, 'r', encoding='utf-8') as f:
                        self.arabic_seeds = json.load(f)
                    break

            self.english_seeds = None
            for en_file in en_files:
                if Path(en_file).exists():
                    with open(en_file, 'r', encoding='utf-8') as f:
                        self.english_seeds = json.load(f)
                    break

            if not self.arabic_seeds or not self.english_seeds:
                raise FileNotFoundError("Could not find QA pairs files")

            self.logger.info(
                f"Loaded {len(self.arabic_seeds)} Arabic seeds, {len(self.english_seeds)} English seeds"
            )
        except Exception as e:
            self.logger.error(f"Failed to load seeds: {e}")
            raise

    def _build_generation_prompt(self,
                                 context_chunks: List[Dict],
                                 language: str,
                                 target_count: int = 8) -> str:
        """Build prompt for generating judgmental examples using context chunks"""

        # Prepare context text from chunks
        context_text = ""
        chunk_refs = []

        for i, chunk in enumerate(context_chunks[:5]):  # Use first 5 chunks
            chunk_text = chunk.get("text", "")[:800]  # Limit each chunk
            context_text += f"\n--- Chunk {i+1} ---\n{chunk_text}\n"
            chunk_refs.append({
                "id": i + 1,
                "original_id": chunk.get("id", f"chunk_{i}")
            })

        if language == "ar":
            # Reduce target count to prevent truncation
            actual_target = min(target_count, 5)
            prompt = f"""
أنت خبير في إنشاء بيانات تدريب للذكاء الاصطناعي. مهمتك إنشاء {actual_target} أمثلة فقط من نوع "judgmental dataset" باللغة العربية.

النص المرجعي:
{context_text}

المطلوب:
1. إنشاء ادعاءات (claims) متنوعة بناءً على النص المرجعي
2. كل ادعاء يجب أن يكون إما صحيح (True) أو خاطئ (False) أو غير واضح (Unknown)
3. تقديم مرجع دقيق من النص المدروس
4. شرح موجز للحكم
5. الحفاظ على توازن 50% صحيح، 30% خاطئ، 20% غير واضح

أعد JSON صالح فقط بدون أي نص إضافي:
[
  {{
    "id": "uuid_string",
    "language": "ar", 
    "claim": "نص الادعاء باللغة العربية",
    "context_chunk_id": رقم_القطعة_المرجعية,
    "context_excerpt": "مقتطف من النص المرجعي ذي الصلة",
    "verdict": "True|False|Unknown",
    "explanation": "شرح موجز للحكم",
    "reference": "نص دقيق من المرجع أو UNKNOWN",
    "suspected_fabrication": false,
    "generator_model": "gemini-2.5-flash",
    "raw_response_path": "",
    "meta": {{"confidence": 0.8, "chunk_source": "context"}}
  }}
]
"""
        else:
            # Reduce target count to prevent truncation
            actual_target = min(target_count, 5)
            prompt = f"""
You are an expert in creating AI training data. Your task is to generate {actual_target} examples for a "judgmental dataset" in English.

Reference text:
{context_text}

Requirements:
1. Create diverse claims based on the reference text
2. Each claim should be either True, False, or Unknown
3. Provide accurate reference from the studied text
4. Brief explanation for the judgment
5. Maintain balance: 50% True, 30% False, 20% Unknown

Return ONLY valid JSON without any additional text:
[
  {{
    "id": "uuid_string",
    "language": "en",
    "claim": "claim text in English",
    "context_chunk_id": chunk_reference_number,
    "context_excerpt": "relevant excerpt from reference text",
    "verdict": "True|False|Unknown", 
    "explanation": "brief explanation of judgment",
    "reference": "exact text from reference or UNKNOWN",
    "suspected_fabrication": false,
    "generator_model": "gemini-2.5-flash",
    "raw_response_path": "",
    "meta": {{"confidence": 0.8, "chunk_source": "context"}}
  }}
]
"""

        return prompt.strip()

    def _generate_with_model(self,
                             chunks: List[Dict],
                             language: str,
                             target_count: int = 20) -> List[Dict]:
        """Generate examples using Gemini model"""

        prompt = self._build_generation_prompt(chunks, language, target_count)

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.logger.info(
                    f"Generating examples with model, attempt {attempt + 1}")

                result = self.gemini_client.call_model(
                    prompt=prompt,
                    model="gemini-2.5-flash",
                    max_tokens=40000,  # Increased to avoid truncation
                    temperature=0.0  # Deterministic output
                )

                if not result["success"]:
                    self.logger.error(
                        f"Model call failed: {result.get('error', 'Unknown error')}"
                    )
                    continue

                # Parse the JSON response
                examples = robust_parse_json_array(result["raw_text"])
                if not examples:
                    self.logger.error(
                        "Failed to parse JSON from model response")
                    continue

                # Post-process examples
                processed_examples = []
                for i, ex in enumerate(examples):
                    if not isinstance(ex, dict):
                        continue

                    # Ensure required fields
                    if "id" not in ex:
                        ex["id"] = str(uuid.uuid4())
                    if "language" not in ex:
                        ex["language"] = language
                    if "generator_model" not in ex:
                        ex["generator_model"] = "gemini-2.5-flash"
                    if "raw_response_path" not in ex:
                        ex["raw_response_path"] = result.get(
                            "raw_response_path", "")
                    if "suspected_fabrication" not in ex:
                        ex["suspected_fabrication"] = False
                    if "meta" not in ex:
                        ex["meta"] = {"confidence": 0.8}

                    # Validate and limit context excerpt
                    if "context_excerpt" in ex and len(
                            ex["context_excerpt"]) > self.context_max_chars:
                        ex["context_excerpt"] = ex[
                            "context_excerpt"][:self.context_max_chars]

                    processed_examples.append(ex)

                self.logger.info(
                    f"Generated {len(processed_examples)} examples")
                return processed_examples

            except Exception as e:
                self.logger.error(
                    f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise
                time.sleep(2**attempt)  # Exponential backoff

        return []

    def batch_verify_examples(self, examples: List[Dict],
                              language: str) -> List[Dict]:
        """Verify examples using Gemini model in batches"""

        if not examples:
            return []

        self.logger.info(f"Verifying {len(examples)} examples with model")

        # Prepare verification items
        verification_items = []
        for ex in examples:
            verification_items.append({
                "id":
                ex["id"],
                "claim":
                ex["claim"],
                "context_excerpt":
                ex.get("context_excerpt", ""),
                "language":
                language,
                "context_chunk_id":
                ex.get("context_chunk_id", 0)
            })

        # Use batch verification
        try:
            verified_results = batch_verify(verification_items)
        except Exception as e:
            self.logger.error(
                f"Batch verification failed, falling back to single verification: {e}"
            )
            verified_results = batch_verify_single(verification_items)

        # Apply verification results back to examples
        result_map = {v["id"]: v for v in verified_results}

        final_examples = []
        for ex in examples:
            ex_id = ex["id"]
            if ex_id in result_map:
                verification = result_map[ex_id]
                ex.update({
                    "verdict":
                    verification.get("verdict", "Unknown"),
                    "explanation":
                    verification.get("explanation", "")[:200],
                    "reference":
                    verification.get("reference", "UNKNOWN"),
                    "suspected_fabrication":
                    verification.get("suspected_fabrication", True),
                    "meta": {
                        **ex.get("meta", {}), "verification_confidence":
                        verification.get("confidence", 0.5)
                    }
                })
            else:
                # Failed verification
                ex.update({
                    "verdict": "Unknown",
                    "explanation": "Verification failed",
                    "reference": "UNKNOWN",
                    "suspected_fabrication": True,
                    "meta": {
                        **ex.get("meta", {}), "verification_failed": True
                    }
                })

            final_examples.append(ex)

        return final_examples

    def _save_progress(self, language: str, examples: List[Dict],
                       seed_index: int):
        """Save generation progress"""
        progress = {
            "language": language,
            "total_examples": len(examples),
            "seed_index": seed_index,
            "timestamp": time.time(),
            "stats": self._compute_stats(examples)
        }

        with open(f"progress/progress_{language}.json", "w",
                  encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    def _compute_stats(self, examples: List[Dict]) -> Dict:
        """Compute dataset statistics"""
        total = len(examples)
        true_count = sum(1 for ex in examples if ex.get("verdict") == "True")
        false_count = sum(1 for ex in examples if ex.get("verdict") == "False")
        unknown_count = sum(1 for ex in examples
                            if ex.get("verdict") == "Unknown")
        fabrication_count = sum(1 for ex in examples
                                if ex.get("suspected_fabrication") is True)

        return {
            "total": total,
            "true": true_count,
            "false": false_count,
            "unknown": unknown_count,
            "fabrications": fabrication_count,
            "fabrication_rate": fabrication_count / total if total > 0 else 0.0
        }

    def _save_smoke_test_summary(self, language: str, stats: Dict,
                                 valid_examples: List[Dict],
                                 failed_raw_paths: List[str]):
        """Save smoke test summary report"""
        summary = {
            "generated_count": stats["total"],
            "verified_model": stats["total"],
            "true_count": stats["true"],
            "false_count": stats["false"],
            "unknown_count": stats.get("unknown", 0),
            "fabrication_rate": stats["fabrication_rate"],
            "failed_candidates": failed_raw_paths,
            "timestamp": time.time(),
            "language": language,
            "max_fabrication_threshold": MAX_FABRICATION_RATE,
            "success": stats["fabrication_rate"] <= MAX_FABRICATION_RATE,
            "method": "gemini_only"
        }

        summary_file = f"data/generation_stage_B/{language}/smoke_test_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary_file

    def run_smoke_test(self, language: str, target_count: int = 15) -> Dict:
        """Run smoke test using only Gemini models"""
        self.logger.info(
            f"Starting smoke test for {language} with {target_count} examples (Gemini only)"
        )

        chunks = self.processor.arabic_chunks if language == "ar" else self.processor.english_chunks

        if not chunks or len(chunks) == 0:
            raise ValueError(
                f"No chunks data available for language {language}")

        # Select random chunks for context
        sample_chunks = random.sample(chunks, min(10, len(chunks)))

        # Generate examples using model
        examples = self._generate_with_model(sample_chunks, language,
                                             target_count)

        if not examples:
            raise RuntimeError("Failed to generate any examples")

        # Verify examples using model
        verified_examples = self.batch_verify_examples(examples, language)

        # Filter valid examples
        valid_examples = []
        failed_raw_paths = []

        for ex in verified_examples:
            is_valid, reason = validate_example_schema(ex,
                                                       self.required_fields)
            if is_valid:
                valid_examples.append(ex)
            else:
                self.logger.warning(f"Invalid example: {reason}")
                if "raw_response_path" in ex:
                    failed_raw_paths.append(ex["raw_response_path"])

        # Apply fabrication post-processing
        from src.fabrication_checker import FabricationChecker
        fab_checker = FabricationChecker(max_fabrication_rate=0.10)
        final_claims = fab_checker.post_process_results(verified_examples)
        quality_report = fab_checker.generate_quality_report(final_claims)

        # Log quality metrics
        self.logger.info(f"Smoke Test Quality Report: {quality_report}")

        # Save results
        self._save_smoke_test_results(final_claims, language)

        return {
            'success': True,
            'generated': len(final_claims),
            'stats': self._calculate_stats(final_claims),
            'quality_report': quality_report,
            'samples': final_claims[:3] if final_claims else []
        }

    def generate_full_dataset(self,
                              language: str,
                              target: int = 2000,
                              progress_bar=None) -> Dict:
        """Generate full dataset for specified language using only Gemini models"""
        self.logger.info(
            f"Starting full generation for {language}, target: {target} (Gemini only)"
        )

        chunks = self.processor.arabic_chunks if language == "ar" else self.processor.english_chunks
        all_examples = []
        processed_chunks = 0

        # Process chunks in batches (smaller to prevent truncation)
        chunk_batch_size = 5
        examples_per_batch = 8

        while len(all_examples) < target and processed_chunks < len(chunks):
            # Select chunk batch
            chunk_batch = chunks[processed_chunks:processed_chunks +
                                 chunk_batch_size]

            # Generate examples for this batch
            batch_examples = self._generate_with_model(chunk_batch, language,
                                                       examples_per_batch)

            if batch_examples:
                # Verify examples
                verified_batch = self.batch_verify_examples(
                    batch_examples, language)

                # Add valid examples
                for ex in verified_batch:
                    is_valid, _ = validate_example_schema(
                        ex, self.required_fields)
                    if is_valid and len(all_examples) < target:
                        all_examples.append(ex)

            processed_chunks += chunk_batch_size

            # Save progress every 50 examples
            if len(all_examples) % 50 == 0:
                self._save_progress(language, all_examples, processed_chunks)

            # Update progress bar if provided
            if progress_bar:
                progress = len(all_examples) / target
                progress_bar.progress(min(progress, 1.0))

            self.logger.info(
                f"Progress: {len(all_examples)}/{target} examples")

            # Rate limiting
            time.sleep(1)

        # Save final results
        output_file = f"data/generation_stage_B/{language}/judgmental_{language}_final.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for example in all_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        stats = self._compute_stats(all_examples)
        return {
            "success": len(all_examples) >= target * 0.9,
            "stats": stats,
            "total_generated": len(all_examples),
            "output_file": output_file,
            "method": "gemini_only"
        }

    def _save_smoke_test_results(self, claims: List[Dict], language: str):
        """Save smoke test results to a JSONL file"""
        output_file = f"data/generation_stage_B/{language}/smoke_test_results_{language}_{len(claims)}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for claim in claims:
                f.write(json.dumps(claim, ensure_ascii=False) + "\n")
        self.logger.info(f"Saved smoke test results to: {output_file}")

    def _calculate_stats(self, claims: List[Dict]) -> Dict:
        """Calculate statistics for generated claims"""
        total = len(claims)
        true_count = sum(1 for claim in claims
                         if claim.get("verdict") == "True")
        false_count = sum(1 for claim in claims
                          if claim.get("verdict") == "False")
        unknown_count = sum(1 for claim in claims
                            if claim.get("verdict") == "Unknown")
        fabrication_count = sum(1 for claim in claims
                                if claim.get("suspected_fabrication"))

        return {
            "total": total,
            "true": true_count,
            "false": false_count,
            "unknown": unknown_count,
            "fabrications": fabrication_count,
            "fabrication_rate":
            fabrication_count / total if total > 0 else 0.0,
        }

    def _generate_uuid(self) -> str:
        """Generate a unique UUID for claims"""
        return str(uuid.uuid4())

    def _generate_claims_from_chunk(self, chunk, language, max_claims=1):
        """Generate claims from a single chunk of text."""
        try:
            # Import prompts and post-check
            from src.prompts import ARABIC_GENERATOR_PROMPT, ENGLISH_GENERATOR_PROMPT
            from src.post_check import mark_fabrication_if_invalid

            # Get the appropriate prompt - strict mode
            if language == "ar":
                prompt = ARABIC_GENERATOR_PROMPT.format(
                    context=chunk.get('text', ''),
                    chunk_id=chunk.get('id', 0),
                    uuid=self._generate_uuid())
            else:
                prompt = ENGLISH_GENERATOR_PROMPT.format(
                    context=chunk.get('text', ''),
                    chunk_id=chunk.get('id', 0),
                    uuid=self._generate_uuid())

            response = self.gemini_client.call_model(
                prompt=prompt,
                model="gemini-2.5-flash",
                max_tokens=40000,
                temperature=0.0,
            )

            if not response.get("success"):
                self.logger.warning(
                    f"Failed response for chunk {chunk.get('id', 0)}: {response.get('error')}"
                )
                return []

            # Check for truncation
            if response.get("finish_reason") == "MAX_TOKENS":
                self.logger.warning(
                    f"Response truncated for chunk {chunk.get('id', 0)}")
                return []

            # Parse the response - expecting single JSON object
            raw_text = response.get("raw_text", "")
            try:
                if raw_text.strip() == "{}":
                    return []  # Model couldn't generate valid claim

                claim = json.loads(raw_text.strip())
                if not isinstance(claim, dict):
                    return []

                claims = [claim]
            except json.JSONDecodeError:
                self.logger.warning(
                    f"Invalid JSON from chunk {chunk.get('id', 0)}")
                return []

            # Ensure claims conform to expected structure and add metadata
            processed_claims = []
            for claim in claims:
                if not isinstance(claim, dict):
                    continue

                # Add default values if missing
                claim.setdefault("id", self._generate_uuid())
                claim.setdefault("language", language)
                claim.setdefault("context_chunk_id", chunk.get('id', 0))
                claim.setdefault("generator_model", "gemini-2.5-flash")
                claim.setdefault("suspected_fabrication", False)
                claim.setdefault("meta", {
                    "confidence": 0.8,
                    "chunk_source": "context"
                })

                # Truncate context excerpt if too long
                if "context_excerpt" in claim and len(
                        claim["context_excerpt"]) > self.context_max_chars:
                    claim["context_excerpt"] = claim[
                        "context_excerpt"][:self.context_max_chars]

                # Apply post-check validation
                claim = mark_fabrication_if_invalid(claim,
                                                    chunk.get('text', ''))

                # Limit the number of claims generated per chunk
                if len(processed_claims) < max_claims:
                    processed_claims.append(claim)
                else:
                    break  # Stop if max_claims is reached

            return processed_claims

        except Exception as e:
            self.logger.error(
                f"Error generating claims from chunk {chunk.get('id', 0)}: {e}"
            )
            return []


def main():
    parser = argparse.ArgumentParser(
        description=
        "Generate AAOIFI judgmental dataset using Gemini models only")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test")
    parser.add_argument("--full",
                        action="store_true",
                        help="Run full generation")
    parser.add_argument("--lang",
                        choices=["ar", "en"],
                        required=True,
                        help="Language")
    parser.add_argument("--target",
                        type=int,
                        default=2000,
                        help="Target examples for full generation")
    parser.add_argument("--count",
                        type=int,
                        default=15,
                        help="Examples for smoke test")

    args = parser.parse_args()

    try:
        generator = DatasetGenerator()

        if args.smoke:
            results = generator.run_smoke_test(args.lang, args.count)
            print(json.dumps(results, indent=2, ensure_ascii=False))
        elif args.full:
            results = generator.generate_full_dataset(args.lang, args.target)
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print("Specify --smoke or --full")

    except Exception as e:
        logging.error(f"Generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
