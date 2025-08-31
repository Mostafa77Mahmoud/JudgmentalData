import json
import uuid
import time
import random
import re
import argparse
import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from src.gemini_client import GeminiClient
from src.parse_utils import parse_json_loose, compute_token_overlap, validate_example_schema
from src.data_processor import DataProcessor


class DatasetGenerator:
    """Generates judgmental verification datasets with batched processing and two-stage verification"""

    def __init__(self, config_path: str = "config/keys.json"):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self._create_directories()

        self.gemini_client = GeminiClient(config_path)
        self.processor = DataProcessor()
        self.processor.load_data()

        self._load_seeds()

        self.required_fields = [
            "id", "language", "claim", "context_chunk_id", "context_excerpt",
            "verdict", "explanation", "reference", "suspected_fabrication",
            "generator_model", "meta"
        ]

    def _create_directories(self):
        """Create necessary output directories"""
        directories = [
            "data/generation_stage_B/ar", "data/generation_stage_B/en",
            "output/alpaca", "archive", "config", "raw"
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _load_seeds(self):
        """Load QA pairs as generation seeds"""
        try:
            with open("inputs/arabic_qa_pairs.json", 'r',
                      encoding='utf-8') as f:
                self.arabic_seeds = json.load(f)
            with open("inputs/english_qa_pairs.json", 'r',
                      encoding='utf-8') as f:
                self.english_seeds = json.load(f)
            self.logger.info(
                f"Loaded {len(self.arabic_seeds)} Arabic seeds, {len(self.english_seeds)} English seeds"
            )
        except Exception as e:
            self.logger.error(f"Failed to load seeds: {e}")
            raise

    def _normalize_language(self, language_input: str) -> Optional[str]:
        lang = str(language_input).lower().strip()
        if lang in ['ar', 'arabic']: return 'ar'
        if lang in ['en', 'english']: return 'en'
        self.logger.error(
            f"Unsupported language: '{language_input}'. Use 'ar' or 'en'.")
        return None

    def _get_batch_generation_prompt(self, language: str, seeds: List[Dict],
                                     chunks: List[Dict],
                                     batch_size: int) -> str:
        limited_seeds = seeds[:batch_size]
        limited_chunks = []
        for chunk in chunks[:3]:
            chunk_text = chunk.get("text", "")[:1000]
            limited_chunks.append({
                "chunk_id": chunk.get("chunk_id", 0),
                "text": chunk_text
            })
        template = {
            "id": "uuid4",
            "language": language,
            "claim": "claim_text",
            "context_chunk_id": 0,
            "context_excerpt": "excerpt≤512",
            "verdict": "True|False",
            "explanation": "≤120chars",
            "reference": "exact_text_or_UNKNOWN",
            "suspected_fabrication": False,
            "generator_model": "models/gemini-2.5-flash",
            "meta": {
                "confidence": 0.8
            }
        }
        prompt = f"""Generate {batch_size} AAOIFI verification examples. Return JSON array only.

TEMPLATE: {json.dumps(template, ensure_ascii=False)}
RULES:
- If chunk has exact match→verdict:True, reference:exact_substring
- Else→verdict:False, reference:UNKNOWN, suspected_fabrication:true
- Keep explanations under 120 chars
- Use provided seeds/chunks only
DATA: {json.dumps({"seeds": limited_seeds, "chunks": limited_chunks}, ensure_ascii=False)}
JSON ARRAY:"""
        return prompt

    def _get_verification_prompt(self, claim: str, chunk_id: int,
                                 chunk_text: str) -> str:
        return f"""You are an AAOIFI verifier. DO NOT INVENT REFERENCES.
Check the following single claim strictly against the provided chunk text. Return EXACTLY one JSON object including:
- verdict: "True" or "False"
- explanation: a short quote from the chunk if True (≤120 chars)
- reference: the exact substring from the chunk OR "UNKNOWN"
- suspected_fabrication: boolean
- confidence: 0-1
INPUT:
{json.dumps({"claim": claim, "chunk_id": chunk_id, "chunk_text": chunk_text}, ensure_ascii=False)}
Rules:
- If the chunk contains a verbatim match that supports the claim, set verdict:"True" and reference to the exact substring.
- Else set verdict:"False", reference:"UNKNOWN", suspected_fabrication:true.
- DO NOT invent numbers, dates, or standard names.
- Temperature=0.0"""

    def _prepare_batch_seeds(self, language: str,
                             batch_size: int) -> Tuple[List[Dict], List[Dict]]:
        seeds_data = self.arabic_seeds if language == "ar" else self.english_seeds
        chunks_data = self.processor.arabic_chunks if language == "ar" else self.processor.english_chunks
        selected_seeds = random.sample(seeds_data,
                                       min(batch_size * 2, len(seeds_data)))
        batch_seeds, used_chunks = [], set()
        for seed in selected_seeds:
            if len(batch_seeds) >= batch_size: break
            chunk_id = seed.get("chunk_id",
                                random.randint(0,
                                               len(chunks_data) - 1))
            claim = seed.get("answer", seed.get("question", ""))
            if claim:
                batch_seeds.append({"claim": claim, "chunk_id": chunk_id})
                used_chunks.add(chunk_id)
        relevant_chunks = [{
            "chunk_id": i,
            "text": chunks_data[i].get("text", "")[:2000]
        } for i in used_chunks if i < len(chunks_data)]
        return batch_seeds[:batch_size], relevant_chunks

    def _verify_candidate(self, candidate: Dict, chunk_text: str) -> Dict:
        claim = candidate.get("claim", "")
        chunk_id = candidate.get("context_chunk_id", 0)
        prompt = self._get_verification_prompt(claim, chunk_id, chunk_text)
        result = self.gemini_client.call_model("models/gemini-2.5-pro",
                                               prompt,
                                               max_tokens=1024)
        if result["success"]:
            verification = parse_json_loose(result["raw_text"])
            if verification and isinstance(verification, dict):
                candidate.update({
                    "verdict":
                    verification.get("verdict", "False"),
                    "explanation":
                    verification.get("explanation", ""),
                    "reference":
                    verification.get("reference", "UNKNOWN"),
                    "suspected_fabrication":
                    verification.get("suspected_fabrication", True),
                    "meta": {
                        **candidate.get("meta", {}), "confidence":
                        verification.get("confidence", 0.5),
                        "verification_model":
                        "gemini-2.5-pro"
                    }
                })
                reference = candidate.get("reference", "")
                if reference and reference != "UNKNOWN":
                    overlap = compute_token_overlap(reference, chunk_text)
                    if overlap < 0.75:
                        candidate.update({
                            "reference": "UNKNOWN",
                            "suspected_fabrication": True,
                            "verdict": "False",
                            "meta": {
                                **candidate["meta"], "token_overlap": overlap
                            }
                        })
                return candidate
        candidate.update({
            "verdict": "False",
            "reference": "UNKNOWN",
            "suspected_fabrication": True,
            "explanation": "Verification failed"
        })
        return candidate

    def _process_batch(self, language: str, batch_size: int) -> List[Dict]:
        seeds, chunks = self._prepare_batch_seeds(language, batch_size)
        if not seeds or not chunks: return []
        prompt = self._get_batch_generation_prompt(language, seeds, chunks,
                                                   batch_size)
        result = self.gemini_client.call_model("models/gemini-2.5-flash",
                                               prompt,
                                               max_tokens=8192)
        if not result["success"]:
            self.logger.error(
                f"Generation failed: {result.get('error', 'API call failed')}")
            return []
        candidates = parse_json_loose(result["raw_text"])
        if not candidates or not isinstance(candidates, list):
            self.logger.error(
                f"Failed to parse batch response. Raw text: {result['raw_text'][:200]}"
            )
            return []
        verified_examples, chunk_lookup = [], {
            c["chunk_id"]: c["text"]
            for c in chunks
        }
        for candidate in candidates:
            is_valid, _ = validate_example_schema(candidate,
                                                  self.required_fields)
            if not is_valid: continue
            chunk_id = candidate.get("context_chunk_id", 0)
            chunk_text = chunk_lookup.get(chunk_id, "")
            if not chunk_text: continue
            verified_candidate = self._verify_candidate(candidate, chunk_text)
            final_valid, _ = validate_example_schema(verified_candidate,
                                                     self.required_fields)
            if final_valid:
                verified_candidate.update({
                    "id":
                    str(uuid.uuid4()),
                    "language":
                    language,
                    "generator_model":
                    result.get("model", "unknown")
                })
                verified_examples.append(verified_candidate)
        return verified_examples

    def _run_generation_loop(self,
                             language: str,
                             target: int,
                             batch_size: int,
                             is_smoke_test: bool = False):
        lang_code = self._normalize_language(language)
        if not lang_code: return []
        self.logger.info(
            f"Starting {'smoke test' if is_smoke_test else 'full generation'} for {language}..."
        )
        examples, total_attempts, failed_batches = [], 0, 0
        max_attempts = target * 3 if is_smoke_test else (target //
                                                         batch_size) * 2 + 10
        while len(examples) < target and total_attempts < max_attempts:
            batch_examples = self._process_batch(lang_code, batch_size)
            total_attempts += 1
            if batch_examples:
                examples.extend(batch_examples)
                failed_batches = 0
                self.logger.info(
                    f"Batch {total_attempts}: generated {len(batch_examples)}. Total: {len(examples)}/{target}"
                )
                time.sleep(2)
            else:
                failed_batches += 1
                self.logger.warning(
                    f"Batch {total_attempts} failed (consecutive failures: {failed_batches})"
                )
                backoff_time = min(60, 5 * (2**min(failed_batches - 1, 4)))
                self.logger.info(f"Backing off for {backoff_time}s")
                time.sleep(backoff_time)
            if failed_batches >= 8:
                self.logger.error(
                    "Too many consecutive failed batches, stopping.")
                break
        return examples[:target]

    # --- FIX: Re-created the public run_smoke_test method ---
    def run_smoke_test(self, language: str, batch_size: int,
                       smoke_total: int) -> Dict:
        """Runs a smoke test and returns a results dictionary."""
        examples = self._run_generation_loop(language,
                                             smoke_total,
                                             batch_size,
                                             is_smoke_test=True)
        success = len(examples) >= smoke_total * 0.8
        return {
            "success": success,
            "total_generated": len(examples),
            "samples": examples[:3]
        }

    # --- FIX: Re-created the public generate_full_dataset method ---
    def generate_full_dataset(self, language: str, target: int,
                              batch_size: int) -> Dict:
        """Runs a full dataset generation and returns a results dictionary."""
        examples = self._run_generation_loop(language, target, batch_size)
        true_count = sum(
            1 for ex in examples
            if isinstance(ex, dict) and ex.get("verdict") == "True")
        return {
            "success": len(examples) == target,
            "total_generated": len(examples),
            "true_verdicts": true_count,
            "samples": examples[:3]
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate AAOIFI judgmental dataset")
    parser.add_argument("--lang",
                        choices=["ar", "en"],
                        required=True,
                        help="Language")
    parser.add_argument("--mode",
                        choices=["smoke", "full"],
                        required=True,
                        help="Generation mode")
    parser.add_argument("--batch_size",
                        type=int,
                        default=2,
                        help="Batch size for generation")
    parser.add_argument("--smoke_total",
                        type=int,
                        default=10,
                        help="Smoke test target")
    parser.add_argument("--target",
                        type=int,
                        default=2000,
                        help="Full generation target")
    args = parser.parse_args()

    try:
        generator = DatasetGenerator()
        results = {}

        # --- FIX: Main function now correctly calls the restored methods ---
        if args.mode == "smoke":
            results = generator.run_smoke_test(args.lang, args.batch_size,
                                               args.smoke_total)
        else:
            results = generator.generate_full_dataset(args.lang, args.target,
                                                      args.batch_size)

        print(json.dumps(results, indent=2, ensure_ascii=False))

    except Exception as e:
        logging.error(f"An unexpected error occurred in main: {e}",
                      exc_info=True)
        print(json.dumps({"success": False, "error": str(e)}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
