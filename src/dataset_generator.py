
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
from src.parse_utils import parse_json_loose, compute_token_overlap, validate_example_schema, find_exact_substring
from src.data_processor import DataProcessor

class DatasetGenerator:
    """Production-ready dataset generator with strict reference verification"""

    def __init__(self, config_path: str = "config/keys.json"):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
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
            "output/alpaca", "raw", "logs", "progress"
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _load_seeds(self):
        """Load QA pairs as generation seeds"""
        try:
            # Try both possible filenames
            ar_files = ["inputs/arabic_qa_pairs.json", "inputs/arabic_qa_pairs (2000).json"]
            en_files = ["inputs/english_qa_pairs.json", "inputs/english_qa_pairs (2000).json"]
            
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
                
            self.logger.info(f"Loaded {len(self.arabic_seeds)} Arabic seeds, {len(self.english_seeds)} English seeds")
        except Exception as e:
            self.logger.error(f"Failed to load seeds: {e}")
            raise

    def _get_best_chunk_for_claim(self, claim: str, language: str) -> Tuple[int, str]:
        """Find best matching chunk for a claim"""
        chunks = self.processor.arabic_chunks if language == "ar" else self.processor.english_chunks
        
        best_chunk_id = 0
        best_overlap = 0.0
        best_excerpt = ""
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "")
            overlap = compute_token_overlap(claim, chunk_text)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_chunk_id = i
                # Extract excerpt (first 512 chars or around the match)
                best_excerpt = chunk_text[:512]
                
        return best_chunk_id, best_excerpt

    def _create_false_variants(self, claim: str, language: str) -> List[str]:
        """Create deterministic false variants of a claim"""
        variants = []
        
        if language == "ar":
            # Arabic polarity flips
            if "يجوز" in claim:
                variants.append(claim.replace("يجوز", "لا يجوز"))
            elif "لا يجوز" in claim:
                variants.append(claim.replace("لا يجوز", "يجوز"))
            elif "مباح" in claim:
                variants.append(claim.replace("مباح", "محرم"))
            elif "محرم" in claim:
                variants.append(claim.replace("محرم", "مباح"))
                
            # Standard number changes
            std_match = re.search(r'رقم (\d+)', claim)
            if std_match:
                current_num = int(std_match.group(1))
                new_num = current_num + 1 if current_num < 50 else current_num - 1
                variants.append(claim.replace(f"رقم {current_num}", f"رقم {new_num}"))
                
        else:
            # English polarity flips
            if "permissible" in claim.lower():
                variants.append(claim.replace("permissible", "prohibited"))
                variants.append(claim.replace("Permissible", "Prohibited"))
            elif "prohibited" in claim.lower():
                variants.append(claim.replace("prohibited", "permissible"))
                variants.append(claim.replace("Prohibited", "Permissible"))
            elif "allowed" in claim.lower():
                variants.append(claim.replace("allowed", "forbidden"))
                variants.append(claim.replace("Allowed", "Forbidden"))
                
            # Standard number changes
            std_match = re.search(r'Standard (\d+)', claim)
            if std_match:
                current_num = int(std_match.group(1))
                new_num = current_num + 1 if current_num < 50 else current_num - 1
                variants.append(claim.replace(f"Standard {current_num}", f"Standard {new_num}"))
                
        # Date shifts
        year_match = re.search(r'(\d{4})', claim)
        if year_match:
            current_year = int(year_match.group(1))
            variants.append(claim.replace(str(current_year), str(current_year + 1)))
            
        return variants[:2]  # Return max 2 variants

    def _generate_candidates_from_seeds(self, seeds: List[Dict], language: str) -> List[Dict]:
        """Generate candidates from QA seeds using local methods"""
        candidates = []
        chunks = self.processor.arabic_chunks if language == "ar" else self.processor.english_chunks
        
        for i, seed in enumerate(seeds):
            # Extract claim from answer or question
            claim = seed.get("answer", seed.get("question", "")).strip()
            if not claim:
                continue
                
            seed_id = seed.get("id", f"seed_{i}")
            
            # Get best chunk match
            chunk_id, excerpt = self._get_best_chunk_for_claim(claim, language)
            
            # Create True candidate
            true_candidate = {
                "id": str(uuid.uuid4()),
                "language": language,
                "claim": claim,
                "context_chunk_id": chunk_id,
                "context_excerpt": excerpt,
                "verdict": "True",
                "explanation": "",
                "reference": "",
                "suspected_fabrication": False,
                "generator_model": "local",
                "raw_response_path": "",
                "meta": {"confidence": 0.0, "seed_id": seed_id}
            }
            candidates.append(true_candidate)
            
            # Create False variants
            false_variants = self._create_false_variants(claim, language)
            for variant in false_variants:
                # Use a different chunk for context shift
                wrong_chunk_id = (chunk_id + random.randint(5, 15)) % len(chunks)
                wrong_excerpt = chunks[wrong_chunk_id].get("text", "")[:512]
                
                false_candidate = {
                    "id": str(uuid.uuid4()),
                    "language": language,
                    "claim": variant,
                    "context_chunk_id": wrong_chunk_id,
                    "context_excerpt": wrong_excerpt,
                    "verdict": "False",
                    "explanation": "",
                    "reference": "UNKNOWN",
                    "suspected_fabrication": True,
                    "generator_model": "local",
                    "raw_response_path": "",
                    "meta": {"confidence": 1.0, "seed_id": seed_id}
                }
                candidates.append(false_candidate)
                
        return candidates

    def _local_pre_verification(self, candidates: List[Dict], language: str) -> Tuple[List[Dict], List[Dict]]:
        """Perform local verification without model calls"""
        verified = []
        needs_model_verification = []
        chunks = self.processor.arabic_chunks if language == "ar" else self.processor.english_chunks
        
        for candidate in candidates:
            chunk_id = candidate.get("context_chunk_id", 0)
            claim = candidate.get("claim", "")
            
            if chunk_id >= len(chunks):
                continue
                
            chunk_text = chunks[chunk_id].get("text", "")
            
            # Check for exact substring match
            exact_match = find_exact_substring(claim, chunk_text)
            if exact_match:
                candidate.update({
                    "verdict": "True",
                    "reference": exact_match[:200],  # Limit reference length
                    "explanation": f"Exact match found: {exact_match[:120]}",
                    "suspected_fabrication": False,
                    "meta": {**candidate["meta"], "confidence": 1.0}
                })
                verified.append(candidate)
                continue
                
            # Check token overlap
            overlap = compute_token_overlap(claim, chunk_text)
            if overlap >= 0.75:
                # Find best matching substring
                reference_substring = self._find_best_reference_substring(claim, chunk_text)
                candidate.update({
                    "verdict": "True",
                    "reference": reference_substring,
                    "explanation": f"High overlap match: {reference_substring[:120]}",
                    "suspected_fabrication": False,
                    "meta": {**candidate["meta"], "confidence": overlap}
                })
                verified.append(candidate)
                continue
                
            # For False candidates created locally, they're already correctly labeled
            if candidate.get("verdict") == "False" and candidate.get("generator_model") == "local":
                candidate.update({
                    "explanation": "Deterministic false variant",
                    "meta": {**candidate["meta"], "confidence": 1.0}
                })
                verified.append(candidate)
                continue
                
            # Needs model verification
            needs_model_verification.append(candidate)
            
        self.logger.info(f"Local verification: {len(verified)} verified, {len(needs_model_verification)} need model")
        return verified, needs_model_verification

    def _find_best_reference_substring(self, claim: str, chunk_text: str) -> str:
        """Find best matching substring in chunk for reference"""
        claim_words = claim.split()
        chunk_words = chunk_text.split()
        
        best_match = ""
        best_score = 0
        
        # Try different window sizes
        for window_size in range(min(len(claim_words), 20), 0, -1):
            for i in range(len(chunk_words) - window_size + 1):
                window = " ".join(chunk_words[i:i+window_size])
                overlap = compute_token_overlap(claim, window)
                if overlap > best_score:
                    best_score = overlap
                    best_match = window
                    
        return best_match[:200] if best_match else "UNKNOWN"

    def _get_batch_verification_prompt(self, candidates: List[Dict], language: str) -> str:
        """Create batch verification prompt for model"""
        chunks = self.processor.arabic_chunks if language == "ar" else self.processor.english_chunks
        
        items = []
        for candidate in candidates:
            chunk_id = candidate.get("context_chunk_id", 0)
            chunk_text = chunks[chunk_id].get("text", "")[:4096]  # Limit chunk size
            
            items.append({
                "id": candidate["id"],
                "claim": candidate["claim"],
                "chunk_id": chunk_id,
                "chunk_text": chunk_text
            })
            
        prompt = f"""You are an AAOIFI verifier. DO NOT INVENT REFERENCES and DO NOT use external knowledge beyond the provided chunk_text.

Input JSON:
{json.dumps({"items": items}, ensure_ascii=False)}

Task: For each item return an object with fields:
- id
- verdict: "True" or "False"  
- explanation: If True: a short verbatim quote (≤120 characters) from chunk_text that supports the claim. If False: a concise reason (≤120 chars).
- reference: the exact substring from chunk_text that you quoted OR "UNKNOWN"
- suspected_fabrication: boolean
- confidence: number 0.0-1.0

Rules:
1. Use ONLY the provided chunk_text to verify.
2. If chunk_text includes a verbatim supporting substring then verdict MUST be "True" and reference MUST equal that substring.
3. Otherwise verdict MUST be "False", reference MUST be "UNKNOWN", and suspected_fabrication true.
4. Always return EXACTLY one JSON ARRAY with verification objects in same order as input.

Return: EXACTLY a JSON array. No additional text or markdown fences. Temperature=0.0."""

        return prompt

    def _batch_verify_with_model(self, candidates: List[Dict], language: str, batch_size: int = 8) -> List[Dict]:
        """Verify candidates using model in batches"""
        verified = []
        
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            prompt = self._get_batch_verification_prompt(batch, language)
            
            # Use Gemini Pro for verification
            result = self.gemini_client.call_model(
                "models/gemini-2.0-flash-exp",
                prompt,
                max_tokens=4096,
                temperature=0.0
            )
            
            if not result["success"]:
                self.logger.error(f"Batch verification failed: {result.get('error')}")
                # Mark all as failed
                for candidate in batch:
                    candidate.update({
                        "verdict": "False",
                        "reference": "UNKNOWN", 
                        "explanation": "Verification failed",
                        "suspected_fabrication": True,
                        "raw_response_path": "",
                        "meta": {**candidate["meta"], "confidence": 0.0}
                    })
                verified.extend(batch)
                continue
                
            # Parse verification results
            verifications = parse_json_loose(result["raw_text"])
            if not verifications or not isinstance(verifications, list):
                self.logger.error(f"Failed to parse verification results: {result['raw_text'][:200]}")
                # Mark all as failed
                for candidate in batch:
                    candidate.update({
                        "verdict": "False",
                        "reference": "UNKNOWN",
                        "explanation": "Parse failed", 
                        "suspected_fabrication": True,
                        "raw_response_path": result.get("raw_response_path", ""),
                        "meta": {**candidate["meta"], "confidence": 0.0}
                    })
                verified.extend(batch)
                continue
                
            # Apply verification results
            for j, verification in enumerate(verifications):
                if j >= len(batch):
                    break
                    
                candidate = batch[j]
                chunks = self.processor.arabic_chunks if language == "ar" else self.processor.english_chunks
                chunk_text = chunks[candidate["context_chunk_id"]].get("text", "")
                
                # Validate reference if verdict is True
                reference = verification.get("reference", "UNKNOWN")
                if (verification.get("verdict") == "True" and 
                    reference != "UNKNOWN" and
                    reference not in chunk_text):
                    # Invalid reference, mark as False
                    verification.update({
                        "verdict": "False",
                        "reference": "UNKNOWN",
                        "suspected_fabrication": True
                    })
                    
                candidate.update({
                    "verdict": verification.get("verdict", "False"),
                    "explanation": verification.get("explanation", "")[:120],
                    "reference": verification.get("reference", "UNKNOWN"),
                    "suspected_fabrication": verification.get("suspected_fabrication", True),
                    "raw_response_path": result.get("raw_response_path", ""),
                    "meta": {
                        **candidate["meta"], 
                        "confidence": verification.get("confidence", 0.0)
                    }
                })
                
            verified.extend(batch)
            
            # Rate limiting
            time.sleep(1)
            
        return verified

    def _save_progress(self, language: str, examples: List[Dict], seed_index: int):
        """Save generation progress"""
        progress = {
            "language": language,
            "total_examples": len(examples),
            "seed_index": seed_index,
            "timestamp": time.time(),
            "stats": self._compute_stats(examples)
        }
        
        with open(f"progress/progress_{language}.json", "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    def _compute_stats(self, examples: List[Dict]) -> Dict:
        """Compute dataset statistics"""
        total = len(examples)
        true_count = sum(1 for ex in examples if ex.get("verdict") == "True")
        false_count = sum(1 for ex in examples if ex.get("verdict") == "False")
        fabrication_count = sum(1 for ex in examples if ex.get("suspected_fabrication") == True)
        
        return {
            "total": total,
            "true": true_count,
            "false": false_count,
            "fabrications": fabrication_count,
            "fabrication_rate": fabrication_count / total if total > 0 else 0.0
        }

    def run_smoke_test(self, language: str, target_count: int = 20) -> Dict:
        """Run smoke test with specified number of examples"""
        self.logger.info(f"Starting smoke test for {language} with {target_count} examples")
        
        # Select seeds for smoke test
        seeds_data = self.arabic_seeds if language == "ar" else self.english_seeds
        smoke_seeds = random.sample(seeds_data, min(target_count, len(seeds_data)))
        
        # Generate candidates
        candidates = self._generate_candidates_from_seeds(smoke_seeds, language)
        self.logger.info(f"Generated {len(candidates)} candidates")
        
        # Local pre-verification
        locally_verified, needs_model = self._local_pre_verification(candidates, language)
        
        # Model verification for remaining candidates
        if needs_model:
            model_verified = self._batch_verify_with_model(needs_model, language, batch_size=4)
            all_examples = locally_verified + model_verified
        else:
            all_examples = locally_verified
            
        # Filter valid examples
        valid_examples = []
        for ex in all_examples:
            is_valid, reason = validate_example_schema(ex, self.required_fields)
            if is_valid:
                valid_examples.append(ex)
            else:
                self.logger.warning(f"Invalid example: {reason}")
                
        # Save smoke test results
        output_file = f"data/generation_stage_B/{language}/smoke_test_{language}_{len(valid_examples)}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for example in valid_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                
        stats = self._compute_stats(valid_examples)
        success = len(valid_examples) >= target_count * 0.8 and stats["fabrication_rate"] <= 0.05
        
        return {
            "success": success,
            "stats": stats,
            "total_generated": len(valid_examples),
            "output_file": output_file,
            "samples": valid_examples[:3]
        }

    def generate_full_dataset(self, language: str, target: int = 2000, progress_bar=None) -> Dict:
        """Generate full dataset for specified language"""
        self.logger.info(f"Starting full generation for {language}, target: {target}")
        
        seeds_data = self.arabic_seeds if language == "ar" else self.english_seeds
        all_examples = []
        processed_seeds = 0
        
        # Process seeds in batches
        batch_size = 50
        while len(all_examples) < target and processed_seeds < len(seeds_data):
            batch_seeds = seeds_data[processed_seeds:processed_seeds + batch_size]
            
            # Generate and process batch
            candidates = self._generate_candidates_from_seeds(batch_seeds, language)
            locally_verified, needs_model = self._local_pre_verification(candidates, language)
            
            if needs_model:
                model_verified = self._batch_verify_with_model(needs_model, language)
                batch_examples = locally_verified + model_verified
            else:
                batch_examples = locally_verified
                
            # Filter valid examples
            for ex in batch_examples:
                is_valid, _ = validate_example_schema(ex, self.required_fields)
                if is_valid and len(all_examples) < target:
                    all_examples.append(ex)
                    
            processed_seeds += batch_size
            
            # Save progress every 50 examples
            if len(all_examples) % 50 == 0:
                self._save_progress(language, all_examples, processed_seeds)
                
            # Update progress bar if provided
            if progress_bar:
                progress = len(all_examples) / target
                progress_bar.progress(min(progress, 1.0))
                
            self.logger.info(f"Progress: {len(all_examples)}/{target} examples")
            
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
            "output_file": output_file
        }


def main():
    parser = argparse.ArgumentParser(description="Generate AAOIFI judgmental dataset")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test")
    parser.add_argument("--full", action="store_true", help="Run full generation")
    parser.add_argument("--lang", choices=["ar", "en"], required=True, help="Language")
    parser.add_argument("--target", type=int, default=2000, help="Target examples for full generation")
    parser.add_argument("--smoke-count", type=int, default=20, help="Examples for smoke test")
    
    args = parser.parse_args()
    
    try:
        generator = DatasetGenerator()
        
        if args.smoke:
            results = generator.run_smoke_test(args.lang, args.smoke_count)
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
