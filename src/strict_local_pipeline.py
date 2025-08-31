
import json
import uuid
import time
from typing import List, Dict, Tuple
from pathlib import Path
import logging

from .local_index import load_chunks, build_index, find_best_chunks_for_claim
from .verify_local import verify_claim_locally
from .overlap import token_overlap_rate
from .strict_prompts import get_strict_arabic_prompt, get_strict_english_prompt

# Thresholds
MAX_FABRICATION_RATE = 0.10
CONTEXT_MAX_CHARS = 2500

class StrictLocalPipeline:
    """Strict local-first verification pipeline"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.arabic_chunks = []
        self.english_chunks = []
        self.arabic_index = {}
        self.english_index = {}
        self._load_data()
    
    def _load_data(self):
        """Load and index all data sources"""
        # Load Arabic chunks
        ar_file = "inputs/arabic_chunks.json"
        if Path(ar_file).exists():
            self.arabic_chunks = load_chunks(ar_file)
            _, self.arabic_chunks, self.arabic_index = build_index(self.arabic_chunks)
            self.logger.info(f"Loaded {len(self.arabic_chunks)} Arabic chunks")
        
        # Load English chunks
        en_file = "inputs/english_chunks.json"
        if Path(en_file).exists():
            self.english_chunks = load_chunks(en_file)
            _, self.english_chunks, self.english_index = build_index(self.english_chunks)
            self.logger.info(f"Loaded {len(self.english_chunks)} English chunks")
    
    def generate_from_qa_pairs(self, language: str, count: int = 50) -> List[Dict]:
        """Generate examples from QA pairs using local verification"""
        qa_file = f"inputs/{language}_qa_pairs (2000).json"
        if not Path(qa_file).exists():
            qa_file = f"inputs/{language}_qa_pairs.json"
        
        if not Path(qa_file).exists():
            raise FileNotFoundError(f"QA pairs file not found for language {language}")
        
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        
        # Select random sample
        import random
        sample_qa = random.sample(qa_pairs, min(count, len(qa_pairs)))
        
        candidates = []
        chunks = self.arabic_chunks if language == "ar" else self.english_chunks
        index = self.arabic_index if language == "ar" else self.english_index
        file_path = f"attached_assets/{language}_chunks.json"
        
        for qa in sample_qa:
            # Extract claim from answer or question
            claim = qa.get("answer", qa.get("question", "")).strip()
            if not claim:
                continue
            
            # Find best matching chunks
            best_chunks = find_best_chunks_for_claim(claim, chunks, index, top_k=1)
            if not best_chunks:
                continue
            
            best_chunk, score = best_chunks[0]
            
            # Perform local verification
            verification = verify_claim_locally(claim, best_chunk, file_path)
            
            # Create structured example
            example = {
                "id": str(uuid.uuid4()),
                "language": language,
                "claim": claim,
                "label": verification["verdict"],
                "explanation": self._generate_explanation(verification, language),
                "confidence": verification["confidence"],
                "evidence": verification["evidence"],
                "reference": self._generate_reference(verification, best_chunk),
                "generator_meta": {
                    "generator_model": "local",
                    "prompt_version": "v1",
                    "seed_id": qa.get("id", "")
                },
                "raw_response_path": "",
                "suspected_fabrication": verification["suspected_fabrication"],
                "needs_manual_review": verification["verdict"] == "Ambiguous"
            }
            
            candidates.append(example)
        
        return candidates
    
    def _generate_explanation(self, verification: Dict, language: str) -> str:
        """Generate explanation based on verification method"""
        method = verification["method"]
        overlap = verification["overlap"]
        
        if language == "ar":
            if method == "exact":
                return "تطابق نصي دقيق"
            elif method == "partial":
                return "تطابق جزئي للكلمات"
            elif method == "high_overlap":
                return f"تشابه عالي في المفردات ({overlap:.2f})"
            elif method == "paraphrase":
                return f"تشابه متوسط، يحتاج مراجعة ({overlap:.2f})"
            else:
                return "لا يوجد دليل كافي"
        else:
            if method == "exact":
                return "Exact text match"
            elif method == "partial":
                return "Partial word sequence match"
            elif method == "high_overlap":
                return f"High vocabulary overlap ({overlap:.2f})"
            elif method == "paraphrase":
                return f"Moderate overlap, needs review ({overlap:.2f})"
            else:
                return "Insufficient evidence"
    
    def _generate_reference(self, verification: Dict, chunk: Dict) -> str:
        """Generate reference string"""
        if verification["evidence"]:
            evidence = verification["evidence"]
            chunk_id = evidence["chunk_id"]
            if evidence["start_char"] != -1:
                return f"{evidence['file_path']}#{chunk_id}[{evidence['start_char']}:{evidence['end_char']}]"
            else:
                return f"{evidence['file_path']}#{chunk_id}"
        else:
            return "UNKNOWN"
    
    def run_smoke_test(self, language: str, count: int = 20) -> Dict:
        """Run smoke test with strict local verification"""
        self.logger.info(f"Running strict smoke test for {language} with {count} examples")
        
        try:
            examples = self.generate_from_qa_pairs(language, count)
            
            # Filter and validate
            valid_examples = []
            for ex in examples:
                if self._validate_example(ex):
                    valid_examples.append(ex)
            
            # Compute stats
            stats = self._compute_stats(valid_examples)
            
            # Save results
            output_dir = Path(f"data/generation_stage_B/{language}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"strict_smoke_test_{language}_{len(valid_examples)}.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for example in valid_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            
            # Check fabrication rate
            success = stats["fabrication_rate"] <= MAX_FABRICATION_RATE
            
            return {
                "success": success,
                "stats": stats,
                "total_generated": len(valid_examples),
                "output_file": str(output_file),
                "samples": valid_examples[:3],
                "error": None if success else f"Fabrication rate too high: {stats['fabrication_rate']:.2%}"
            }
            
        except Exception as e:
            self.logger.error(f"Smoke test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "stats": {},
                "total_generated": 0
            }
    
    def _validate_example(self, example: Dict) -> bool:
        """Validate example structure"""
        required_fields = ["id", "language", "claim", "label", "confidence", "evidence"]
        return all(field in example for field in required_fields)
    
    def _compute_stats(self, examples: List[Dict]) -> Dict:
        """Compute dataset statistics"""
        total = len(examples)
        if total == 0:
            return {"total": 0, "fabrication_rate": 0.0}
        
        true_count = sum(1 for ex in examples if ex.get("label") == "True")
        false_count = sum(1 for ex in examples if ex.get("label") == "False")
        unknown_count = sum(1 for ex in examples if ex.get("label") == "Unknown")
        fabrication_count = sum(1 for ex in examples if ex.get("suspected_fabrication") is True)
        
        return {
            "total": total,
            "true": true_count,
            "false": false_count,
            "unknown": unknown_count,
            "fabrications": fabrication_count,
            "fabrication_rate": fabrication_count / total
        }
