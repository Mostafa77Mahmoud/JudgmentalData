
import json
import uuid
import os
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

class StrictDatasetGenerator:
    """Strict dataset generator that follows no-hallucination rules"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.attached_assets_dir = Path("attached_assets")
        
        # Load all available files from attached_assets
        self.available_files = {}
        self._load_attached_assets()
    
    def _load_attached_assets(self):
        """Load all files from attached_assets directory"""
        if not self.attached_assets_dir.exists():
            self.logger.warning("attached_assets directory not found")
            return
        
        for file_path in self.attached_assets_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.available_files[str(file_path)] = content
                    self.logger.info(f"Loaded {file_path}: {len(content)} chars")
            except Exception as e:
                self.logger.error(f"Failed to load {file_path}: {e}")
    
    def _find_exact_evidence(self, claim: str, language: str) -> Dict:
        """Find exact evidence from attached_assets files"""
        best_evidence = {
            "file_path": "attached_assets/no_evidence_found.txt",
            "excerpt": "No supporting evidence found in attached_assets",
            "start_char": 0,
            "end_char": 0,
            "match_type": "inferred"
        }
        
        # Search through all available files
        for file_path, content in self.available_files.items():
            # Try exact match first
            if language == "ar":
                start_pos = content.find(claim)
            else:
                start_pos = content.lower().find(claim.lower())
            
            if start_pos != -1:
                end_pos = start_pos + len(claim)
                excerpt = content[start_pos:min(end_pos + 100, len(content))]  # Add some context
                return {
                    "file_path": file_path,
                    "excerpt": excerpt[:750],
                    "start_char": start_pos,
                    "end_char": end_pos,
                    "match_type": "exact"
                }
            
            # Try paraphrase matching with high threshold
            paraphrase_match = self._find_paraphrase_match(claim, content, file_path)
            if paraphrase_match and paraphrase_match["overlap"] >= 0.85:
                return {
                    "file_path": file_path,
                    "excerpt": paraphrase_match["excerpt"][:750],
                    "start_char": paraphrase_match["start_char"],
                    "end_char": paraphrase_match["end_char"],
                    "match_type": "paraphrase"
                }
        
        return best_evidence
    
    def _find_paraphrase_match(self, claim: str, content: str, file_path: str) -> Optional[Dict]:
        """Find paraphrase match with token overlap"""
        claim_tokens = set(claim.split())
        if len(claim_tokens) == 0:
            return None
        
        words = content.split()
        window_size = min(len(claim_tokens) * 3, 100)
        best_overlap = 0
        best_match = None
        
        for i in range(len(words) - window_size + 1):
            window_text = " ".join(words[i:i + window_size])
            window_tokens = set(window_text.split())
            
            overlap = len(claim_tokens & window_tokens) / len(claim_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                
                # Calculate character positions
                text_before = " ".join(words[:i])
                start_char = len(text_before) + (1 if text_before else 0)
                end_char = start_char + len(window_text)
                
                best_match = {
                    "overlap": overlap,
                    "excerpt": window_text,
                    "start_char": start_char,
                    "end_char": end_char
                }
        
        return best_match
    
    def generate_strict_example(self, claim: str, language: str, target_label: str) -> Dict:
        """Generate a single example following strict rules"""
        example_id = str(uuid.uuid4())
        
        # Find evidence first
        evidence = self._find_exact_evidence(claim, language)
        
        # Determine label based on evidence quality
        if evidence["match_type"] == "exact":
            actual_label = "True"
            confidence = 0.95
            suspected_fabrication = False
            needs_manual_review = False
            explanation = "Exact match found in source material" if language == "en" else "تطابق دقيق موجود في المصدر"
        elif evidence["match_type"] == "paraphrase":
            actual_label = "True"
            confidence = 0.80
            suspected_fabrication = False
            needs_manual_review = False
            explanation = "Strong paraphrase match in source" if language == "en" else "تطابق قوي بالمعنى في المصدر"
        else:
            # No evidence found
            if target_label == "False":
                actual_label = "False"
                explanation = "No evidence supports this claim" if language == "en" else "لا يوجد دليل يدعم هذا الادعاء"
            else:
                actual_label = "Unknown"
                explanation = "No evidence found in attached_assets" if language == "en" else "لا يوجد دليل في المصادر المرفقة"
            
            confidence = 0.90
            suspected_fabrication = True
            needs_manual_review = True
        
        return {
            "id": example_id,
            "language": language,
            "claim": claim,
            "label": actual_label,
            "explanation": explanation,
            "confidence": confidence,
            "evidence": evidence,
            "reference": f"{evidence['file_path']} - {evidence['match_type']} match",
            "generator_meta": {
                "generator_model": "strict_local_generator",
                "prompt_version": "v1",
                "seed_id": f"strict_{example_id[:8]}"
            },
            "raw_response_path": "",
            "suspected_fabrication": suspected_fabrication,
            "needs_manual_review": needs_manual_review
        }
    
    def generate_batch(self, target_count: int, language: str, distribution: Dict[str, float] = None) -> List[Dict]:
        """Generate a batch of examples with specified distribution"""
        if distribution is None:
            distribution = {"True": 0.4, "False": 0.4, "Unknown": 0.2}
        
        examples = []
        
        # Generate sample claims (this would normally come from your seed data)
        sample_claims = self._get_sample_claims(language, target_count)
        
        for i, claim in enumerate(sample_claims):
            # Determine target label based on distribution
            if i / target_count < distribution["True"]:
                target_label = "True"
            elif i / target_count < distribution["True"] + distribution["False"]:
                target_label = "False"
            else:
                target_label = "Unknown"
            
            example = self.generate_strict_example(claim, language, target_label)
            examples.append(example)
        
        return examples[:target_count]
    
    def _get_sample_claims(self, language: str, count: int) -> List[str]:
        """Get sample claims from available content"""
        claims = []
        
        # Extract potential claims from attached_assets files
        for file_path, content in self.available_files.items():
            sentences = re.split(r'[.!?]\s+', content)
            for sentence in sentences[:count]:
                if len(sentence.strip()) > 20 and len(sentence.strip()) < 200:
                    claims.append(sentence.strip())
                if len(claims) >= count:
                    break
            if len(claims) >= count:
                break
        
        return claims[:count]
    
    def validate_no_hallucination(self, examples: List[Dict]) -> Dict:
        """Validate that examples don't contain hallucinations"""
        total = len(examples)
        fabricated = 0
        needs_review = 0
        validation_errors = []
        
        for example in examples:
            # Check if evidence file exists
            evidence_file = example["evidence"]["file_path"]
            if not Path(evidence_file).exists():
                validation_errors.append(f"Evidence file not found: {evidence_file}")
                continue
            
            # Verify evidence positions
            try:
                with open(evidence_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                start_char = example["evidence"]["start_char"]
                end_char = example["evidence"]["end_char"]
                excerpt = example["evidence"]["excerpt"]
                
                if example["evidence"]["match_type"] == "exact":
                    actual_excerpt = content[start_char:end_char]
                    if actual_excerpt != excerpt:
                        validation_errors.append(f"Exact match validation failed for {example['id']}")
                
            except Exception as e:
                validation_errors.append(f"Failed to validate evidence for {example['id']}: {e}")
            
            if example["suspected_fabrication"]:
                fabricated += 1
            if example["needs_manual_review"]:
                needs_review += 1
        
        fabrication_rate = fabricated / max(1, total - needs_review)
        
        return {
            "total_examples": total,
            "fabricated_count": fabricated,
            "needs_manual_review": needs_review,
            "fabrication_rate": fabrication_rate,
            "validation_errors": validation_errors,
            "success": fabrication_rate <= 0.10 and len(validation_errors) == 0
        }
