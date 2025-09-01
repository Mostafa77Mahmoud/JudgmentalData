import re
import unicodedata
from typing import Tuple, Optional, List, Dict
from difflib import SequenceMatcher
from src.data_processor import DataProcessor
import logging

logger = logging.getLogger(__name__)


class ReferenceVerifier:
    """Verifies references against source text to prevent hallucinations"""

    BATCH_VERIFY_SIZE = 4  # Default batch size for verification

    def __init__(self, processor: DataProcessor):
        self.processor = processor

    def normalize_for_comparison(self, text: str, language: str = "en") -> str:
        """Normalize text for reference comparison"""
        if not text:
            return ""

        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)

        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if language == "en":
            text = text.lower()
        elif language == "ar":
            # Remove Arabic diacritics for better matching
            text = re.sub(r'[\u064B-\u0652\u0670\u0640]', '', text)

        return text

    def compute_token_overlap(self,
                              reference: str,
                              source_text: str,
                              language: str = "en") -> float:
        """Compute token overlap ratio between reference and source"""
        ref_norm = self.normalize_for_comparison(reference, language)
        source_norm = self.normalize_for_comparison(source_text, language)

        ref_tokens = set(ref_norm.split())
        source_tokens = set(source_norm.split())

        if not ref_tokens:
            return 0.0

        overlap = len(ref_tokens & source_tokens)
        return overlap / len(ref_tokens)

    def compute_levenshtein_similarity(self,
                                       reference: str,
                                       source_text: str,
                                       language: str = "en") -> float:
        """Compute normalized Levenshtein similarity"""
        ref_norm = self.normalize_for_comparison(reference, language)
        source_norm = self.normalize_for_comparison(source_text, language)

        # For efficiency, limit source text length for comparison
        if len(source_norm) > 10000:
            # Find the best matching substring in source
            best_ratio = 0.0
            ref_len = len(ref_norm)

            for i in range(0,
                           len(source_norm) - ref_len + 1,
                           100):  # Step by 100 chars
                substr = source_norm[i:i +
                                     ref_len * 2]  # Check 2x reference length
                ratio = SequenceMatcher(None, ref_norm, substr).ratio()
                best_ratio = max(best_ratio, ratio)

            return best_ratio
        else:
            return SequenceMatcher(None, ref_norm, source_norm).ratio()

    def extract_structured_reference(self,
                                     reference: str,
                                     language: str = "en") -> Optional[dict]:
        """Extract structured information from reference strings"""
        if not reference or reference.upper() == "UNKNOWN":
            return None

        patterns = {
            "ar": [
                r'المعيار الشرعي رقم \((\d+)\)', r'البند (\d+/\d+/?\d*)',
                r'الفقرة (\d+)', r'الصفحة (\d+)'
            ],
            "en": [
                r"Shari'ah Standard No\. \((\d+)\)", r"Standard No\. (\d+)",
                r"Clause (\d+/\d+/?\d*)", r"Paragraph (\d+)", r"Page (\d+)"
            ]
        }

        for pattern in patterns.get(language, patterns["en"]):
            match = re.search(pattern, reference, re.IGNORECASE)
            if match:
                return {
                    "type": "structured",
                    "pattern": pattern,
                    "value": match.group(1),
                    "full_match": match.group(0)
                }

        return {"type": "unstructured", "text": reference}

    def verify_reference(
            self,
            reference: str,
            language: str = "en",
            token_threshold: float = 0.75,
            levenshtein_threshold: float = 0.75) -> Tuple[bool, dict]:
        """
        Verify if reference exists in source text

        Returns:
            Tuple of (is_valid, verification_details)
        """
        if not reference or reference.strip().upper() == "UNKNOWN":
            return True, {
                "reference": "UNKNOWN",
                "suspected_fabrication": False,
                "verification_method": "unknown_reference"
            }

        source_text = self.processor.get_source_text(language)
        if not source_text:
            return False, {
                "reference": reference,
                "suspected_fabrication": True,
                "verification_method": "no_source_text",
                "error": "Source text not available"
            }

        # Step 1: Exact substring match (after normalization)
        ref_norm = self.normalize_for_comparison(reference, language)
        source_norm = self.normalize_for_comparison(source_text, language)

        if ref_norm in source_norm:
            return True, {
                "reference": reference,
                "suspected_fabrication": False,
                "verification_method": "exact_match"
            }

        # Step 2: Token overlap analysis
        token_overlap = self.compute_token_overlap(reference, source_text,
                                                   language)

        if token_overlap >= token_threshold:
            return True, {
                "reference": reference,
                "suspected_fabrication": False,
                "verification_method": "token_overlap",
                "token_overlap_ratio": token_overlap
            }

        # Step 3: Levenshtein similarity for fuzzy matching
        levenshtein_sim = self.compute_levenshtein_similarity(
            reference, source_text, language)

        if levenshtein_sim >= levenshtein_threshold:
            return True, {
                "reference": reference,
                "suspected_fabrication": False,
                "verification_method": "levenshtein_similarity",
                "similarity_score": levenshtein_sim
            }

        # Step 4: Structured reference validation
        structured_ref = self.extract_structured_reference(reference, language)
        if structured_ref and structured_ref.get("type") == "structured":
            # For structured references, be more lenient
            if token_overlap >= 0.5:  # Lower threshold for structured refs
                return True, {
                    "reference": reference,
                    "suspected_fabrication": False,
                    "verification_method": "structured_reference_partial",
                    "token_overlap_ratio": token_overlap,
                    "structured_info": structured_ref
                }

        # Reference not found - likely fabricated
        return False, {
            "reference": "UNKNOWN",
            "suspected_fabrication": True,
            "verification_method": "not_found",
            "original_reference": reference,
            "token_overlap_ratio": token_overlap,
            "similarity_score": levenshtein_sim
        }

    def verify_batch(self, candidates: List[Dict]) -> List[Dict]:
        """Verify a batch of candidates with fallback to individual verification"""
        try:
            # Try batch verification first
            results = []
            batch_size = getattr(self, 'BATCH_VERIFY_SIZE', 4)

            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i + batch_size]
                try:
                    batch_results = self._verify_batch_internal(batch)
                    results.extend(batch_results)
                except Exception as e:
                    logger.warning(
                        f"Batch verification failed, falling back to individual: {str(e)}"
                    )
                    # Fallback to one-by-one verification for this batch
                    for candidate in batch:
                        try:
                            reference = candidate.get("reference", "")
                            result = self.verify_reference(reference)
                            results.append(result)

                        except Exception as single_error:
                            logger.error(
                                f"Individual verification failed for candidate {candidate.get('id', 'unknown')}: {str(single_error)}"
                            )
                            # Mark for manual review
                            result = candidate.copy()
                            result['verification_status'] = 'manual_review'
                            result['error'] = str(single_error)
                            results.append(result)

            return results

        except Exception as e:
            logger.error(f"Batch verification completely failed: {str(e)}")
            # Last resort: return all as manual review
            return [
                dict(candidate,
                     verification_status='manual_review',
                     error=str(e)) for candidate in candidates
            ]

    def _verify_batch_internal(self, batch: List[Dict]) -> List[Dict]:
        """Internal batch verification logic"""
        results = []
        for candidate in batch:
            # Assuming candidate is a dictionary containing 'reference' and potentially other fields
            # If candidate itself is the reference string, adjust accordingly
            reference_text = candidate.get('reference') if isinstance(
                candidate, dict) else candidate
            if not reference_text:
                logger.warning(
                    "Skipping verification for candidate with no reference.")
                results.append({
                    "reference": "",
                    "suspected_fabrication": True,
                    "verification_method": "missing_reference_in_batch",
                    "error": "Reference not found in candidate data"
                })
                continue

            try:
                is_valid, details = self.verify_reference(reference_text)
                # Merge original candidate data with verification results
                merged_result = {**candidate, **details, 'is_valid': is_valid}
                results.append(merged_result)
            except Exception as e:
                logger.error(
                    f"Verification failed for reference '{reference_text[:50]}...': {str(e)}"
                )
                # Mark for manual review if verification fails
                manual_review_result = candidate.copy()
                manual_review_result['verification_status'] = 'manual_review'
                manual_review_result['error'] = str(e)
                results.append(manual_review_result)
        return results

    def find_best_reference(
            self,
            claim: str,
            language: str = "en",
            context_chunk_id: Optional[int] = None) -> Tuple[str, dict]:
        """
        Find the best reference for a given claim

        Returns:
            Tuple of (reference_text, verification_details)
        """

        # If context chunk provided, search within that chunk first
        if context_chunk_id is not None:
            chunk = self.processor.get_chunk_by_id(context_chunk_id, language)
            if chunk:
                chunk_text = chunk.get("text", "")

                # Simple approach: find the most similar sentence/paragraph
                chunk_norm = self.normalize_for_comparison(
                    chunk_text, language)

                # Split into sentences and find best match
                sentences = re.split(r'[.!?]\s+', chunk_norm)

                best_sentence = ""
                best_score = 0.0

                for sentence in sentences:
                    if len(sentence.strip()) < 10:  # Skip very short sentences
                        continue

                    score = self.compute_token_overlap(claim, sentence,
                                                       language)
                    if score > best_score:
                        best_score = score
                        best_sentence = sentence

                if best_score > 0.3:  # Reasonable threshold
                    return best_sentence[:200] + "..." if len(
                        best_sentence) > 200 else best_sentence, {
                            "verification_method": "chunk_sentence_match",
                            "chunk_id": context_chunk_id,
                            "match_score": best_score
                        }

        # Fallback: return UNKNOWN
        return "UNKNOWN", {
            "verification_method": "no_suitable_reference",
            "suspected_fabrication": False
        }

    def verify_locally(self, reference: str,
                       chunk_content: str) -> Tuple[bool, Dict]:
        """Local verification using exact matching, token overlap, and semantic similarity"""

        # Check for exact substring match
        if reference.lower() in chunk_content.lower():
            return True, {
                "exact_substring": True,
                "overlap": 1.0,
                "method": "exact"
            }

        # Check token overlap
        ref_tokens = set(reference.lower().split())
        chunk_tokens = set(chunk_content.lower().split())

        if len(ref_tokens) == 0:
            return False, {
                "exact_substring": False,
                "overlap": 0.0,
                "method": "none"
            }

        overlap = len(ref_tokens.intersection(chunk_tokens)) / len(ref_tokens)

        # Require 85% overlap for verification
        if overlap >= 0.85:
            return True, {
                "exact_substring": False,
                "overlap": overlap,
                "method": "token_overlap"
            }

        # Fallback: Basic semantic similarity using word overlap with lower threshold
        # This helps with paraphrasing and different word forms
        if overlap >= 0.6:  # Lower threshold for semantic similarity
            # Check if key terms are present
            ref_words = reference.lower().split()
            chunk_words = chunk_content.lower().split()

            # Count significant word matches (longer than 3 characters)
            significant_ref = [w for w in ref_words if len(w) > 3]
            significant_matches = sum(1 for w in significant_ref if any(
                w in cw or cw in w for cw in chunk_words))

            if significant_ref and significant_matches / len(
                    significant_ref) >= 0.7:
                return True, {
                    "exact_substring": False,
                    "overlap": overlap,
                    "semantic_score":
                    significant_matches / len(significant_ref),
                    "method": "semantic"
                }

        return False, {
            "exact_substring": False,
            "overlap": overlap,
            "method": "failed"
        }
