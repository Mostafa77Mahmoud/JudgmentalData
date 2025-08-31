
from typing import Dict, Optional
from .find_evidence import find_exact_in_chunk, find_partial_match
from .overlap import token_overlap_rate

# Thresholds (adjustable)
EXACT_THRESHOLD = 1.0
HIGH_OVERLAP = 0.85
PARAPHRASE_OVERLAP = 0.72
MIN_OVERLAP = 0.3

def verify_claim_locally(claim: str, chunk: dict, file_path: str = "") -> dict:
    """
    Local verification without model calls
    Returns verdict with evidence metadata
    """
    chunk_text = chunk.get("text", "")
    chunk_id = chunk.get("id", "unknown")
    
    # Try exact match first
    start_char, end_char = find_exact_in_chunk(claim, chunk_text)
    if start_char != -1:
        return {
            "verdict": "True",
            "method": "exact",
            "evidence": {
                "file_path": file_path,
                "chunk_id": chunk_id,
                "start_char": start_char,
                "end_char": end_char,
                "match_type": "exact"
            },
            "overlap": 1.0,
            "confidence": 0.99,
            "suspected_fabrication": False
        }
    
    # Try partial match (consecutive words)
    partial_match = find_partial_match(claim, chunk_text, min_words=3)
    if partial_match:
        start_char, end_char, matched_text = partial_match
        return {
            "verdict": "True",
            "method": "partial",
            "evidence": {
                "file_path": file_path,
                "chunk_id": chunk_id,
                "start_char": start_char,
                "end_char": end_char,
                "match_type": "partial",
                "matched_text": matched_text
            },
            "overlap": len(matched_text.split()) / len(claim.split()),
            "confidence": 0.85,
            "suspected_fabrication": False
        }
    
    # Calculate token overlap
    overlap = token_overlap_rate(claim, chunk_text)
    
    if overlap >= HIGH_OVERLAP:
        return {
            "verdict": "True",
            "method": "high_overlap",
            "evidence": {
                "file_path": file_path,
                "chunk_id": chunk_id,
                "start_char": -1,
                "end_char": -1,
                "match_type": "paraphrase"
            },
            "overlap": overlap,
            "confidence": 0.8,
            "suspected_fabrication": False
        }
    
    if overlap >= PARAPHRASE_OVERLAP:
        return {
            "verdict": "Ambiguous",
            "method": "paraphrase",
            "evidence": {
                "file_path": file_path,
                "chunk_id": chunk_id,
                "start_char": -1,
                "end_char": -1,
                "match_type": "inferred"
            },
            "overlap": overlap,
            "confidence": 0.6,
            "suspected_fabrication": False
        }
    
    if overlap >= MIN_OVERLAP:
        return {
            "verdict": "Unknown",
            "method": "low_overlap",
            "evidence": None,
            "overlap": overlap,
            "confidence": 0.3,
            "suspected_fabrication": True
        }
    
    return {
        "verdict": "Unknown",
        "method": "no_match",
        "evidence": None,
        "overlap": overlap,
        "confidence": 0.1,
        "suspected_fabrication": True
    }
