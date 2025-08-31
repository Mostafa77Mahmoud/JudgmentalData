
import re
from typing import Set

def normalize_for_overlap(s: str) -> str:
    """Normalize text for overlap calculation"""
    # Remove punctuation and extra spaces
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()

def token_overlap_rate(ref: str, chunk: str) -> float:
    """Calculate token overlap rate between reference and chunk"""
    ref_tokens = set(normalize_for_overlap(ref).split())
    chunk_tokens = set(normalize_for_overlap(chunk).split())
    
    if not ref_tokens:
        return 0.0
    
    return len(ref_tokens & chunk_tokens) / len(ref_tokens)

def get_overlapping_tokens(ref: str, chunk: str) -> Set[str]:
    """Get set of overlapping tokens"""
    ref_tokens = set(normalize_for_overlap(ref).split())
    chunk_tokens = set(normalize_for_overlap(chunk).split())
    return ref_tokens & chunk_tokens
