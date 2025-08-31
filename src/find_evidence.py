
import re
from typing import Tuple, Optional

def find_exact_in_chunk(reference: str, chunk_text: str) -> Tuple[int, int]:
    """
    Find exact substring match and return character offsets
    Returns start,end (char offsets) in chunk_text if exact match exists, else (-1,-1)
    """
    if not reference or not chunk_text:
        return -1, -1
    
    # Try direct exact substring
    idx = chunk_text.find(reference)
    if idx != -1:
        return idx, idx + len(reference)
    
    # Fallback: normalize whitespace and try again
    def normalize_spaces(s):
        return re.sub(r'\s+', ' ', s).strip()
    
    normalized_ref = normalize_spaces(reference)
    normalized_chunk = normalize_spaces(chunk_text)
    
    pos = normalized_chunk.find(normalized_ref)
    if pos == -1:
        return -1, -1
    
    # Map normalized position back to original text (approximation)
    # Find the first token to locate approximate position
    first_token = normalized_ref.split()[0] if normalized_ref.split() else ""
    if first_token:
        match = re.search(re.escape(first_token), chunk_text, re.IGNORECASE)
        if match:
            start = match.start()
            end = min(start + len(normalized_ref), len(chunk_text))
            return start, end
    
    return -1, -1

def find_partial_match(claim: str, chunk_text: str, min_words: int = 3) -> Optional[Tuple[int, int, str]]:
    """
    Find partial match of at least min_words consecutive words
    Returns (start, end, matched_text) or None
    """
    claim_words = claim.split()
    if len(claim_words) < min_words:
        return None
    
    # Try different window sizes
    for window_size in range(len(claim_words), min_words - 1, -1):
        for i in range(len(claim_words) - window_size + 1):
            phrase = " ".join(claim_words[i:i + window_size])
            start, end = find_exact_in_chunk(phrase, chunk_text)
            if start != -1:
                return start, end, phrase
    
    return None
