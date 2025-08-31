
import json
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

def parse_json_loose(text: str) -> Optional[Any]:
    """Robust JSON parsing with fallback strategies"""
    logger = logging.getLogger(__name__)
    
    if not text or not text.strip():
        return None
        
    # Remove common markdown artifacts
    text = text.strip()
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE) 
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    
    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
        
    # Try to find JSON object or array
    json_patterns = [
        r'\{.*\}',  # JSON object
        r'\[.*\]'   # JSON array
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
                
    # Try to extract balanced braces/brackets
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue
            
        depth = 0
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx:i+1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                        
    logger.warning(f"Failed to parse JSON from text: {text[:200]}...")
    return None

def compute_token_overlap(ref_text: str, chunk_text: str) -> float:
    """Compute token overlap between reference and chunk text"""
    if not ref_text or not chunk_text:
        return 0.0
        
    # Normalize text for comparison
    ref_tokens = set(normalize_text_for_overlap(ref_text).split())
    chunk_tokens = set(normalize_text_for_overlap(chunk_text).split())
    
    if not ref_tokens:
        return 0.0
        
    intersection = ref_tokens.intersection(chunk_tokens)
    return len(intersection) / len(ref_tokens)

def normalize_text_for_overlap(text: str) -> str:
    """Normalize text for token overlap computation"""
    import unicodedata
    
    # Unicode normalization
    text = unicodedata.normalize('NFKD', text)
    
    # Remove diacritics (Arabic tashkeel)
    text = re.sub(r'[\u064B-\u065F\u0670\u06D6-\u06ED]', '', text)
    
    # Normalize punctuation and whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Lowercase for English, preserve Arabic casing
    if re.search(r'[a-zA-Z]', text):
        text = text.lower()
        
    return text.strip()

def validate_example_schema(example: Dict, required_fields: List[str]) -> Tuple[bool, str]:
    """Validate example against required schema"""
    if not isinstance(example, dict):
        return False, "Example is not a dictionary"
        
    # Check required fields
    for field in required_fields:
        if field not in example:
            return False, f"Missing required field: {field}"
            
    # Validate specific constraints
    if example.get("verdict") not in ["True", "False"]:
        return False, "verdict must be 'True' or 'False'"
        
    if example.get("verdict") == "True":
        if not example.get("reference") or example.get("reference") == "UNKNOWN":
            return False, "True verdict requires valid reference"
        if not example.get("explanation"):
            return False, "True verdict requires explanation"
            
    if not isinstance(example.get("suspected_fabrication"), bool):
        return False, "suspected_fabrication must be boolean"
        
    context_excerpt = example.get("context_excerpt", "")
    if len(context_excerpt) > 512:
        return False, "context_excerpt exceeds 512 characters"
        
    return True, "Valid"

def find_exact_substring(reference: str, chunk_text: str) -> Optional[str]:
    """Find exact substring match in chunk text"""
    if not reference or not chunk_text or reference == "UNKNOWN":
        return None
        
    # Try exact match first
    if reference in chunk_text:
        return reference
        
    # Try normalized match
    norm_ref = normalize_text_for_overlap(reference)
    norm_chunk = normalize_text_for_overlap(chunk_text)
    
    if norm_ref in norm_chunk:
        # Find the actual substring in original text
        ref_words = norm_ref.split()
        chunk_words = chunk_text.split()
        
        for i in range(len(chunk_words) - len(ref_words) + 1):
            candidate = ' '.join(chunk_words[i:i+len(ref_words)])
            if normalize_text_for_overlap(candidate) == norm_ref:
                return candidate
                
    return None
