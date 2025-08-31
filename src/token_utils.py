
import math
from typing import List

CHARS_PER_TOKEN = 4  # Heuristic for mixed Arabic/English

def estimate_tokens(text: str) -> int:
    """Estimate token count from text length"""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))

def split_by_token_budget(text: str, max_tokens: int) -> List[str]:
    """Split long text into chunks within token budget"""
    tokens = estimate_tokens(text)
    if tokens <= max_tokens:
        return [text]
    
    # Calculate approximate parts needed
    approx_parts = math.ceil(tokens / max_tokens)
    part_len = math.ceil(len(text) / approx_parts)
    
    # Split by character length (rough approximation)
    parts = []
    for i in range(0, len(text), part_len):
        part = text[i:i + part_len]
        if part.strip():
            parts.append(part)
    
    return parts

def validate_token_limits(prompt: str, max_output_tokens: int, max_input_tokens: int) -> tuple[bool, str]:
    """Validate that prompt and expected output are within limits"""
    estimated_input_tokens = estimate_tokens(prompt)
    
    if estimated_input_tokens > max_input_tokens:
        return False, f"Input tokens ({estimated_input_tokens}) exceed limit ({max_input_tokens})"
    
    if max_output_tokens > 60000:  # Gemini 2.5 hard limit
        return False, f"Output tokens ({max_output_tokens}) exceed Gemini 2.5 limit (60000)"
    
    return True, "OK"
