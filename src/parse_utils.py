
import json
import re
from typing import Optional, Any, Set

def clean_model_output(text: str) -> str:
    """Strip markdown fences and clean response text"""
    if not text:
        return ""
    
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*', '', text)
    
    # Remove common prefixes
    text = re.sub(r'^[^{\[]*', '', text)
    
    # Remove trailing non-JSON characters
    text = re.sub(r'[^}\]]*$', '', text)
    
    return text.strip()

def parse_json_loose(text: str) -> Optional[Any]:
    """Attempt to parse JSON with error recovery"""
    if not text:
        return None
    
    # Clean first
    cleaned = clean_model_output(text)
    
    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Extract JSON array or object with regex
    patterns = [
        r'(\[.*\])',  # Array
        r'(\{.*\})'   # Object
    ]
    
    for pattern in patterns:
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            json_str = match.group(1)
            
            # Try to fix common issues
            json_str = _fix_json_issues(json_str)
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    
    return None

def _fix_json_issues(json_str: str) -> str:
    """Fix common JSON formatting issues"""
    # Remove trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Fix unbalanced braces by removing characters from end
    brace_count = 0
    bracket_count = 0
    last_valid_pos = len(json_str)
    
    for i, char in enumerate(json_str):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        elif char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
        
        # If we have balanced braces/brackets, mark this position
        if brace_count >= 0 and bracket_count >= 0:
            if (json_str[0] == '{' and brace_count == 0) or (json_str[0] == '[' and bracket_count == 0):
                last_valid_pos = i + 1
                break
    
    return json_str[:last_valid_pos]

def compute_token_overlap(a: str, b: str) -> float:
    """Compute token overlap between two strings"""
    if not a or not b:
        return 0.0
    
    # Normalize text
    def normalize(text: str) -> Set[str]:
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return set(text.split())
    
    tokens_a = normalize(a)
    tokens_b = normalize(b)
    
    if not tokens_a:
        return 0.0
    
    intersection = tokens_a & tokens_b
    return len(intersection) / len(tokens_a)

def validate_example_schema(example: dict, required_fields: list) -> tuple[bool, list]:
    """Validate example against required schema"""
    errors = []
    
    # Check required fields
    for field in required_fields:
        if field not in example:
            errors.append(f"Missing field: {field}")
    
    # Validate specific field types and values
    if "verdict" in example and example["verdict"] not in ["True", "False"]:
        errors.append("Invalid verdict value (must be 'True' or 'False')")
    
    if "language" in example and example["language"] not in ["ar", "en"]:
        errors.append("Invalid language value")
    
    if "suspected_fabrication" in example and not isinstance(example["suspected_fabrication"], bool):
        errors.append("suspected_fabrication must be boolean")
    
    return len(errors) == 0, errors
