
import time
import random
import json
import logging
import os
import threading
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from google import genai
from google.genai import types
from src.gemini_config import API_KEYS, MODELS, BATCH_SIZE, MAX_RETRIES, CONTEXT_MAX_CHARS, VERIFIER_MODEL, VERIFIER_TEMPERATURE, MAX_OUTPUT_TOKENS

# Add missing constants
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 60.0

logger = logging.getLogger(__name__)
Path("raw").mkdir(parents=True, exist_ok=True)
Path("manual_review").mkdir(parents=True, exist_ok=True)

class NoTextPartsError(Exception):
    """Raised when no textual parts found in API response"""
    pass

def save_raw_response(response_obj, model_name="unknown_model", attempt: int = 0) -> str:
    """Save raw API response with better serialization"""
    os.makedirs("raw", exist_ok=True)
    ts = int(time.time() * 1000)
    safe_name = model_name.replace("/", "_").replace("models_", "")
    path = f"raw/{ts}_{safe_name}_att{attempt}.resp.json"
    
    try:
        with open(path, "w", encoding="utf8") as f:
            # Handle the new google-genai response format
            if hasattr(response_obj, 'text'):
                response_data = {
                    "text": response_obj.text,
                    "candidates": [
                        {
                            "content": {"parts": [{"text": part.text}] for part in candidate.content.parts if hasattr(part, 'text')},
                            "finish_reason": candidate.finish_reason.name if hasattr(candidate, 'finish_reason') else "UNKNOWN"
                        } for candidate in response_obj.candidates
                    ],
                    "usage_metadata": {
                        "prompt_token_count": response_obj.usage_metadata.prompt_token_count if hasattr(response_obj, 'usage_metadata') else 0,
                        "candidates_token_count": response_obj.usage_metadata.candidates_token_count if hasattr(response_obj, 'usage_metadata') else 0,
                        "total_token_count": response_obj.usage_metadata.total_token_count if hasattr(response_obj, 'usage_metadata') else 0
                    }
                }
            else:
                response_data = {"error": "No text attribute", "raw": str(response_obj)}
            
            json.dump(response_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        # Fallback to plain text if JSON fails
        try:
            with open(path.replace(".json", ".txt"), "w", encoding="utf8") as f:
                f.write(str(response_obj))
            path = path.replace(".json", ".txt")
        except Exception:
            logger.error(f"Failed to save raw response: {e}")
    return path

def rotate_key(attempt_index: int) -> str:
    """Rotate API keys"""
    return API_KEYS[attempt_index % len(API_KEYS)]

def backoff_with_jitter(attempt: int) -> float:
    """Exponential backoff with jitter"""
    base = min(2 ** attempt, MAX_BACKOFF)
    jitter = base * 0.2 * (random.random() * 2 - 1)
    return max(0.1, base + jitter)

def send_verify_request(model_name: str, api_key: str, prompt_text: str, max_tokens: int, attempt: int) -> str:
    """Send verification request using new Google GenAI SDK"""
    try:
        client = genai.Client(api_key=api_key)
        
        config = types.GenerateContentConfig(
            temperature=VERIFIER_TEMPERATURE,
            max_output_tokens=max_tokens,
            response_mime_type="application/json"
        )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt_text,
            config=config
        )

        # Save raw response for debugging
        raw_path = save_raw_response(response, f"verify_{model_name}", attempt)

        # Check if response has text
        if not hasattr(response, 'text') or not response.text:
            # Check finish reason
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    if candidate.finish_reason.name == "MAX_TOKENS":
                        raise ValueError(f"Response truncated due to MAX_TOKENS. Increase max_output_tokens. Raw saved to {raw_path}")
                    elif candidate.finish_reason.name in ["SAFETY", "OTHER"]:
                        raise ValueError(f"Response blocked due to {candidate.finish_reason.name}. Raw saved to {raw_path}")
            
            raise NoTextPartsError(f"No text parts found in response. Raw saved to {raw_path}")
        
        return response.text

    except Exception as e:
        # Save error details
        error_data = {
            "error": str(e),
            "model": model_name,
            "attempt": attempt,
            "timestamp": time.time()
        }
        error_path = save_raw_response(error_data, f"error_{attempt}", attempt)
        logger.error(f"API call failed, error saved to {error_path}")
        raise

def find_json_bounds(text: str) -> Optional[str]:
    """Find first balanced JSON array or object in text"""
    if not text:
        return None
    
    starts = [(m.start(), m.group()) for m in re.finditer(r'[\[\{]', text)]
    for start_index, start_char in starts:
        stack = []
        for i, ch in enumerate(text[start_index:], start_index):
            if ch in ('[', '{'):
                stack.append(ch)
            elif ch in (']', '}'):
                if not stack:
                    break
                stack.pop()
            if not stack:
                return text[start_index:i+1]
    return None

def robust_parse_json_array(text: str) -> Optional[List[Dict]]:
    """Parse model response into JSON list with robust error handling"""
    if not text or len(text.strip()) < 10:
        logger.warning(f"Response too short ({len(text)} chars): {text[:100]}")
        return None

    # Remove common wrapper text and markdown
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Direct parse attempt
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Find balanced JSON structure
    candidate = find_json_bounds(text)
    if candidate:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass

    logger.error(f"Failed to parse JSON from response: {text[:200]}")
    return None

def create_manual_review_item(candidate_id: str, claim: str, chunk_excerpt: str, raw_path: str, model: str, key_index: int, error: str):
    """Create manual review item for failed candidates"""
    review_item = {
        "candidate_id": candidate_id,
        "claim": claim,
        "chunk_excerpt": chunk_excerpt[:500],
        "raw_response_path": raw_path,
        "model_used": model,
        "key_index": key_index,
        "timestamp": time.time(),
        "error": error
    }

    review_file = f"manual_review/{candidate_id}_{int(time.time())}.json"
    with open(review_file, "w", encoding="utf-8") as f:
        json.dump(review_item, f, ensure_ascii=False, indent=2)

    logger.info(f"Created manual review item: {review_file}")
    return review_file

# Strict verifier prompt
VERIFIER_PROMPT = '''
You are a strict JSON-only verifier. Respond with EXACTLY one JSON array containing verification objects and NOTHING else.

For each input item, determine if the claim can be VERIFIED from the context excerpt:

1. If claim has exact/near-exact substring match in context: verdict="True", reference=exact matched text, explanation=brief quote justification, suspected_fabrication=false, confidence=0.90-1.00
2. If context lacks evidence: verdict="False", reference="UNKNOWN", explanation=brief reason, suspected_fabrication=true, confidence=0.1-0.5

Rules:
- Never invent references
- Use exact verdict values "True" or "False"
- Output array length must match input length
- Each object must have: id, language, claim, context_chunk_id, context_excerpt, verdict, explanation, reference, suspected_fabrication, generator_model, raw_response_path, meta

Return ONLY the JSON array:
'''

def prepare_verifier_request(items: List[Dict], max_tokens: int) -> str:
    """Prepare verification request"""
    return VERIFIER_PROMPT + "\n\nINPUT_ITEMS:\n" + json.dumps(items, ensure_ascii=False)

def batch_verify(items: List[Dict]) -> List[Dict]:
    """Batch verify items using new Google GenAI SDK"""
    assert len(items) <= BATCH_SIZE, f"batch size must be <= {BATCH_SIZE}"
    
    # Ensure truncation
    for item in items:
        item["context_excerpt"] = item.get("context_excerpt", "")[:CONTEXT_MAX_CHARS]

    # Calculate tokens conservatively - increase max_tokens significantly
    est_tokens_per_item = 400  # Increased estimate
    max_tokens = min(MAX_OUTPUT_TOKENS * 2, est_tokens_per_item * max(1, len(items)))  # Double the limit
    
    last_err = None
    last_raw_path = None

    for attempt in range(1, MAX_RETRIES + 1):
        api_key = rotate_key(attempt - 1)
        key_index = (attempt - 1) % len(API_KEYS)

        try:
            prompt_text = prepare_verifier_request(items, max_tokens)
            resp_text = send_verify_request(VERIFIER_MODEL, api_key, prompt_text, max_tokens, attempt)
            
            parsed = robust_parse_json_array(resp_text)
            if parsed is None:
                raise ValueError("Failed to parse JSON response")
            
            if len(parsed) != len(items):
                if isinstance(parsed, list) and len(parsed) == 1 and len(items) == 1:
                    # Single item case
                    pass
                elif len(parsed) == 0:
                    raise ValueError("Empty response array")
                else:
                    logger.warning(f"Length mismatch: got {len(parsed)}, expected {len(items)}")

            # Annotate results
            for i, result in enumerate(parsed):
                result.setdefault("raw_response_path", save_raw_response(resp_text, VERIFIER_MODEL, attempt))
                result.setdefault("generator_model", VERIFIER_MODEL)
                result.setdefault("meta", {}).setdefault("confidence", 0.5)
            
            return parsed

        except Exception as e:
            last_err = e
            logger.error(f"Verification failed with key {key_index}, attempt {attempt}: {str(e)}")
            
            # Create manual review items
            for item in items:
                create_manual_review_item(
                    item.get("id", "unknown"),
                    item.get("claim", ""),
                    item.get("context_excerpt", ""),
                    last_raw_path or "unknown",
                    VERIFIER_MODEL,
                    key_index,
                    f"Attempt {attempt}: {str(e)}"
                )

            if attempt < MAX_RETRIES:
                wait_time = backoff_with_jitter(attempt)
                logger.info(f"Backing off for {wait_time:.2f}s (attempt {attempt})")
                time.sleep(wait_time)

    logger.error(f"Batch verify failed after {MAX_RETRIES} attempts: {last_err}")

    # Return failed verification results
    failed_results = []
    for item in items:
        failed_result = {
            "id": item.get("id", "unknown"),
            "language": item.get("language", "unknown"),
            "claim": item.get("claim", ""),
            "context_chunk_id": item.get("context_chunk_id", 0),
            "context_excerpt": item.get("context_excerpt", ""),
            "verdict": "False",
            "explanation": f"Verification failed: {str(last_err)[:50]}",
            "reference": "UNKNOWN",
            "suspected_fabrication": True,
            "generator_model": VERIFIER_MODEL,
            "raw_response_path": last_raw_path or "",
            "meta": {"confidence": 0.0, "verification_failed": True}
        }
        failed_results.append(failed_result)

    return failed_results

def batch_verify_single(items: List[Dict]) -> List[Dict]:
    """Verify candidates one by one (fallback)"""
    results = []
    for item in items:
        single_result = batch_verify([item])
        results.extend(single_result)
    return results

# Legacy GeminiClient class for backward compatibility
class GeminiClient:
    """Legacy wrapper using new Google GenAI SDK"""

    def __init__(self, config_file: str):
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()
        self.api_keys = API_KEYS
        self.current_key_index = 0
        self.key_states = {i: {"blocked_until": 0, "requests_made": 0}
                          for i in range(len(self.api_keys))}
        Path("raw").mkdir(exist_ok=True)

    def _get_next_available_key(self) -> Tuple[Optional[str], int]:
        """Get next available API key"""
        with self.lock:
            current_time = time.time()
            
            for i in range(len(self.api_keys)):
                next_index = (self.current_key_index + i) % len(self.api_keys)
                if self.key_states[next_index]["blocked_until"] <= current_time:
                    self.current_key_index = next_index
                    return self.api_keys[next_index], next_index
            
            return None, -1

    def _block_key(self, key_index: int, duration: int = 300):
        """Block a key for specified duration"""
        with self.lock:
            self.key_states[key_index]["blocked_until"] = time.time() + duration
            self.logger.warning(f"Blocked key {key_index} for {duration}s")

    def call_model(self, prompt: str, model: str = "models/gemini-2.5-flash", max_tokens: int = 8192,
                   temperature: float = 0.0, max_attempts: int = 3) -> Dict:
        """Call Gemini model using new SDK"""
        
        for attempt in range(max_attempts):
            key, key_index = self._get_next_available_key()
            
            if key is None:
                return {"success": False, "error": "No API keys available", "raw_text": ""}

            try:
                client = genai.Client(api_key=key)
                
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

                start_time = time.time()
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config
                )
                latency = time.time() - start_time

                # Update key state
                with self.lock:
                    self.key_states[key_index]["requests_made"] += 1

                raw_path = save_raw_response(response, model, attempt)
                response_text = response.text if hasattr(response, 'text') else ""

                return {
                    "success": True,
                    "raw_text": response_text,
                    "model": model,
                    "key_index": key_index,
                    "latency": latency,
                    "raw_response_path": raw_path
                }

            except Exception as e:
                self.logger.error(f"Error with key {key_index}, attempt {attempt + 1}: {e}")
                
                if "quota" in str(e).lower() or "exhausted" in str(e).lower():
                    self._block_key(key_index, 300)
                elif "rate" in str(e).lower():
                    self._block_key(key_index, 60)
                
                if attempt < max_attempts - 1:
                    wait_time = backoff_with_jitter(attempt)
                    time.sleep(wait_time)
                else:
                    return {"success": False, "error": str(e), "raw_text": ""}

        return {"success": False, "error": "Max attempts exceeded", "raw_text": ""}
