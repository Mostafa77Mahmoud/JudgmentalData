import time
import random
import json
import logging
import os
import threading
import re
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import google.generativeai as genai
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

def save_raw_response(response_obj: Any, model_name: str = "unknown_model", attempt: int = 0) -> str:
    """Save raw API response with better serialization"""
    os.makedirs("raw", exist_ok=True)
    ts = int(time.time() * 1000)
    safe_name = model_name.replace("/", "_").replace("models_", "")
    path = f"raw/{ts}_{safe_name}_att{attempt}.resp.json"

    try:
        with open(path, "w", encoding="utf8") as f:
            # Handle the google-generativeai response format
            if hasattr(response_obj, 'text'):
                response_data = {
                    "text": response_obj.text,
                    "candidates": [],
                    "usage_metadata": {}
                }

                # Extract candidates if available
                if hasattr(response_obj, 'candidates') and response_obj.candidates:
                    for candidate in response_obj.candidates:
                        candidate_data = {
                            "content": {"parts": []},
                            "finish_reason": "UNKNOWN"
                        }

                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'text'):
                                    candidate_data["content"]["parts"].append({"text": part.text})

                        if hasattr(candidate, 'finish_reason'):
                            candidate_data["finish_reason"] = candidate.finish_reason.name if hasattr(candidate.finish_reason, 'name') else str(candidate.finish_reason)

                        response_data["candidates"].append(candidate_data)

                # Extract usage metadata if available
                if hasattr(response_obj, 'usage_metadata'):
                    usage = response_obj.usage_metadata
                    response_data["usage_metadata"] = {
                        "prompt_token_count": getattr(usage, 'prompt_token_count', 0),
                        "candidates_token_count": getattr(usage, 'candidates_token_count', 0),
                        "total_token_count": getattr(usage, 'total_token_count', 0)
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

def extract_text(response):
    """Extract text from response candidates properly"""
    if not response.candidates:
        return None

    candidate = response.candidates[0]

    # Check if content exists and has parts
    if not hasattr(candidate, 'content') or not candidate.content:
        return None

    if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
        return None

    parts = candidate.content.parts
    texts = []
    for p in parts:
        if hasattr(p, "text") and p.text:
            texts.append(p.text)

    return "\n".join(texts) if texts else None

def send_verify_request(model_name: str, api_key: str, prompt_text: str, max_tokens: int, attempt: int) -> str:
    """Send verification request using Google Generative AI SDK"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        generation_config = genai.GenerationConfig(
            temperature=VERIFIER_TEMPERATURE,
            max_output_tokens=max_tokens,
            response_mime_type="application/json"
        )

        response = model.generate_content(
            prompt_text,
            generation_config=generation_config
        )

        # Save raw response for debugging
        raw_path = save_raw_response(response, f"verify_{model_name}", attempt)

        # Extract text properly from response
        result = extract_text(response)
        if not result:
            # Check finish reason for better error handling
            if hasattr(response, 'candidates') and response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish_reason_name = candidate.finish_reason.name if hasattr(candidate.finish_reason, 'name') else str(candidate.finish_reason)
                    if finish_reason_name == "MAX_TOKENS":
                        raise ValueError(f"Response truncated due to MAX_TOKENS. Increase max_output_tokens. Raw saved to {raw_path}")
                    elif finish_reason_name == "STOP":
                        raise ValueError(f"Response completed but no text parts found. Raw saved to {raw_path}")

            raise ValueError(f"Empty response or no text parts. Raw saved to {raw_path}")

        # Check finish reason
        if response.candidates and response.candidates[0].finish_reason:
            finish_reason = response.candidates[0].finish_reason
            if finish_reason == 2:  # MAX_TOKENS
                # Try to extract partial text first
                partial_text = extract_text(response)
                if partial_text and len(partial_text) > 100:  # If we got some reasonable text
                    logger.warning(f"Response truncated but got partial text ({len(partial_text)} chars)")
                    return partial_text
                raise ValueError(f"Response truncated due to MAX_TOKENS. Increase max_output_tokens. Raw saved to {raw_path}")
            elif finish_reason in [3, 4]:  # SAFETY, RECITATION
                raise ValueError(f"Response blocked due to safety/recitation (reason: {finish_reason}). Raw saved to {raw_path}")

        return result

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

    # Clean the response text more thoroughly
    text = text.strip()

    # Remove all markdown blocks
    import re
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)

    # Remove any leading/trailing explanatory text
    text = re.sub(r'^[^[\{]*', '', text)  # Remove text before JSON starts
    text = re.sub(r'[^\]\}]*$', '', text)  # Remove text after JSON ends
    text = text.strip()

    # Direct parse attempt
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError as e:
        logger.warning(f"Direct JSON parse failed: {e}")

    # Find balanced JSON structure
    candidate = find_json_bounds(text)
    if candidate:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError as e:
            logger.warning(f"Balanced JSON parse failed: {e}")

    # Try to fix common JSON issues
    try:
        # Fix incomplete JSON by adding closing brackets
        if text.count('[') > text.count(']'):
            text += ']' * (text.count('[') - text.count(']'))
        if text.count('{') > text.count('}'):
            text += '}' * (text.count('{') - text.count('}'))

        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    logger.error(f"Failed to parse JSON from response: {text[:200]}")
    return None

def create_manual_review_item(candidate_id: str, claim: str, chunk_excerpt: str, raw_path: str, model: str, key_index: int, error: str) -> str:
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
    os.makedirs("manual_review", exist_ok=True)
    with open(review_file, "w", encoding="utf-8") as f:
        json.dump(review_item, f, ensure_ascii=False, indent=2)

    logger.info(f"Created manual review item: {review_file}")
    return review_file

# Updated verifier prompt to be more strict and reduce fabrication
VERIFIER_PROMPT = '''
You are a fact verification system. You must output ONLY valid JSON, without markdown, without explanations, without comments.

CRITICAL RULES:
1. If the provided context does not contain enough information to decide, set "verdict": "Unknown" and leave "explanation" empty
2. Never invent, hallucate, or rephrase information not explicitly present in the context
3. Always copy exact phrases from the provided context when filling fields
4. Only set verdict="True" if the claim is LITERALLY present or can be directly inferred from the context
5. If you cannot find explicit evidence, use verdict="False" or "Unknown"

Output format - ONLY JSON array, no markdown blocks:
[
  {
    "id": "copy_from_input",
    "language": "copy_from_input",
    "claim": "copy_from_input",
    "context_chunk_id": copy_number_from_input,
    "context_excerpt": "copy_from_input",
    "verdict": "True|False|Unknown",
    "explanation": "brief_reasoning_or_empty_if_unknown",
    "reference": "exact_text_from_context_or_UNKNOWN",
    "suspected_fabrication": true_if_false_or_unknown,
    "generator_model": "local",
    "raw_response_path": "",
    "meta": {"confidence": 0.1_to_1.0}
  }
]
'''

def prepare_verifier_request(items: List[Dict], max_tokens: int) -> str:
    """Prepare verification request with language-specific prompts"""
    # Detect language from first item
    language = items[0].get("language", "en") if items else "en"

    try:
        from src.prompts import ARABIC_VERIFIER_PROMPT, ENGLISH_VERIFIER_PROMPT
        if language == "ar":
            base_prompt = ARABIC_VERIFIER_PROMPT
        else:
            base_prompt = ENGLISH_VERIFIER_PROMPT
    except ImportError:
        # Fallback to default prompt
        base_prompt = VERIFIER_PROMPT

    return base_prompt + "\n\nINPUT_ITEMS:\n" + json.dumps(items, ensure_ascii=False)

def batch_verify(items: List[Dict]) -> List[Dict]:
    """Batch verify items using Google Generative AI SDK"""
    assert len(items) <= BATCH_SIZE, f"batch size must be <= {BATCH_SIZE}"

    # Ensure truncation
    for item in items:
        item["context_excerpt"] = item.get("context_excerpt", "")[:CONTEXT_MAX_CHARS]

    # Calculate tokens conservatively based on language
    language = items[0].get("language", "en") if items else "en"
    est_tokens_per_item = 800 if language == "ar" else 400  # Arabic needs more tokens

    # Use smaller max tokens to avoid truncation
    max_tokens = min(MAX_OUTPUT_TOKENS, est_tokens_per_item * max(1, len(items)))

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

            # Create manual review items for persistent failures
            if attempt == MAX_RETRIES:
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

# Production-ready GeminiClient class
class GeminiClient:
    """Production-ready Gemini client with load balancing and failover"""

    def __init__(self, config_file: str = None):
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
            current_time = int(time.time())

            for i in range(len(self.api_keys)):
                next_index = (self.current_key_index + i) % len(self.api_keys)
                if self.key_states[next_index]["blocked_until"] <= current_time:
                    self.current_key_index = next_index
                    return self.api_keys[next_index], next_index

            return None, -1

    def _block_key(self, key_index: int, duration: int = 300):
        """Block a key for specified duration"""
        with self.lock:
            self.key_states[key_index]["blocked_until"] = int(time.time()) + duration
            self.logger.warning(f"Blocked key {key_index} for {duration}s")

    def call_model(self, prompt: str, model: str = "gemini-1.5-flash", max_tokens: int = 8192,
                   temperature: float = 0.0, max_attempts: int = 3) -> Dict:
        """Call Gemini model using Google Generative AI SDK"""

        # Use available models from config
        available_models = MODELS + ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
        if model not in available_models:
            self.logger.warning(f"Model {model} not in available list, using first available model")
            model = available_models[0]

        for attempt in range(max_attempts):
            key, key_index = self._get_next_available_key()

            if key is None:
                return {"success": False, "error": "No API keys available", "raw_text": ""}

            try:
                genai.configure(api_key=key)
                model_instance = genai.GenerativeModel(model)

                generation_config = genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                )

                start_time = time.time()
                response = model_instance.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                latency = time.time() - start_time

                # Update key state
                with self.lock:
                    self.key_states[key_index]["requests_made"] += 1

                raw_path = save_raw_response(response, model, attempt)
                response_text = extract_text(response) or ""

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

                # Handle different error types
                error_str = str(e).lower()
                if "quota" in error_str or "exhausted" in error_str or "429" in error_str:
                    self._block_key(key_index, 600)  # Block for 10 minutes
                elif "rate" in error_str or "limit" in error_str:
                    self._block_key(key_index, 120)   # Block for 2 minutes
                elif "safety" in error_str:
                    self.logger.warning(f"Safety filter triggered for model {model}")

                if attempt < max_attempts - 1:
                    wait_time = backoff_with_jitter(attempt)
                    time.sleep(wait_time)
                else:
                    return {"success": False, "error": str(e), "raw_text": ""}

        return {"success": False, "error": "Max attempts exceeded", "raw_text": ""}

    def get_key_status(self) -> Dict:
        """Get status of all API keys"""
        with self.lock:
            current_time = int(time.time())
            status = {}
            for i, key in enumerate(self.api_keys):
                masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
                status[f"key_{i}"] = {
                    "masked_key": masked_key,
                    "available": self.key_states[i]["blocked_until"] <= current_time,
                    "blocked_until": self.key_states[i]["blocked_until"],
                    "requests_made": self.key_states[i]["requests_made"]
                }
            return status