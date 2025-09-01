import time
import random
import json
import logging
import os
import threading
import re
import itertools
import gzip
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path

# Use Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    USE_NEW_SDK = True
except ImportError:
    raise ImportError(
        "google-genai is required. Install with: pip install google-genai")

from src.gemini_config import API_KEYS, MODELS, BATCH_SIZE, MAX_RETRIES, CONTEXT_MAX_CHARS, VERIFIER_MODEL, VERIFIER_TEMPERATURE, MAX_OUTPUT_TOKENS, MAX_INPUT_TOKENS, VERIFICATION_OUTPUT_TOKENS

# Add missing constants
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 60.0

logger = logging.getLogger(__name__)
Path("raw").mkdir(parents=True, exist_ok=True)
Path("manual_review").mkdir(parents=True, exist_ok=True)


class NoTextPartsError(Exception):
    """Raised when no textual parts found in API response"""
    pass


class APIKeyManager:
    """Simple API key manager for rotation"""

    def __init__(self, keys: List[str]):
        self.keys = keys
        self.current_index = 0
        self.lock = threading.Lock()

    def get_next_key(self) -> str:
        with self.lock:
            key = self.keys[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.keys)
            return key


def save_raw_response(response_obj: Any,
                      model_name: str = "unknown_model",
                      attempt: int = 0) -> str:
    """Save raw API response with better serialization"""
    os.makedirs("raw", exist_ok=True)
    ts = int(time.time() * 1000)
    safe_name = model_name.replace("/", "_").replace("models_", "")
    path = f"raw/{ts}_{safe_name}_att{attempt}.resp.json"

    try:
        with open(path, "w", encoding="utf8") as f:
            # Handle the google-genai response format
            if hasattr(response_obj, 'text'):
                response_data = {
                    "text": response_obj.text,
                    "candidates": [],
                    "usage_metadata": {}
                }

                # Extract candidates if available
                if hasattr(response_obj,
                           'candidates') and response_obj.candidates:
                    for candidate in response_obj.candidates:
                        candidate_data = {
                            "content": {
                                "parts": []
                            },
                            "finish_reason": "UNKNOWN"
                        }

                        if hasattr(candidate, 'content') and hasattr(
                                candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'text'):
                                    candidate_data["content"]["parts"].append(
                                        {"text": part.text})

                        if hasattr(candidate, 'finish_reason'):
                            candidate_data[
                                "finish_reason"] = candidate.finish_reason.name if hasattr(
                                    candidate.finish_reason, 'name') else str(
                                        candidate.finish_reason)

                        response_data["candidates"].append(candidate_data)

                # Extract usage metadata if available
                if hasattr(response_obj, 'usage_metadata'):
                    usage = response_obj.usage_metadata
                    response_data["usage_metadata"] = {
                        "prompt_token_count":
                        getattr(usage, 'prompt_token_count', 0),
                        "candidates_token_count":
                        getattr(usage, 'candidates_token_count', 0),
                        "total_token_count":
                        getattr(usage, 'total_token_count', 0)
                    }
            else:
                response_data = {
                    "error": "No text attribute",
                    "raw": str(response_obj)
                }

            json.dump(response_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        # Fallback to plain text if JSON fails
        try:
            with open(path.replace(".json", ".txt"), "w",
                      encoding="utf8") as f:
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
    base = min(2**attempt, MAX_BACKOFF)
    jitter = base * 0.2 * (random.random() * 2 - 1)
    return max(0.1, base + jitter)


def validate_token_limits(prompt: str,
                          max_output_tokens: int) -> Tuple[bool, str]:
    """Validate that prompt and expected output are within token limits"""
    # Rough estimation: 1 token ≈ 3-4 characters for mixed languages
    estimated_input_tokens = len(prompt) // 3

    if estimated_input_tokens > MAX_INPUT_TOKENS:
        return False, f"Input tokens ({estimated_input_tokens}) exceed limit ({MAX_INPUT_TOKENS})"

    if max_output_tokens > MAX_OUTPUT_TOKENS:
        return False, f"Output tokens ({max_output_tokens}) exceed limit ({MAX_OUTPUT_TOKENS})"

    return True, "OK"


def extract_text_from_response(response) -> str:
    """Extract text from various response formats"""
    # Empty response
    if not response:
        return ""

    # Handle dict response (from JSON API)
    if isinstance(response, dict):
        if "candidates" in response:
            texts = []
            for candidate in response["candidates"]:
                if isinstance(candidate, dict):
                    content = candidate.get("content", {})
                    if isinstance(content, dict) and "parts" in content:
                        for part in content["parts"]:
                            if isinstance(part, dict) and "text" in part:
                                texts.append(part["text"])
                    elif isinstance(content, dict) and "text" in content:
                        texts.append(content["text"])
            return "".join(texts)

        if "output" in response:
            output = response["output"]
            if isinstance(output, list):
                return "".join(
                    [p.get("text", "") for p in output if isinstance(p, dict)])
            elif isinstance(output, dict) and "text" in output:
                return output["text"]

    # Default fallback return
    return ""


# Handle SDK response object
def parse_response(response: dict) -> str:
    texts = []

    # لو في candidates
    candidates = response.get("candidates", [])
    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        for part in parts:
            if "text" in part:
                texts.append(part["text"])

    if texts:
        return "".join(texts)

    # fallback لو في response["text"]
    if "text" in response:
        return response["text"]

    return str(response)


def _verify_schema():
    """Simplified JSON Schema without additionalProperties"""
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string"
                },
                "verdict": {
                    "type": "string"
                },
                "explanation": {
                    "type": "string"
                },
                "reference": {
                    "type": "string"
                }
            },
            "required": ["id", "verdict", "explanation", "reference"]
        }
    }


def _build_verify_prompt(items: List[Dict], lang: str) -> str:
    """Build concise verification prompt"""
    if lang == "ar":
        return (
            "تحقق من الادعاءات التالية اعتمادًا على المقتطفات المرفقة فقط.\n"
            "أعد فقط JSON صالح (بدون أي نص إضافي). كل عنصر يجب أن يحتوي:\n"
            "id، verdict (True/False/Unknown)، explanation (≤ 200 حرف، موجز جدًا)، reference.\n"
            "لا تُضِف مقدمات أو تعليقات.\n\n" +
            json.dumps(items, ensure_ascii=False))
    else:
        return (
            "Verify the following claims strictly using the provided excerpts only.\n"
            "Return ONLY valid JSON (no extra text). Each item must have:\n"
            "id, verdict (True/False/Unknown), explanation (≤ 200 chars, very concise), reference.\n"
            "No preface or commentary.\n\n" +
            json.dumps(items, ensure_ascii=False))


def send_verify_request(model_name: str, api_key: str, items: List[Dict],
                        lang: str, attempt: int) -> List[Dict]:
    """Send structured verification request using Google GenAI SDK"""
    try:
        client = genai.Client(api_key=api_key)
        prompt_text = _build_verify_prompt(items, lang)

        response = client.models.generate_content(
            model=model_name,
            contents=prompt_text,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=VERIFICATION_OUTPUT_TOKENS,
                response_mime_type="application/json",
                response_schema=_verify_schema()))

        # Save raw response for debugging
        raw_path = save_raw_response(response, f"verify_{model_name}", attempt)

        # Check finish reason first
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(
                    candidate,
                    'finish_reason') and candidate.finish_reason is not None:
                finish_reason = candidate.finish_reason
                finish_reason_name = finish_reason.name if hasattr(
                    finish_reason, 'name') else str(finish_reason)

                if finish_reason_name == "MAX_TOKENS" or finish_reason == 2:
                    raise RuntimeError(
                        f"Verification truncated due to MAX_TOKENS. Raw saved to {raw_path}"
                    )
                elif finish_reason in [3, 4] or finish_reason_name in [
                        "SAFETY", "RECITATION"
                ]:
                    raise RuntimeError(
                        f"Response blocked due to safety/recitation. Raw saved to {raw_path}"
                    )

        # Extract structured JSON response
        text = None
        try:
            text = response.text
        except Exception:
            # Fallback extraction
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if (hasattr(candidate, 'content')
                        and candidate.content is not None
                        and hasattr(candidate.content, 'parts')
                        and candidate.content.parts is not None):
                    texts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            texts.append(part.text)
                    text = "".join(texts).strip()

        if not text:
            raise RuntimeError(
                f"Empty verification response. Raw saved to {raw_path}")

        # Parse structured JSON
        try:
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError("Verifier returned non-array JSON")
            return data
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse JSON from verifier: {e}\nRaw: {text[:500]}")
            raise RuntimeError(
                f"Invalid JSON from verifier. Raw saved to {raw_path}")

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
                return text[start_index:i + 1]
    return None


def robust_parse_json_array(text: str) -> Optional[List[Dict]]:
    """Parse model response into JSON list with robust error handling"""
    if not text or len(text.strip()) < 10:
        logger.warning(f"Response too short ({len(text)} chars): {text[:100]}")
        return None

    # Clean the response text more thoroughly
    text = text.strip()

    # Remove all markdown blocks
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


def create_manual_review_item(candidate_id: str, claim: str,
                              chunk_excerpt: str, raw_path: str, model: str,
                              key_index: int, error: str) -> str:
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


def chunked(lst, n):
    """Split list into chunks of size n"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def batch_verify(items: List[Dict]) -> List[Dict]:
    """Batch verify items using structured output with 5-item chunks"""
    if not items:
        return []

    # Ensure truncation
    for item in items:
        item["context_excerpt"] = item.get("context_excerpt",
                                           "")[:CONTEXT_MAX_CHARS]

    language = items[0].get("language", "en") if items else "en"

    # Split into small chunks of 5 items to prevent MAX_TOKENS
    ITEMS_PER_CALL = 5
    all_results = []

    for chunk in chunked(items, ITEMS_PER_CALL):
        for attempt in range(1, MAX_RETRIES + 1):
            api_key = rotate_key(attempt - 1)

            try:
                # Prepare verification items (minimal structure)
                verify_items = []
                for item in chunk:
                    verify_items.append({
                        "id":
                        item["id"],
                        "claim":
                        item["claim"],
                        "context_excerpt":
                        item["context_excerpt"]
                    })

                verified = send_verify_request(VERIFIER_MODEL, api_key,
                                               verify_items, language, attempt)

                # Map results back to original structure
                result_map = {v["id"]: v for v in verified}
                for item in chunk:
                    item_id = item["id"]
                    if item_id in result_map:
                        v = result_map[item_id]
                        result = {
                            **item,
                            "verdict":
                            v["verdict"],
                            "explanation":
                            v["explanation"][:200],  # Strict limit
                            "reference":
                            v.get("reference", "UNKNOWN"),
                            "suspected_fabrication":
                            v["verdict"] in ["False", "Unknown"],
                            "generator_model":
                            VERIFIER_MODEL,
                            "raw_response_path":
                            "",
                            "meta": {
                                "confidence":
                                0.8 if v["verdict"] == "True" else 0.2
                            }
                        }
                        all_results.append(result)
                    else:
                        # Missing result
                        failed_result = {
                            **item, "verdict": "False",
                            "explanation": "Verification incomplete",
                            "reference": "UNKNOWN",
                            "suspected_fabrication": True,
                            "generator_model": VERIFIER_MODEL,
                            "raw_response_path": "",
                            "meta": {
                                "confidence": 0.0
                            }
                        }
                        all_results.append(failed_result)

                break  # Success, exit retry loop

            except Exception as e:
                logger.error(
                    f"Verification chunk failed, attempt {attempt}: {str(e)}")

                if attempt == MAX_RETRIES:
                    # Add failed results for this chunk
                    for item in chunk:
                        failed_result = {
                            **item, "verdict": "False",
                            "explanation":
                            f"Verification failed: {str(e)[:50]}",
                            "reference": "UNKNOWN",
                            "suspected_fabrication": True,
                            "generator_model": VERIFIER_MODEL,
                            "raw_response_path": "",
                            "meta": {
                                "confidence": 0.0,
                                "verification_failed": True
                            }
                        }
                        all_results.append(failed_result)
                else:
                    wait_time = backoff_with_jitter(attempt)
                    time.sleep(wait_time)

        # Small delay between chunks
        if len(all_results) < len(items):
            time.sleep(0.5)

    return all_results


def batch_verify_single(items: List[Dict]) -> List[Dict]:
    """Verify candidates one by one (fallback)"""
    results = []
    for item in items:
        single_result = batch_verify([item])
        results.extend(single_result)
    return results


class GeminiClient:
    """Production-ready Gemini client with load balancing and failover"""

    def __init__(self, config_path: str = "config/keys.json"):
        """Initialize Gemini client with API keys"""
        self.config_path = config_path
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.models = self._load_models()
        self.key_manager = APIKeyManager(
            self.api_keys) if self.api_keys else None

        # Initialize key state tracking
        self.lock = threading.Lock()
        self.key_states = {}
        current_time = int(time.time())
        for i in range(len(self.api_keys)):
            self.key_states[i] = {
                "blocked_until": current_time,
                "requests_made": 0
            }

        logger.info(f"Loaded {len(self.api_keys)} unique API keys")
        logger.info(
            f"Loaded {len(self.models)} models from config: {list(self.models.keys())}"
        )

    def _load_api_keys(self) -> List[str]:
        """Load API keys from config file"""
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found: {self.config_path}")
            return []

        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
                keys = config.get("API_KEYS", [])
                if not isinstance(keys, list):
                    logger.error("Invalid 'API_KEYS' format in config file.")
                    return []
                # Ensure keys are unique and not empty
                unique_keys = list(
                    set(k for k in keys if k and isinstance(k, str)))
                if len(unique_keys) != len(keys):
                    logger.warning(
                        "Duplicate or invalid API keys found and removed.")
                return unique_keys
        except json.JSONDecodeError:
            logger.error(
                f"Failed to decode JSON from config file: {self.config_path}")
            return []
        except Exception as e:
            logger.error(
                f"Error loading API keys from {self.config_path}: {e}")
            return []

    def _load_models(self) -> Dict[str, Dict]:
        """Load model configurations from config file"""
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found: {self.config_path}")
            return {}

        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
                models = config.get("models", {})
                if not isinstance(models, dict):
                    logger.error("Invalid 'models' format in config file.")
                    return {}
                return models
        except json.JSONDecodeError:
            logger.error(
                f"Failed to decode JSON from config file: {self.config_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading models from {self.config_path}: {e}")
            return {}

    def _safe_filename(self, prefix, model):
        """Generate safe filename for raw responses"""
        ts = int(time.time() * 1000)
        model_safe = re.sub(r'[^A-Za-z0-9_.-]', '_', model)
        return f"raw/{ts}_{model_safe}.resp.json"

    def _save_raw_response(self, response_obj, prompt, attempt):
        """Save raw response to file for debugging"""
        filename = self._safe_filename("resp", "model")

        try:
            os.makedirs("raw", exist_ok=True)
            response_data = {
                "prompt": prompt[:1000],  # Limited prompt for reference
                "response": str(response_obj),
                "timestamp": time.time(),
                "attempt": attempt
            }

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)
            return filename
        except Exception as e:
            logger.warning(f"Failed to save raw response: {e}")
            return None

    def _get_next_available_key(self) -> Tuple[Optional[str], Optional[int]]:
        """Get next available API key"""
        with self.lock:
            current_time = int(time.time())

            # Try to find an available key
            for i in range(len(self.api_keys)):
                key_index = (self.current_key_index + i) % len(self.api_keys)
                if self.key_states[key_index]["blocked_until"] <= current_time:
                    self.current_key_index = (key_index + 1) % len(
                        self.api_keys)
                    return self.api_keys[key_index], key_index

            # No available keys
            return None, None

    def _block_key(self, key_index: int, duration_seconds: int):
        """Block a key for a specified duration"""
        with self.lock:
            block_until = int(time.time()) + duration_seconds
            self.key_states[key_index]["blocked_until"] = block_until
            logger.warning(f"Blocked key {key_index} until {block_until}")

    def call_model(self,
                   prompt: str,
                   model: str = "gemini-2.5-flash",
                   max_tokens: int = 40000,
                   temperature: float = 0.0,
                   max_attempts: int = 3) -> Dict:
        """Call Gemini model using Google GenAI SDK"""

        # Use available models from config
        available_models = MODELS + [
            "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"
        ]
        if model not in available_models:
            logger.warning(
                f"Model {model} not in available list, using first available model"
            )
            model = available_models[0]

        for attempt in range(max_attempts):
            key, key_index = self._get_next_available_key()

            if key is None:
                return {
                    "success": False,
                    "error": "No API keys available",
                    "raw_text": ""
                }

            try:
                client = genai.Client(api_key=key)

                # Check input token estimate
                estimated_input_tokens = len(prompt) // 3  # Rough estimate
                if estimated_input_tokens > MAX_INPUT_TOKENS:
                    logger.warning(
                        f"Input tokens ({estimated_input_tokens}) may exceed limit ({MAX_INPUT_TOKENS})"
                    )

                start_time = time.time()
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=min(max_tokens, MAX_OUTPUT_TOKENS),
                        response_mime_type="application/json"))
                latency = time.time() - start_time

                # Handle response with new SDK format
                text_content = None
                if hasattr(response,
                           'candidates') and response.candidates and len(
                               response.candidates) > 0:
                    candidate = response.candidates[0]

                    # Check finish reason
                    finish_reason = getattr(candidate, 'finish_reason', None)
                    if finish_reason and finish_reason != 'STOP':
                        # Handle MAX_TOKENS with smart retry
                        finish_reason_str = finish_reason.name if hasattr(
                            finish_reason, 'name') else str(finish_reason)
                        if "MAX_TOKENS" in finish_reason_str.upper(
                        ) and attempt < max_attempts - 1:
                            if max_tokens < MAX_OUTPUT_TOKENS:
                                new_max_tokens = min(MAX_OUTPUT_TOKENS,
                                                     max_tokens * 2)
                                logger.info(
                                    f"Truncated - increasing max_output_tokens from {max_tokens} to {new_max_tokens}"
                                )
                                # Use the current key and retry with increased tokens
                                return self.call_model(prompt, model,
                                                       new_max_tokens,
                                                       temperature,
                                                       max_attempts)
                            else:
                                logger.warning(
                                    "Truncated even at maximum allowed tokens - consider splitting request"
                                )

                        error_msg = f"Model finished with reason: {finish_reason_str}. Response may be incomplete or blocked."
                        logger.warning(error_msg)
                        raw_path = self._save_raw_response(
                            response, prompt, attempt)

                        # If it's truncation, still return partial content if available
                        if "MAX_TOKENS" in finish_reason_str.upper():
                            # Try to extract partial content
                            pass  # Continue to text extraction below
                        else:
                            raise Exception(
                                f"{error_msg} Raw response saved to: {raw_path}"
                            )

                    # Extract text from new SDK response format
                    if (hasattr(candidate, 'content')
                            and candidate.content is not None
                            and hasattr(candidate.content, 'parts')
                            and candidate.content.parts is not None):
                        text_parts = [
                            part.text for part in candidate.content.parts
                            if hasattr(part, 'text') and part.text
                        ]
                        text_content = "\n".join(text_parts)

                # Fallback to response.text if available
                if not text_content and hasattr(response, 'text'):
                    text_content = response.text

                # Update key state
                with self.lock:
                    self.key_states[key_index]["requests_made"] += 1

                raw_path = self._save_raw_response(response, prompt, attempt)
                response_text = text_content or ""  # Use extracted text or fallback

                return {
                    "success": True,
                    "raw_text": response_text,
                    "model": model,
                    "key_index": key_index,
                    "latency": latency,
                    "raw_response_path": raw_path
                }

            except Exception as e:
                logger.error(
                    f"Error with key {key_index}, attempt {attempt + 1}: {e}")

                # Save error details for analysis
                error_data = {
                    "error": str(e),
                    "model": model,
                    "prompt": prompt[:200] + "...",  # Log truncated prompt
                    "attempt": attempt,
                    "timestamp": time.time(),
                    "key_index": key_index
                }
                self._save_raw_response(error_data, f"error_{attempt}",
                                        attempt)

                # Handle different error types
                error_str = str(e).lower()
                if "quota" in error_str or "exhausted" in error_str or "429" in error_str or "rate limit" in error_str:
                    if key_index is not None:
                        self._block_key(key_index, 600)  # Block for 10 minutes
                elif "safety" in error_str:
                    logger.warning(
                        f"Safety filter triggered for model {model}")

                if attempt < max_attempts - 1:
                    wait_time = backoff_with_jitter(attempt)
                    time.sleep(wait_time)
                else:
                    return {"success": False, "error": str(e), "raw_text": ""}

        return {
            "success": False,
            "error": "Max attempts exceeded",
            "raw_text": ""
        }

    def get_key_status(self) -> Dict:
        """Get status of all API keys"""
        with self.lock:
            current_time = int(time.time())
            status = {}
            for i, key in enumerate(self.api_keys):
                masked_key = key[:8] + "..." + key[-4:] if len(
                    key) > 12 else "***"
                status[f"key_{i}"] = {
                    "masked_key": masked_key,
                    "available": self.key_states[i]["blocked_until"]
                    <= current_time,
                    "blocked_until": self.key_states[i]["blocked_until"],
                    "requests_made": self.key_states[i]["requests_made"]
                }
            return status
