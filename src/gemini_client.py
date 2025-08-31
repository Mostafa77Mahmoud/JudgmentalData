# src/gemini_client.py
import time
import random
import json
import logging
import os
import re
import threading
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from google.protobuf.json_format import MessageToDict
from src.gemini_config import API_KEYS, MODELS, BATCH_SIZE, MAX_RETRIES, INITIAL_BACKOFF, MAX_BACKOFF, CONTEXT_MAX_CHARS, VERIFIER_MODEL, VERIFIER_TEMPERATURE, MAX_OUTPUT_TOKENS

logger = logging.getLogger(__name__)
Path("raw").mkdir(parents=True, exist_ok=True)
Path("manual_review").mkdir(parents=True, exist_ok=True)

class NoTextPartsError(Exception):
    """Raised when no textual parts found in API response"""
    pass

def save_raw_body(response, prefix):
    ts = int(time.time())
    os.makedirs("raw", exist_ok=True)
    path = f"raw/{ts}_{prefix}.resp.json"
    try:
        with open(path, "w", encoding="utf8") as f:
            # Attempt to serialize using MessageToDict for better protobuf handling
            try:
                f.write(json.dumps(MessageToDict(response._raw_response), ensure_ascii=False, indent=2))
            except Exception as e:
                logger.warning(f"Could not serialize response with MessageToDict: {e}. Falling back to raw text.")
                f.write(response.text) # Fallback to raw text if MessageToDict fails
        return path
    except Exception as e:
        logger.error(f"Failed to save raw response to {path}: {e}")
        return None


def extract_text_from_parsed(body_dict: Dict) -> Optional[str]:
    """
    Extracts text from a parsed response dictionary.
    """
    if not isinstance(body_dict, dict):
        return None

    # Try known nested structures for candidates
    for candidate_list_key in ["candidates", "outputs", "choices", "responses"]:
        candidates = body_dict.get(candidate_list_key)
        if isinstance(candidates, list) and candidates:
            first_candidate = candidates[0]
            if isinstance(first_candidate, dict):
                # Look for 'content' or 'message' which might contain 'parts'
                for content_key in ["content", "message", "response"]:
                    content = first_candidate.get(content_key)
                    if isinstance(content, dict) and "parts" in content:
                        parts = content["parts"]
                        if isinstance(parts, list):
                            text_parts = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
                            return "".join(text_parts)
                    # Handle cases where content itself might be the text
                    if isinstance(content, str):
                        return content
                # If no 'content' or 'message', check candidate directly for 'text'
                if "text" in first_candidate and isinstance(first_candidate["text"], str):
                    return first_candidate["text"]

    # Fallback: Check top-level keys for text directly
    for key in ["output_text", "response_text", "text", "content", "message"]:
        if isinstance(body_dict.get(key), str):
            return body_dict.get(key)

    return None


def extract_text_from_response(resp):
    """
    resp: requests.Response or a parsed dict
    returns: text string or raises ValueError with saved path info
    """
    if hasattr(resp, "text"):
        raw_text = resp.text
        try:
            body = resp.json()
        except Exception:
            body = None
    else:
        body = resp
        raw_text = json.dumps(body, ensure_ascii=False)
    saved_path = save_raw_response(raw_text, "verify")

    # Try known shapes
    def try_from_candidate_list(lst):
        if not isinstance(lst, list): return None
        first = lst[0] if lst else None
        if not first: return None
        # candidate may have nested output -> parts
        for field in ("output", "response", "content", "message"):
            cand_part = first.get(field) if isinstance(first, dict) else None
            if isinstance(cand_part, dict) and "parts" in cand_part:
                parts = cand_part["parts"]
                if isinstance(parts, list):
                    return "".join(p.get("text","") for p in parts if isinstance(p, dict))
            if isinstance(cand_part, list):
                # try list of parts
                text_acc = []
                for item in cand_part:
                    if isinstance(item, dict) and "text" in item:
                        text_acc.append(item["text"])
                if text_acc:
                    return "".join(text_acc)
            if isinstance(cand_part, str):
                return cand_part
        # fallback: check first-level keys
        for k in ("text","content","message"):
            if isinstance(first.get(k), str):
                return first.get(k)
        return None

    # body-level tries
    if isinstance(body, dict):
        for key in ("candidates","outputs","output","choices","responses"):
            val = body.get(key)
            if val:
                t = try_from_candidate_list(val) if isinstance(val, list) else (val.get("text") if isinstance(val, dict) else None)
                if t:
                    return t
        # try top-level simple text field
        for k in ("text","output_text","response_text"):
            if isinstance(body.get(k), str):
                return body.get(k)

    # if no text found:
    raise NoTextPartsError(f"No textual parts found in API response. Raw saved to {saved_path}")

def rotate_key(attempt_index: int) -> str:
    return API_KEYS[attempt_index % len(API_KEYS)]

def exponential_backoff_sleep(attempt: int):
    base = INITIAL_BACKOFF * (2 ** (attempt - 1))
    jitter = random.uniform(0, base * 0.1)
    sleep_for = min(base + jitter, MAX_BACKOFF)
    logger.info("backoff sleep: %.2fs (attempt %d)", sleep_for, attempt)
    time.sleep(sleep_for)

def send_verify_request(model_name: str, api_key: str, prompt_text: str, max_tokens: int, attempt: int) -> str:
    """
    Send verification request using Google Generative AI SDK with robust parsing
    """
    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": VERIFIER_TEMPERATURE,
        "max_output_tokens": max_tokens,
    }

    model_instance = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config
    )

    try:
        response = model_instance.generate_content(prompt_text)

        # Save raw response for debugging
        raw_response_data = {
            "candidates": getattr(response, 'candidates', []),
            "prompt_feedback": getattr(response, 'prompt_feedback', None),
            "usage_metadata": getattr(response, 'usage_metadata', None)
        }

        # Try to extract text using robust parsing
        try:
            return extract_text_from_response(raw_response_data)
        except (NoTextPartsError, ValueError) as e:
            # Log the specific error and re-raise
            logger.error(f"Failed to extract text from response: {e}")
            raise

    except Exception as e:
        # Save error details
        error_data = {
            "error": str(e),
            "model": model_name,
            "attempt": attempt,
            "timestamp": time.time()
        }
        error_path = save_raw_response(json.dumps(error_data, ensure_ascii=False), f"error_{attempt}")
        logger.error(f"API call failed, error saved to {error_path}")
        raise

def find_json_bounds(text: str):
    """Find first balanced JSON array or object in text and return substring, or None."""
    if not text:
        return None
    # search for opening bracket
    starts = [(m.start(), m.group()) for m in re.finditer(r'[\[\{]', text)]
    for start_index, start_char in starts:
        stack = []
        for i, ch in enumerate(text[start_index:], start_index):
            if ch in ('[', '{'):
                stack.append(ch)
            elif ch in (']', '}'):
                if not stack:
                    break
                opening = stack.pop()
                # allow mismatch but continue
            if not stack:
                return text[start_index:i+1]
    return None

def robust_parse_json_array(text: str):
    """Try several heuristics to parse model response into a JSON list of objects."""
    if not text:
        return None

    # Basic validation - reject if too short or contains error markers
    if len(text.strip()) < 50:
        logger.warning(f"Response too short ({len(text)} chars): {text[:100]}")
        return None

    if any(marker in text.lower() for marker in ["error", "traceback", "exception"]):
        logger.warning(f"Response contains error markers: {text[:200]}")
        return None

    # remove code fences and common wrapper text
    text = re.sub(r"^```(?:json)?\n", "", text.strip())
    text = re.sub(r"\n```$", "", text)
    text = text.strip()

    # direct parse
    try:
        j = json.loads(text)
        if isinstance(j, list):
            return j
        if isinstance(j, dict):
            return [j]
    except Exception:
        pass

    # find first balanced structure
    candidate = find_json_bounds(text)
    if candidate:
        try:
            j = json.loads(candidate)
            if isinstance(j, list):
                return j
            if isinstance(j, dict):
                return [j]
        except Exception:
            # try simple repair: if opened '[' but missing ']', add closing brackets
            if candidate.count('[') > candidate.count(']'):
                repaired = candidate + (']' * (candidate.count('[') - candidate.count(']')))
                try:
                    j = json.loads(repaired)
                    if isinstance(j, list):
                        return j
                    if isinstance(j, dict):
                        return [j]
                except Exception:
                    pass
    # fallback: regex-extract top-level objects
    objs = re.findall(r'\{(?:[^{}]|\{[^}]*\})*\}', text, flags=re.DOTALL)
    parsed = []
    for o in objs:
        try:
            parsed.append(json.loads(o))
        except Exception:
            continue
    if parsed:
        return parsed
    return None

def create_manual_review_item(candidate_id: str, claim: str, chunk_excerpt: str, raw_path: str, model: str, key_index: int, error: str):
    """Create manual review item for failed candidates"""
    review_item = {
        "candidate_id": candidate_id,
        "claim": claim,
        "chunk_excerpt": chunk_excerpt[:500],  # Truncate for readability
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

# The strict verifier prompt to include in requests (kept as constant)
VERIFIER_PROMPT = r"""
System: You are a strict JSON-only verifier. You MUST respond with exactly one JSON array containing one object per input item and NOTHING else (no prose, no backticks, no commentary).

Input: an array `items` where each item has:
- id: string
- claim: string
- context_excerpt: string (already truncated to <=512 chars)
- language: "ar" or "en"
- context_chunk_id: integer

Task: For each item decide whether the claim can be VERIFIED from the context excerpt.
Rules:
1. If an exact substring of the claim (or a near-literal short quote) exists inside context_excerpt, set verdict to "True", reference to that exact matched substring, explanation to a one-sentence quote-based justification (<=30 words), suspected_fabrication false, confidence 0.90-1.00.
2. If the context does NOT contain evidence to support the claim, set verdict to "False", reference: "UNKNOWN", explanation: short (<=20 words), suspected_fabrication true or false as appropriate, confidence 0.1-0.5.
3. Never invent references. If unsure, choose "False" with reference "UNKNOWN".
4. Use verdict values exactly "True" or "False".
5. The output JSON array length must match input length and preserve input order. Each object MUST include fields:
   { "id", "language", "claim", "context_chunk_id", "context_excerpt", "verdict", "explanation", "reference", "suspected_fabrication", "generator_model", "raw_response_path", "meta" }
6. meta must contain at least {"confidence": <0-1>, "seed_id": "<seed>"}. generator_model must be the model name used.

Return only the JSON array.
"""

def prepare_verifier_request(items: List[Dict], max_tokens: int):
    # Build the request text for the Gemini model
    prompt_text = VERIFIER_PROMPT + "\n\nINPUT_ITEMS_JSON:\n" + json.dumps(items, ensure_ascii=False)
    return prompt_text

def batch_verify(items: List[Dict]) -> List[Dict]:
    """
    items: list dict with id, claim, context_excerpt, language, context_chunk_id
    returns: parsed verification objects in the same order
    """
    assert len(items) <= BATCH_SIZE, f"batch size must be <= {BATCH_SIZE}"
    # ensure truncation
    for it in items:
        it["context_excerpt"] = it.get("context_excerpt","")[:CONTEXT_MAX_CHARS]

    # estimate tokens per item; conservative default
    est_tokens_per_item = 300
    max_tokens = min(MAX_OUTPUT_TOKENS, est_tokens_per_item * max(1, len(items)))

    last_err = None
    last_raw_path = None

    for attempt in range(1, MAX_RETRIES + 1):
        api_key = rotate_key(attempt - 1)
        key_index = (attempt - 1) % len(API_KEYS)

        try:
            prompt_text = prepare_verifier_request(items, max_tokens)
            resp_text = send_verify_request(VERIFIER_MODEL, api_key, prompt_text, max_tokens, attempt)
            ts = int(time.time())
            raw_path = f"raw/{ts}_verify_{attempt}_{VERIFIER_MODEL.replace('/','_')}.resp.txt"
            with open(raw_path, "w", encoding="utf8") as f:
                f.write(resp_text)

            parsed = robust_parse_json_array(resp_text)
            if parsed is None:
                raise ValueError("robust_parse_json_array returned None")
            if not isinstance(parsed, list) or len(parsed) != len(items):
                # If length mismatch, attempt to wrap single object into array
                if isinstance(parsed, dict):
                    parsed = [parsed]
                else:
                    raise ValueError(f"Parsed length mismatch: parsed_len={len(parsed) if isinstance(parsed, list) else 'NA'} expected={len(items)}")

            # annotate raw path and generator_model
            for p in parsed:
                p.setdefault("raw_response_path", raw_path)
                p.setdefault("generator_model", VERIFIER_MODEL)
            return parsed

        except NoTextPartsError as e:
            last_err = e
            last_raw_path = str(e).split("Raw saved to ")[-1] if "Raw saved to" in str(e) else None
            logger.error("No text parts error with key index %d, attempt %d: %s", key_index, attempt, str(e))

            # Create manual review items for each candidate
            for item in items:
                create_manual_review_item(
                    item.get("id", "unknown"),
                    item.get("claim", ""),
                    item.get("context_excerpt", ""),
                    last_raw_path or "unknown",
                    VERIFIER_MODEL,
                    key_index,
                    f"NoTextPartsError: {str(e)}"
                )

            if attempt < MAX_RETRIES:
                exponential_backoff_sleep(attempt)
                continue
            else:
                break

        except Exception as e:
            last_err = e
            logger.error("Unexpected error with key index %d, attempt %d: %s", key_index, attempt, str(e))
            if attempt < MAX_RETRIES:
                exponential_backoff_sleep(attempt)
                continue
            else:
                break

    logger.error("Batch verify failed after %d attempts: %s", MAX_RETRIES, last_err)

    # Return failed verification results for each item
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
    """Verify candidates one by one (fallback for failed batch verification)"""
    results = []
    for item in items:
        single_result = batch_verify([item])
        results.extend(single_result)
    return results

# Legacy GeminiClient class for backward compatibility
class GeminiClient:
    """Legacy wrapper to maintain compatibility with existing code"""

    def __init__(self, config_file: str):
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()

        # Use the centralized API keys
        self.api_keys = API_KEYS
        self.current_key_index = 0
        self.key_states = {i: {"blocked_until": 0, "requests_made": 0, "last_response": None} 
                          for i in range(len(self.api_keys))}

        # Create raw responses directory
        Path("raw").mkdir(exist_ok=True)

        # Use centralized models
        self.models = MODELS

    def _get_next_available_key(self) -> Tuple[Optional[str], int]:
        """Get next available API key, rotating if current is blocked"""
        with self.lock:
            current_time = time.time()

            # Check if current key is available
            current_state = self.key_states[self.current_key_index]
            if current_state["blocked_until"] <= current_time:
                key = self.api_keys[self.current_key_index]
                return key, self.current_key_index

            # Find next available key
            for i in range(len(self.api_keys)):
                next_index = (self.current_key_index + i + 1) % len(self.api_keys)
                if self.key_states[next_index]["blocked_until"] <= current_time:
                    self.current_key_index = next_index
                    key = self.api_keys[next_index]
                    return key, next_index

            # All keys are blocked, find the one that unblocks soonest
            min_blocked_until = min(state["blocked_until"] for state in self.key_states.values())
            wait_time = max(0, min_blocked_until - current_time)
            self.logger.warning(f"All keys blocked. Waiting {wait_time:.1f}s until next available")

            return None, -1

    def _block_key(self, key_index: int, duration: int = 300):
        """Block a key for specified duration"""
        with self.lock:
            self.key_states[key_index]["blocked_until"] = time.time() + duration
            self.logger.warning(f"Blocked key {key_index} for {duration}s")

    def _save_raw_response(self, prompt: str, response: str, model: str, key_index: int) -> str:
        """Save raw API request and response"""
        timestamp = int(time.time())
        filename = f"raw/{timestamp}_{key_index}_{model.replace('/', '_')}.resp.txt"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"TIMESTAMP: {timestamp}\n")
                f.write(f"MODEL: {model}\n")
                f.write(f"KEY_INDEX: {key_index}\n")
                f.write(f"PROMPT:\n{prompt}\n")
                f.write(f"\nRESPONSE:\n{response}\n")
            return filename
        except Exception as e:
            self.logger.error(f"Failed to save raw response: {e}")
            return ""

    def call_model(self, prompt: str, model: str = "models/gemini-2.5-pro", max_tokens: int = 8192,
                   temperature: float = 0.0, max_attempts: int = 3) -> Dict:
        """Call Gemini model with robust error handling and key rotation"""

        for attempt in range(max_attempts):
            key, key_index = self._get_next_available_key()

            if key is None:
                # All keys blocked, wait for next available
                min_blocked_until = min(state["blocked_until"] for state in self.key_states.values())
                wait_time = max(0, min_blocked_until - time.time())
                if wait_time > 0:
                    self.logger.info(f"Waiting {wait_time:.1f}s for key to become available")
                    time.sleep(wait_time)
                    continue
                else:
                    return {"success": False, "error": "No API keys available", "raw_text": ""}

            try:
                # Configure the API key
                genai.configure(api_key=key)

                # Configure the model
                generation_config = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }

                model_obj = genai.GenerativeModel(
                    model_name=model,
                    generation_config=generation_config
                )

                # Make the API call
                start_time = time.time()
                response = model_obj.generate_content(prompt)
                latency = time.time() - start_time

                # Update key state
                with self.lock:
                    self.key_states[key_index]["requests_made"] += 1
                    self.key_states[key_index]["last_response"] = time.time()

                # Use robust text extraction from parsed response
                body_dict = safe_serialize_response(response) # Ensure response is serialized safely
                response_text = extract_text_from_parsed(body_dict)

                if response_text is None:
                    logger.warning(f"No text extracted from {model}, attempt {attempt + 1}")
                    response_text = "" # Ensure it's an empty string if extraction fails
                    save_raw_body(response, "notext")

                raw_path = self._save_raw_response(prompt, response_text, model, key_index)

                return {
                    "success": True,
                    "raw_text": response_text,
                    "model": model,
                    "key_index": key_index,
                    "latency": latency,
                    "raw_response_path": raw_path
                }

            except google_exceptions.ResourceExhausted as e:
                self.logger.warning(f"Key {key_index} quota exhausted: {e}")
                self._block_key(key_index, 300)  # Block for 5 minutes
                continue

            except google_exceptions.TooManyRequests as e:
                self.logger.warning(f"Rate limit hit for key {key_index}: {e}")
                self._block_key(key_index, 60)  # Block for 1 minute
                continue

            except Exception as e:
                self.logger.error(f"Unexpected error with key {key_index}, attempt {attempt + 1}: {e}")
                # Save raw response on any exception for debugging
                try:
                    save_raw_body(response, f"exception_{attempt+1}")
                except NameError: # response might not be defined if error happened before its assignment
                    pass 

                if attempt < max_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    return {"success": False, "error": str(e), "raw_text": ""}

        return {"success": False, "error": "Max attempts exceeded", "raw_text": ""}


def safe_serialize_response(response):
    """ Safely serializes the response object, returning a dictionary. """
    try:
        # Use MessageToDict for proper protobuf serialization if available
        return MessageToDict(response._raw_response)
    except AttributeError:
        # Fallback if _raw_response is not available or not a protobuf message
        logger.warning("Could not access _raw_response for MessageToDict, using response.text")
        try:
            return response.json()
        except Exception:
            return {"text": response.text} # Return text if json parsing fails


def extract_text_from_parsed(body_dict: Dict) -> Optional[str]:
    """
    Extracts text from a parsed response dictionary, looking into candidates.
    """
    if not isinstance(body_dict, dict):
        return None

    # Prioritize 'candidates' which is common in Gemini API responses
    for candidate_list_key in ["candidates", "outputs", "choices", "responses"]:
        candidates = body_dict.get(candidate_list_key)
        if isinstance(candidates, list) and candidates:
            first_candidate = candidates[0]
            if isinstance(first_candidate, dict):
                # Look for 'content' which often contains 'parts'
                content = first_candidate.get("content")
                if isinstance(content, dict) and "parts" in content:
                    parts = content["parts"]
                    if isinstance(parts, list):
                        # Extract text from each part
                        text_parts = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
                        return "".join(text_parts)
                # If no 'parts', the 'content' itself might be the text
                if isinstance(content, str):
                    return content
                # Check if 'text' is directly in the candidate
                if "text" in first_candidate and isinstance(first_candidate["text"], str):
                    return first_candidate["text"]
                # Check for 'finish_reason' to understand if response was cut short
                if first_candidate.get("finish_reason") in ["MAX_TOKENS", "OTHER", "SAFETY"]:
                    logger.warning(f"Response may be truncated due to finish_reason: {first_candidate.get('finish_reason')}")


    # Fallback to top-level keys if candidates structure not found
    for key in ["output_text", "response_text", "text", "content", "message"]:
        if isinstance(body_dict.get(key), str):
            return body_dict.get(key)

    return None