import os
import json
import time
import re
import threading
from typing import Dict, Optional, Any
from pathlib import Path

        # --- FIX: Imports for older SDK version (0.8.5) that your environment is using ---
import google.generativeai as genai
from google.generativeai.client import configure
from google.generativeai.generative_models import GenerativeModel
from google.generativeai import types
from google.api_core import exceptions as google_exceptions
import logging

class GeminiClient:
            """Robust Gemini client with key rotation, batching, and comprehensive error handling"""

            def __init__(self, config_path: str = "config/keys.json"):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)

                self.api_keys = self.config["keys"]
                self.rate_limit_per_min = self.config["per_key_rate_limit_per_min"]

                self.current_key_index = 0
                self.key_usage_count = {i: 0 for i in range(len(self.api_keys))}
                self.key_usage_reset_time = {i: time.time() for i in range(len(self.api_keys))}
                self.blocked_until = {i: 0.0 for i in range(len(self.api_keys))}

                self.lock = threading.Lock()

                Path("raw").mkdir(exist_ok=True)
                Path("logs").mkdir(exist_ok=True)

                logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f'logs/pipeline_run_{int(time.time())}.log'),
                        logging.StreamHandler()
                    ]
                )
                self.logger = logging.getLogger(__name__)

            def _is_key_available(self, key_index: int) -> bool:
                current_time = time.time()
                if self.blocked_until[key_index] > current_time:
                    return False
                if current_time - self.key_usage_reset_time[key_index] >= 60:
                    self.key_usage_count[key_index] = 0
                    self.key_usage_reset_time[key_index] = current_time
                return self.key_usage_count[key_index] < self.rate_limit_per_min

            def _get_next_available_key(self) -> Optional[int]:
                for i in range(len(self.api_keys)):
                    key_idx = (self.current_key_index + i) % len(self.api_keys)
                    if self._is_key_available(key_idx):
                        return key_idx
                return None

            def _block_key(self, key_index: int, duration: float):
                self.blocked_until[key_index] = time.time() + duration
                self.logger.warning(f"Blocked key {key_index} for {duration:.1f}s")

            def _save_raw_response(self, response_text: str, metadata: Dict) -> str:
                timestamp = int(time.time())
                key_idx = metadata.get('key_index', 0)
                model_name_raw = metadata.get('model', 'unknown')
                model = model_name_raw.replace('/', '_').replace('-', '_')
                filename = f"{timestamp}_{key_idx}_{model}.resp.txt"
                filepath = Path("raw") / filename
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(response_text)
                    return str(filepath)
                except Exception as e:
                    self.logger.error(f"Failed to save raw response to {filepath}: {e}")
                    return f"failed_to_save_{filename}"

            def _wait_for_available_key(self) -> int:
                while True:
                    available_key = self._get_next_available_key()
                    if available_key is not None:
                        return available_key

                    current_time = time.time()
                    min_wait = float('inf')

                    for i, blocked_time in self.blocked_until.items():
                        if blocked_time > current_time:
                            wait_time = blocked_time - current_time
                            min_wait = min(min_wait, wait_time)

                    for i, reset_time in self.key_usage_reset_time.items():
                        if self.key_usage_count[i] >= self.rate_limit_per_min:
                            wait_time = 60 - (current_time - reset_time)
                            if wait_time > 0:
                                min_wait = min(min_wait, wait_time)

                    wait_duration = min(min_wait, 10) if min_wait != float('inf') else 1.0
                    self.logger.info(f"All keys busy. Waiting {wait_duration:.1f}s")
                    time.sleep(wait_duration)

            def call_model(self, model_name: str, prompt: str, max_tokens: int = 8192) -> Dict[str, Any]:
                with self.lock:
                    key_idx = self._wait_for_available_key()
                    self.current_key_index = key_idx
                    # --- FIX: Use the directly imported configure function ---
                    configure(api_key=self.api_keys[key_idx])
                    self.key_usage_count[key_idx] += 1

                for attempt in range(3):
                    try:
                        # --- FIX: Use the directly imported classes and types from the older SDK ---
                        model = GenerativeModel(model_name)
                        generation_config = types.GenerationConfig(
                            temperature=0.0,
                            max_output_tokens=max_tokens,
                            top_p=0.1
                        )
                        safety_settings={
                            types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: types.HarmBlockThreshold.BLOCK_NONE,
                            types.HarmCategory.HARM_CATEGORY_HARASSMENT: types.HarmBlockThreshold.BLOCK_NONE,
                            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: types.HarmBlockThreshold.BLOCK_NONE,
                            types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: types.HarmBlockThreshold.BLOCK_NONE,
                        }

                        response = model.generate_content(
                            prompt,
                            generation_config=generation_config,
                            safety_settings=safety_settings
                        )

                        response_text, is_truncated = "", False
                        finish_reason = None
                        if response.prompt_feedback.block_reason:
                            finish_reason = response.prompt_feedback.block_reason.name

                        if response.candidates:
                            candidate = response.candidates[0]
                            finish_reason = candidate.finish_reason.name
                            if hasattr(candidate, 'content') and candidate.content.parts:
                                response_text = candidate.content.parts[0].text

                            if candidate.finish_reason.name == 'MAX_TOKENS':
                                is_truncated = True
                                self.logger.warning("Response truncated due to max tokens")

                        success = len(response_text.strip()) > 0 and not is_truncated
                        metadata = {"key_index": key_idx, "model": model_name, "finish_reason": finish_reason}
                        raw_path = self._save_raw_response(response_text, metadata)

                        return {
                            "success": success, "raw_text": response_text, "model": model_name,
                            "truncated": is_truncated, "raw_path": raw_path, "error": None if success else f"Generation failed: {finish_reason}"
                        }

                    except google_exceptions.ResourceExhausted as e:
                        error_str = str(e)
                        match = re.search(r'retry_delay {\s*seconds: (\d+)\s*}', error_str)
                        retry_delay = float(match.group(1)) if match else 60.0

                        self._block_key(key_idx, retry_delay)
                        self.logger.error(f"Rate limited on key {key_idx}. Parsed retry_delay: {retry_delay}s.")

                        with self.lock:
                            key_idx = self._wait_for_available_key()
                            configure(api_key=self.api_keys[key_idx])
                            self.current_key_index = key_idx
                            self.key_usage_count[key_idx] += 1

                    except Exception as e:
                        self.logger.error(f"API call failed on attempt {attempt + 1} with key {key_idx}: {e}")
                        time.sleep(2 ** attempt)

                return {"success": False, "raw_text": "", "model": model_name, "truncated": False, "raw_path": None, "error": "All API attempts failed"}