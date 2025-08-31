
import os
import time
import json
import logging
import threading
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

class GeminiClient:
    """Robust Gemini client with key rotation, rate limiting, and error handling"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()
        
        # Load API keys from environment or config
        self.api_keys = self._load_api_keys(config_path)
        if not self.api_keys:
            raise ValueError("No valid API keys found")
            
        self.current_key_index = 0
        self.key_states = {i: {"blocked_until": 0, "requests_made": 0, "last_response": None} 
                          for i in range(len(self.api_keys))}
        
        # Create raw responses directory
        Path("raw").mkdir(exist_ok=True)
        
        # Load models from config or use defaults
        self.models = self._load_models_from_config(config_path)
        if not self.models:
            # Default to 2.5 models if no config
            self.models = [
                "models/gemini-2.5-pro",
                "models/gemini-2.5-flash", 
                "models/gemini-2.5-flash-lite"
            ]
        
    def _load_api_keys(self, config_path: Optional[str]) -> List[str]:
        """Load API keys from environment variables or config file"""
        keys = []
        
        # Try environment variables first
        env_keys = [
            os.getenv("GEMINI_API_KEY"),
            os.getenv("GEMINI_KEY_1"), 
            os.getenv("GEMINI_KEY_2"),
            os.getenv("GEMINI_KEY_3"),
            os.getenv("GEMINI_KEY_4")
        ]
        
        # Add hardcoded keys as fallback
        fallback_keys = [
            "AIzaSyAspAo_UHjOCKxbmtaPCtldZ7g6XowHoV4",
            "AIzaSyCLfpievRZO_J_Ryme_1-1T4SjVBOPCfjI", 
            "AIzaSyAIPk1An1O6sZiro64Q4R9PjVrqvPkSVvQ",
            "AIzaSyBbidR_bEfiMrhOufE4PAHrYEBvuPuqakg"
        ]
        
        all_keys = env_keys + fallback_keys
        keys = [key for key in all_keys if key and key.strip()]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keys = []
        for key in keys:
            if key not in seen:
                seen.add(key)
                unique_keys.append(key)
                
        self.logger.info(f"Loaded {len(unique_keys)} unique API keys")
        return unique_keys

    def _load_models_from_config(self, config_path: Optional[str]) -> List[str]:
        """Load model list from config file"""
        if not config_path or not Path(config_path).exists():
            return []
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                
            models = config_data.get("models", [])
            if models:
                self.logger.info(f"Loaded {len(models)} models from config: {models}")
                return models
                
        except Exception as e:
            self.logger.warning(f"Failed to load models from config: {e}")
            
        return []
        
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
            
    def call_model(self, model: str, prompt: str, max_tokens: int = 8192, 
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
                
                model_instance = genai.GenerativeModel(
                    model_name=model,
                    generation_config=generation_config
                )
                
                # Make the API call
                start_time = time.time()
                response = model_instance.generate_content(prompt)
                latency = time.time() - start_time
                
                # Update key state
                with self.lock:
                    self.key_states[key_index]["requests_made"] += 1
                    self.key_states[key_index]["last_response"] = time.time()
                
                # Log metadata
                self._log_metadata(key_index, model, attempt + 1, latency, "200", prompt[:100])
                
                response_text = response.text if response.text else ""
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
                if attempt < max_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                continue
                
        return {"success": False, "error": "Max attempts exceeded", "raw_text": ""}
        
    def _log_metadata(self, key_index: int, model: str, attempt: int, 
                     latency: float, status: str, prompt_preview: str):
        """Log request metadata"""
        metadata = {
            "timestamp": time.time(),
            "key_index": key_index,
            "model": model,
            "attempt": attempt,
            "latency": latency,
            "status": status,
            "prompt_preview": prompt_preview
        }
        
        try:
            with open("logs/metadata.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(metadata) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to log metadata: {e}")
