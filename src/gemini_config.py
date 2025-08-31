import json
import os
from pathlib import Path
from typing import List, Optional

# Load configuration from config/keys.json
config_file = Path("config/keys.json")
if config_file.exists():
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    API_KEYS = config.get("API_KEYS", [])
    MODELS = config.get("DEFAULT_MODELS",
                        ["gemini-2.5-flash", "gemini-2.5-flash-lite"])
    SINGLE_MODEL_FALLBACK = config.get("SINGLE_MODEL_FALLBACK")
    MAX_FABRICATION_RATE = config.get("MAX_FABRICATION_RATE", 0.10)
    BATCH_SIZE = config.get("BATCH_SIZE", 4)
    CONTEXT_MAX_CHARS = config.get("CONTEXT_MAX_CHARS", 2500)
    MAX_OUTPUT_TOKENS = config.get("MAX_OUTPUT_TOKENS", 4096)
    MAX_RETRIES = config.get("MAX_RETRIES", 5)
    VERIFIER_MODEL = config.get("VERIFIER_MODEL", "gemini-1.5-flash")
    VERIFIER_TEMPERATURE = config.get("VERIFIER_TEMPERATURE", 0.0)
else:
    # Fallback to environment variables
    API_KEYS = [
        os.getenv("GEMINI_API_KEY"),
        os.getenv("GEMINI_KEY_1"),
        os.getenv("GEMINI_KEY_2"),
        os.getenv("GEMINI_KEY_3"),
        os.getenv("GEMINI_KEY_4")
    ]
    API_KEYS = [key for key in API_KEYS if key]
    MODELS = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"]
    SINGLE_MODEL_FALLBACK = os.getenv("SINGLE_MODEL_FALLBACK")
    MAX_FABRICATION_RATE = float(os.getenv("MAX_FABRICATION_RATE", "0.10"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
    CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "800"))
    MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
    VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "gemini-2.5-flash")
    VERIFIER_TEMPERATURE = float(os.getenv("VERIFIER_TEMPERATURE", "0.0"))

# Use single model fallback if specified
if SINGLE_MODEL_FALLBACK:
    MODELS = [SINGLE_MODEL_FALLBACK]

# Verification settings
VERIFIER_MODEL = MODELS[0]  # Use first model for verification
VERIFIER_TEMPERATURE = 0.0
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 16.0

print(f"Loaded {len(API_KEYS)} unique API keys")
print(f"Loaded {len(MODELS)} models from config: {MODELS}")
