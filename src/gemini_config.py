
# src/gemini_config.py
API_KEYS = [
    "AIzaSyAspAo_UHjOCKxbmtaPCtldZ7g6XowHoV4",
    "AIzaSyCLfpievRZO_J_Ryme_1-1T4SjVBOPCfjI",
    "AIzaSyAIPk1An1O6sZiro64Q4R9PjVrqvPkSVvQ",
    "AIzaSyBbidR_bEfiMrhOufE4PAHrYEBvuPuqakg"
]

# Order models with verification model first
MODELS = [
    "models/gemini-2.5-pro",
    "models/gemini-2.5-flash",
    "models/gemini-2.5-flash-lite"
]

# Tuning
BATCH_SIZE = 6               # safe default (4-8 recommended)
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 30.0
CONTEXT_MAX_CHARS = 512
VERIFIER_MODEL = "models/gemini-2.5-pro"
VERIFIER_TEMPERATURE = 0.0
