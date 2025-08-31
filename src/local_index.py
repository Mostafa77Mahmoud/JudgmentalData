
import json
import re
from collections import defaultdict
from typing import List, Tuple, Dict
from pathlib import Path

def load_chunks(path: str) -> List[dict]:
    """Load chunks from JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Look for chunks in different possible keys
            for key in ['arabic_chunks', 'english_chunks', 'chunks', 'data']:
                if key in data:
                    return data[key]
        return []

def build_index(chunks: List[dict]) -> Tuple[str, List[dict], dict]:
    """
    Build inverted index and normalize chunks
    Returns: full_text, normalized_chunks, inverted_index
    """
    full_text_parts = []
    inverted = defaultdict(list)
    
    for i, chunk in enumerate(chunks):
        # Extract text from various possible fields
        text = (chunk.get("text") or 
                chunk.get("content") or 
                chunk.get("chunk_text") or 
                chunk.get("excerpt", ""))
        
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text).strip()
        chunk["text"] = text
        chunk["chunk_index"] = i
        
        # Ensure chunk has an ID
        if "id" not in chunk:
            chunk["id"] = f"chunk_{i}"
        
        # Build token inverted index
        tokens = text.split()
        for j, token in enumerate(tokens):
            # Clean token for indexing
            clean_token = re.sub(r'[^\w]', '', token.lower())
            if clean_token:
                inverted[clean_token].append((chunk["id"], j, i))
        
        full_text_parts.append(text)
    
    return " ".join(full_text_parts), chunks, inverted

def find_best_chunks_for_claim(claim: str, chunks: List[dict], inverted_index: dict, top_k: int = 3) -> List[Tuple[dict, float]]:
    """Find best matching chunks for a claim using inverted index"""
    claim_tokens = [re.sub(r'[^\w]', '', t.lower()) for t in claim.split()]
    claim_tokens = [t for t in claim_tokens if t]
    
    chunk_scores = defaultdict(float)
    
    for token in claim_tokens:
        if token in inverted_index:
            for chunk_id, pos, chunk_idx in inverted_index[token]:
                chunk_scores[chunk_idx] += 1.0 / len(claim_tokens)
    
    # Sort by score and return top chunks
    scored_chunks = [(chunks[idx], score) for idx, score in chunk_scores.items()]
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    return scored_chunks[:top_k]
