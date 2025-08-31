import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import unicodedata
import re

class DataProcessor:
    """Processes AAOIFI data files and handles text normalization"""
    
    def __init__(self):
        self.arabic_chunks = []
        self.english_chunks = []
        self.arabic_qa_pairs = []
        self.english_qa_pairs = []
        self.arabic_text = ""
        self.english_text = ""
        
    def load_data(self):
        """Load all required data files"""
        try:
            # Load chunks
            with open("inputs/arabic_chunks.json", 'r', encoding='utf-8') as f:
                self.arabic_chunks = json.load(f)
                
            with open("inputs/english_chunks.json", 'r', encoding='utf-8') as f:
                self.english_chunks = json.load(f)
                
            # Load QA pairs
            with open("inputs/arabic_qa_pairs.json", 'r', encoding='utf-8') as f:
                self.arabic_qa_pairs = json.load(f)
                
            with open("inputs/english_qa_pairs.json", 'r', encoding='utf-8') as f:
                self.english_qa_pairs = json.load(f)
                
            # Load cleaned texts
            with open("inputs/arabic_cleaned.txt", 'r', encoding='utf-8') as f:
                self.arabic_text = f.read()
                
            with open("inputs/english_cleaned.txt", 'r', encoding='utf-8') as f:
                self.english_text = f.read()
                
        except Exception as e:
            raise Exception(f"Failed to load data files: {str(e)}")
    
    def normalize_text(self, text: str, language: str = "en") -> str:
        """Normalize text for comparison and matching"""
        if not text:
            return ""
            
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Language-specific normalization
        if language == "en":
            text = text.lower()
        elif language == "ar":
            # Optional: remove Arabic diacritics for better matching
            text = re.sub(r'[\u064B-\u0652\u0670\u0640]', '', text)
            
        return text
    
    def get_chunk_by_id(self, chunk_id: int, language: str) -> Optional[Dict]:
        """Get chunk by ID for specified language"""
        chunks = self.arabic_chunks if language == "ar" else self.english_chunks
        
        for chunk in chunks:
            if chunk.get("id") == chunk_id:
                return chunk
        return None
    
    def get_source_text(self, language: str) -> str:
        """Get the full source text for specified language"""
        return self.arabic_text if language == "ar" else self.english_text
    
    def extract_context_excerpt(self, chunk_id: int, language: str, max_tokens: int = 512) -> str:
        """Extract context excerpt from chunk, limited by tokens"""
        chunk = self.get_chunk_by_id(chunk_id, language)
        if not chunk:
            return ""
            
        text = chunk.get("text", "")
        
        # Simple token estimation (split by whitespace)
        tokens = text.split()
        if len(tokens) <= max_tokens:
            return text
            
        # Truncate and add ellipsis
        truncated_tokens = tokens[:max_tokens]
        return " ".join(truncated_tokens) + "..."
    
    def search_similar_chunks(self, query: str, language: str, limit: int = 5) -> List[Dict]:
        """Search for chunks similar to query text"""
        chunks = self.arabic_chunks if language == "ar" else self.english_chunks
        normalized_query = self.normalize_text(query, language)
        
        # Simple keyword-based search
        query_tokens = set(normalized_query.split())
        
        results = []
        for chunk in chunks:
            chunk_text = self.normalize_text(chunk.get("text", ""), language)
            chunk_tokens = set(chunk_text.split())
            
            # Calculate overlap ratio
            if query_tokens:
                overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
                if overlap > 0.1:  # Minimum overlap threshold
                    results.append({
                        "chunk": chunk,
                        "score": overlap
                    })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return [r["chunk"] for r in results[:limit]]
    
    def get_qa_pairs_by_chunk(self, chunk_id: int, language: str) -> List[Dict]:
        """Get QA pairs associated with a specific chunk"""
        qa_pairs = self.arabic_qa_pairs if language == "ar" else self.english_qa_pairs
        
        return [qa for qa in qa_pairs if qa.get("chunk_id") == chunk_id]
    
    def validate_data_integrity(self) -> Dict[str, bool]:
        """Validate that all required data is loaded and consistent"""
        checks = {
            "arabic_chunks_loaded": bool(self.arabic_chunks),
            "english_chunks_loaded": bool(self.english_chunks),
            "arabic_qa_loaded": bool(self.arabic_qa_pairs),
            "english_qa_loaded": bool(self.english_qa_pairs),
            "arabic_text_loaded": bool(self.arabic_text),
            "english_text_loaded": bool(self.english_text)
        }
        
        # Check chunk ID consistency
        if self.arabic_chunks:
            chunk_ids = [chunk.get("id") for chunk in self.arabic_chunks]
            checks["arabic_chunk_ids_sequential"] = chunk_ids == list(range(len(chunk_ids)))
            
        if self.english_chunks:
            chunk_ids = [chunk.get("id") for chunk in self.english_chunks]
            checks["english_chunk_ids_sequential"] = chunk_ids == list(range(len(chunk_ids)))
        
        # Check QA-chunk relationships
        if self.arabic_qa_pairs and self.arabic_chunks:
            max_chunk_id = max(chunk.get("id", -1) for chunk in self.arabic_chunks)
            qa_chunk_ids = [qa.get("chunk_id", -1) for qa in self.arabic_qa_pairs]
            checks["arabic_qa_valid_chunk_refs"] = all(0 <= cid <= max_chunk_id for cid in qa_chunk_ids)
            
        if self.english_qa_pairs and self.english_chunks:
            max_chunk_id = max(chunk.get("id", -1) for chunk in self.english_chunks)
            qa_chunk_ids = [qa.get("chunk_id", -1) for qa in self.english_qa_pairs]
            checks["english_qa_valid_chunk_refs"] = all(0 <= cid <= max_chunk_id for cid in qa_chunk_ids)
        
        return checks
