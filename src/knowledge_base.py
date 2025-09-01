import json
import re
from typing import Dict, List, Optional
from src.data_processor import DataProcessor


class AAOIFIKnowledgeBase:
    """Knowledge base for querying AAOIFI standards"""

    def __init__(self):
        self.processor = None
        self.index = {"ar": {}, "en": {}}

    def load_data(self, processor: DataProcessor):
        """Load data from processor and build index"""
        self.processor = processor
        self._build_search_index()

    def _build_search_index(self):
        """Build search index for both languages"""
        if not self.processor:
            return

        # Index Arabic chunks
        for chunk in self.processor.arabic_chunks:
            chunk_id = chunk.get("id")
            text = self.processor.normalize_text(chunk.get("text", ""), "ar")
            keywords = self._extract_keywords(text, "ar")

            for keyword in keywords:
                if keyword not in self.index["ar"]:
                    self.index["ar"][keyword] = []
                self.index["ar"][keyword].append(chunk_id)

        # Index English chunks
        for chunk in self.processor.english_chunks:
            chunk_id = chunk.get("id")
            text = self.processor.normalize_text(chunk.get("text", ""), "en")
            keywords = self._extract_keywords(text, "en")

            for keyword in keywords:
                if keyword not in self.index["en"]:
                    self.index["en"][keyword] = []
                self.index["en"][keyword].append(chunk_id)

    def _extract_keywords(self, text: str, language: str) -> List[str]:
        """Extract keywords from text for indexing"""
        if language == "ar":
            # Arabic keywords
            keywords = []

            # Standard references
            standards = re.findall(r'المعيار الشرعي رقم \(\d+\)', text)
            keywords.extend(standards)

            # Common Islamic finance terms
            islamic_terms = [
                'الربا', 'الغرر', 'الميسر', 'الحلال', 'الحرام', 'المرابحة',
                'المضاربة', 'الإجارة', 'السلم', 'الاستصناع', 'المشاركة',
                'الصكوك', 'الضمان', 'العملات', 'البطاقة', 'الائتمان',
                'التمويل', 'الاستثمار'
            ]

            for term in islamic_terms:
                if term in text:
                    keywords.append(term)

            # Extract other significant words (length > 3)
            words = text.split()
            for word in words:
                if len(word) > 3 and word.isalpha():
                    keywords.append(word)

        else:
            # English keywords
            keywords = []

            # Standard references
            standards = re.findall(r"Shari'ah Standard No\. \(\d+\)", text)
            keywords.extend(standards)

            # Islamic finance terms
            islamic_terms = [
                'riba', 'gharar', 'halal', 'haram', 'murabahah', 'mudarabah',
                'ijarah', 'salam', 'istisna', 'musharakah', 'sukuk',
                'guarantee', 'currency', 'card', 'credit', 'financing',
                'investment', 'islamic', 'sharia', 'shariah', 'permissible',
                'prohibited'
            ]

            text_lower = text.lower()
            for term in islamic_terms:
                if term in text_lower:
                    keywords.append(term)

            # Extract other significant words
            words = re.findall(r'\b\w{4,}\b', text_lower)
            keywords.extend(words)

        return list(set(keywords))  # Remove duplicates

    def search(self, query: str, language: str, limit: int = 10) -> List[Dict]:
        """Search the knowledge base"""
        if not self.processor:
            return []

        # Normalize query
        normalized_query = self.processor.normalize_text(query, language)
        query_keywords = self._extract_keywords(normalized_query, language)

        if not query_keywords:
            return []

        # Score chunks based on keyword matches
        chunk_scores = {}

        for keyword in query_keywords:
            if keyword in self.index[language]:
                for chunk_id in self.index[language][keyword]:
                    if chunk_id not in chunk_scores:
                        chunk_scores[chunk_id] = 0
                    chunk_scores[chunk_id] += 1

        # Get top scoring chunks
        sorted_chunks = sorted(chunk_scores.items(),
                               key=lambda x: x[1],
                               reverse=True)
        top_chunk_ids = [chunk_id for chunk_id, _ in sorted_chunks[:limit]]

        # Prepare results
        results = []

        for chunk_id in top_chunk_ids:
            chunk = self.processor.get_chunk_by_id(chunk_id, language)
            if chunk:
                # Calculate confidence based on keyword matches
                confidence = chunk_scores[chunk_id] / len(query_keywords)
                confidence = min(confidence, 1.0)  # Cap at 1.0

                # Try to find reference
                reference = self._find_reference_in_chunk(
                    chunk, query, language)

                results.append({
                    "chunk_id":
                    chunk_id,
                    "content":
                    chunk.get("text", ""),
                    "confidence":
                    confidence,
                    "word_count":
                    chunk.get("word_count", 0),
                    "reference":
                    reference,
                    "standard":
                    self._extract_standard_number(chunk.get("text", ""),
                                                  language)
                })

        return results

    def _find_reference_in_chunk(self, chunk: Dict, query: str,
                                 language: str) -> Optional[str]:
        """Find the most relevant reference within a chunk"""
        text = chunk.get("text", "")

        # Look for standard references
        if language == "ar":
            standards = re.findall(r'المعيار الشرعي رقم \(\d+\)', text)
            if standards:
                return standards[0]

            # Look for clause references
            clauses = re.findall(r'البند \d+/\d+/?\d*', text)
            if clauses:
                return clauses[0]
        else:
            standards = re.findall(r"Shari'ah Standard No\. \(\d+\)", text)
            if standards:
                return standards[0]

            # Look for clause references
            clauses = re.findall(r'Clause \d+/\d+/?\d*', text)
            if clauses:
                return clauses[0]

        return None

    def _extract_standard_number(self, text: str,
                                 language: str) -> Optional[str]:
        """Extract standard number from text"""
        if language == "ar":
            match = re.search(r'المعيار الشرعي رقم \((\d+)\)', text)
            if match:
                return f"المعيار رقم {match.group(1)}"
        else:
            match = re.search(r"Shari'ah Standard No\. \((\d+)\)", text)
            if match:
                return f"Standard No. {match.group(1)}"

        return None

    def get_standard_overview(self, standard_number: int,
                              language: str) -> Optional[Dict]:
        """Get overview of a specific standard"""
        if not self.processor:
            return None
        chunks = self.processor.arabic_chunks if language == "ar" else self.processor.english_chunks

        pattern = f"المعيار الشرعي رقم \\({standard_number}\\)" if language == "ar" else f"Shari'ah Standard No\\. \\({standard_number}\\)"

        relevant_chunks = []
        for chunk in chunks:
            if re.search(pattern, chunk.get("text", "")):
                relevant_chunks.append(chunk)

        if not relevant_chunks:
            return None

        # Combine text from relevant chunks
        combined_text = "\n\n".join(
            [chunk.get("text", "") for chunk in relevant_chunks])

        return {
            "standard_number":
            standard_number,
            "language":
            language,
            "chunk_count":
            len(relevant_chunks),
            "total_words":
            sum(chunk.get("word_count", 0) for chunk in relevant_chunks),
            "content":
            combined_text[:2000] +
            "..." if len(combined_text) > 2000 else combined_text,
            "chunk_ids": [chunk.get("id") for chunk in relevant_chunks]
        }

    def list_available_standards(self, language: str) -> List[Dict]:
        """List all available standards"""
        if not self.processor:
            return []
        chunks = self.processor.arabic_chunks if language == "ar" else self.processor.english_chunks

        pattern = r"المعيار الشرعي رقم \((\d+)\)" if language == "ar" else r"Shari'ah Standard No\. \((\d+)\)"

        standards = {}

        for chunk in chunks:
            text = chunk.get("text", "")
            matches = re.findall(pattern, text)

            for match in matches:
                standard_num = int(match)
                if standard_num not in standards:
                    standards[standard_num] = {
                        "number": standard_num,
                        "chunk_ids": [],
                        "total_words": 0
                    }

                standards[standard_num]["chunk_ids"].append(chunk.get("id"))
                standards[standard_num]["total_words"] += chunk.get(
                    "word_count", 0)

        # Sort by standard number
        return [standards[num] for num in sorted(standards.keys())]
