import os
from typing import List, Dict, Tuple
import openai
import re

class KnowledgeBaseAgent:
    def __init__(self, api_key: str, kb_file_path: str = "KBDocs/knowledge_base.txt"):
        self.client = openai.OpenAI(api_key=api_key)
        self.kb_data = self._load_kb(kb_file_path)

    def _load_kb(self, file_path: str) -> List[Dict[str, str]]:
        """
        Parses the KB text file into a structured list.
        """
        if not os.path.exists(file_path):
            print(f"KB file not found at {file_path}")
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split by the separator line
        raw_entries = content.split("========================================")
        kb_entries = []

        for entry in raw_entries:
            if not entry.strip():
                continue
            
            parsed_entry = {}
            lines = entry.strip().split("\n")
            
            current_key = None
            buffer = []
            
            for line in lines:
                if ":" in line and not line.startswith(" "): # Simple heuristic for keys
                    # Save previous key
                    if current_key:
                        parsed_entry[current_key] = "\n".join(buffer).strip()
                    
                    # Start new key
                    parts = line.split(":", 1)
                    current_key = parts[0].strip()
                    buffer = [parts[1].strip()] if len(parts) > 1 else []
                else:
                    buffer.append(line)
            
            # Save last key
            if current_key:
                parsed_entry[current_key] = "\n".join(buffer).strip()
            
            if "CATEGORY" in parsed_entry and "TROUBLESHOOTING STEPS" in parsed_entry:
                kb_entries.append(parsed_entry)

        return kb_entries

    def search(self, query: str, category: str = None) -> Tuple[str, float]:
        """
        Searches the knowledge base for the most relevant article.
        Returns (Answer, Confidence Score).
        """
        # Filter by category if provided
        candidates = self.kb_data
        if category and category != "Unknown":
            candidates = [doc for doc in self.kb_data if doc.get("CATEGORY") == category]
            
        if not candidates:
            candidates = self.kb_data

        best_match = None
        best_score = 0.0

        # Simple Jaccard similarity for mock purposes (word overlap)
        # In production, use embeddings here.
        query_words = set(query.lower().split())
        
        for doc in candidates:
            # Combine Title and Description for matching
            doc_text = (doc.get("TITLE", "") + " " + doc.get("DESCRIPTION", "")).lower()
            doc_words = set(doc_text.split())
            
            intersection = query_words.intersection(doc_words)
            union = query_words.union(doc_words)
            if not union:
                score = 0.0
            else:
                score = len(intersection) / len(union)
            
            if score > best_score:
                best_score = score
                best_match = doc

        # Threshold for "found"
        # Increased threshold to 0.2 to avoid false positives from common words like "to", "the"
        if best_score > 0.2 and best_match: 
            return f"**{best_match.get('TITLE')}**\n\n{best_match.get('TROUBLESHOOTING STEPS')}\n\n**Resolution:** {best_match.get('RESOLUTION')}", best_score
        
        return "No relevant KB article found.", 0.0
