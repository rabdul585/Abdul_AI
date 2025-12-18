import os
from typing import Dict, Any
from serpapi import GoogleSearch

class WebSearchAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def search(self, query: str, category: str = "Unknown") -> str:
        """
        Performs a web search for the given query using SerpApi.
        """
        if not self.api_key:
            return "[Web Search Error] SerpApi Key not found. Please configure SERPAPI_API_KEY in .env."

        if category and category != "Unknown":
            search_query = f"{category} issue: {query}"
        else:
            search_query = query
        
        try:
            params = {
                "q": search_query,
                "api_key": self.api_key,
                "num": 3 # Limit results
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # Debugging: Print raw results to see why it might be empty
            if "error" in results:
                print(f"SerpApi Error: {results['error']}")
                return f"[Web Search Error] {results['error']}"
                
            organic_results = results.get("organic_results", [])
            
            if not organic_results:
                print(f"SerpApi returned no organic results. Raw keys: {results.keys()}")
                # Fallback to 'answer_box' or 'snippet' if organic is missing but others exist
                if "answer_box" in results:
                     return f"[Web Search Result] {results['answer_box'].get('snippet', results['answer_box'].get('title'))}"
                
                return f"I searched the web for '{query}' but couldn't find specific details. Please try rephrasing or escalating the ticket."
            
            summary = []
            for result in organic_results:
                title = result.get("title")
                snippet = result.get("snippet")
                link = result.get("link")
                summary.append(f"- **[{title}]({link})**: {snippet}")
            
            return f"[Web Search Result] Found the following info:\n\n" + "\n\n".join(summary)
            
        except Exception as e:
            return f"[Web Search Error] Failed to search web: {str(e)}"
