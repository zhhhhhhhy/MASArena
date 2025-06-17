# coding: utf-8
import os
import json
from typing import List, Dict, Any

from langchain.tools import StructuredTool
from dotenv import load_dotenv

from benchmark.src.tools.base import ToolFactory

# Try to import TavilyClient and handle ImportError
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

SEARCH_API = "tavily_search"

class TavilySearch:
    def __init__(self):
        if not TAVILY_AVAILABLE:
            self.client = None
            return
            
        api_key = os.getenv("TAVILY_API_KEY")
        if api_key:
            self.client = TavilyClient(api_key=api_key)
        else:
            self.client = None

    def search(self, query: str) -> str:
        """Perform a web search using the Tavily API and return a summary of results."""
        if self.client is None:
            if not TAVILY_AVAILABLE:
                return "Error: tavily-python is not installed. Please install it with `pip install tavily-python`."
            return "Error: TAVILY_API_KEY is not configured in .env file."

        try:
            # Perform the search
            response = self.client.search(query=query, search_depth="advanced")
            
            # Extract and format the results
            results = response.get('results', [])
            summary = []
            for res in results:
                summary.append(f"Title: {res.get('title')}\nURL: {res.get('url')}\nContent: {res.get('content')}\n---")
            
            return "\n".join(summary) if summary else "No results found."

        except Exception as e:
            return f"An unexpected error occurred during search: {e}"


@ToolFactory.register(name=SEARCH_API, desc="A tool for searching the web using the Tavily API.")
class SearchApiTool:
    def __init__(self):
        self.search_api = TavilySearch()

    def get_tools(self) -> List[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.search_api.search,
                name="tavily_search",
                description="Performs a web search for a given query using Tavily API and returns detailed results.",
            )
        ] 