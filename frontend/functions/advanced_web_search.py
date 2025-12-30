"""
title: Advanced Web Search & Research Agent
author: Aegis Studio
version: 2.0.0
description: Complete web search and research capabilities with caching and analytics
required_open_webui_version: 0.3.0
"""

from pydantic import BaseModel, Field
from typing import Optional, Callable, Any, List, Dict
import json
import re
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

class Tools:
    class Valves(BaseModel):
        """Configuration parameters"""
        MAX_SEARCH_RESULTS: int = Field(
            default=10,
            description="Maximum search results to return"
        )
        MAX_SCRAPE_LENGTH: int = Field(
            default=5000,
            description="Maximum characters to extract from webpage"
        )
        ENABLE_AUTO_RESEARCH: bool = Field(
            default=True,
            description="Automatically research complex topics"
        )
        ENABLE_CACHING: bool = Field(
            default=True,
            description="Cache search results"
        )
        USER_AGENT: str = Field(
            default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            description="User agent for web requests"
        )
        TIMEOUT: int = Field(
            default=10,
            description="Request timeout in seconds"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.cache = {}
        self.stats = {
            "total_searches": 0,
            "total_scrapes": 0,
            "cache_hits": 0,
            "errors": 0,
        }

    # ========================================================================
    # CORE SEARCH FUNCTIONS
    # ========================================================================

    def web_search(
        self,
        query: str,
        max_results: Optional[int] = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Search the web using DuckDuckGo.
        
        Args:
            query: Search query
            max_results: Number of results (overrides default)
            
        Returns:
            Formatted search results
        """
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return self._error_response(
                "DuckDuckGo library not installed",
                "Run: pip install duckduckgo-search"
            )
        
        if not query or not query.strip():
            return self._error_response("Empty query", "Please provide a search query")
        
        # Emit searching status
        if __event_emitter__:
            __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"Searching web: {query}",
                    "done": False
                }
            })
        
        # Check cache
        cache_key = f"search:{query}:{max_results or self.valves.MAX_SEARCH_RESULTS}"
        if self.valves.ENABLE_CACHING and cache_key in self.cache:
            self.stats["cache_hits"] += 1
            if __event_emitter__:
                __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "Retrieved from cache",
                        "done": True
                    }
                })
            return self.cache[cache_key]
        
        try:
            results = []
            max_res = max_results or self.valves.MAX_SEARCH_RESULTS
            
            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, max_results=max_res))
                
                for idx, result in enumerate(search_results, 1):
                    results.append({
                        "position": idx,
                        "title": result.get("title", "No title"),
                        "snippet": result.get("body", "No description"),
                        "url": result.get("href", ""),
                    })
            
            # Format results
            formatted = self._format_search_results(results, query)
            
            # Cache results
            if self.valves.ENABLE_CACHING:
                self.cache[cache_key] = formatted
            
            # Update stats
            self.stats["total_searches"] += 1
            
            # Emit completion
            if __event_emitter__:
                __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"Found {len(results)} results",
                        "done": True
                    }
                })
            
            return formatted
        
        except Exception as e:
            self.stats["errors"] += 1
            return self._error_response("Search failed", str(e))

    def scrape_url(
        self,
        url: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Scrape content from a URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            Extracted content
        """
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return self._error_response(
                "Dependencies missing",
                "Run: pip install requests beautifulsoup4"
            )
        
        if not self._is_valid_url(url):
            return self._error_response("Invalid URL", f"URL appears invalid: {url}")
        
        # Emit scraping status
        if __event_emitter__:
            __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"Fetching {url}",
                    "done": False
                }
            })
        
        # Check cache
        cache_key = f"scrape:{url}"
        if self.valves.ENABLE_CACHING and cache_key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[cache_key]
        
        try:
            headers = {"User-Agent": self.valves.USER_AGENT}
            response = requests.get(
                url,
                headers=headers,
                timeout=self.valves.TIMEOUT
            )
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extract text
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean text
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = '\n'.join(lines)
            
            # Truncate if needed
            if len(text) > self.valves.MAX_SCRAPE_LENGTH:
                text = text[:self.valves.MAX_SCRAPE_LENGTH] + "\n\n[Content truncated...]"
            
            # Format result
            title = soup.title.string if soup.title else "Unknown"
            formatted = f"# {title}\n\n**Source:** {url}\n\n{text}"
            
            # Cache result
            if self.valves.ENABLE_CACHING:
                self.cache[cache_key] = formatted
            
            # Update stats
            self.stats["total_scrapes"] += 1
            
            # Emit completion
            if __event_emitter__:
                __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"Scraped {len(text)} characters",
                        "done": True
                    }
                })
            
            return formatted
        
        except Exception as e:
            self.stats["errors"] += 1
            return self._error_response("Scraping failed", str(e))

    def research_topic(
        self,
        topic: str,
        num_sources: int = 3,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Perform comprehensive research on a topic.
        
        Args:
            topic: Research topic
            num_sources: Number of sources to analyze
            
        Returns:
            Comprehensive research report
        """
        if __event_emitter__:
            __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"🔬 Researching: {topic}",
                    "done": False
                }
            })
        
        # Step 1: Search
        search_results = self.web_search(topic, max_results=num_sources + 2)
        
        # Extract URLs
        urls = self._extract_urls(search_results)[:num_sources]
        
        # Step 2: Scrape sources
        report = f"# Research Report: {topic}\n\n"
        report += "## Search Results\n\n" + search_results + "\n\n"
        report += "## Detailed Analysis\n\n"
        
        for idx, url in enumerate(urls, 1):
            if __event_emitter__:
                __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"📖 Analyzing source {idx}/{len(urls)}",
                        "done": False
                    }
                })
            
            content = self.scrape_url(url)
            report += f"### Source {idx}\n\n{content}\n\n---\n\n"
        
        if __event_emitter__:
            __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"Research complete! Analyzed {len(urls)} sources",
                    "done": True
                }
            })
        
        return report

    def get_stats(self) -> str:
        """Get usage statistics"""
        stats_str = "# Web Search Statistics\n\n"
        stats_str += f"**Total Searches:** {self.stats['total_searches']}\n"
        stats_str += f"**Total Scrapes:** {self.stats['total_scrapes']}\n"
        stats_str += f"**Cache Hits:** {self.stats['cache_hits']}\n"
        stats_str += f"**Errors:** {self.stats['errors']}\n"
        stats_str += f"**Cache Size:** {len(self.cache)} entries\n"
        
        return stats_str

    def clear_cache(self) -> str:
        """Clear the search cache"""
        cache_size = len(self.cache)
        self.cache.clear()
        return f"Cleared {cache_size} cached entries"

    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================

    def _format_search_results(self, results: List[Dict], query: str) -> str:
        """Format search results as markdown"""
        if not results:
            return f"No results found for: {query}"
        
        formatted = f"# Search Results: {query}\n\n"
        formatted += f"*Found {len(results)} results*\n\n"
        
        for result in results:
            formatted += f"## {result['position']}. {result['title']}\n"
            formatted += f"{result['snippet']}\n"
            formatted += f"{result['url']}\n\n"
        
        return formatted

    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'(https?://[^\s]+)'
        return re.findall(url_pattern, text)

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return url_pattern.match(url) is not None

    def _error_response(self, error_type: str, message: str) -> str:
        """Format error response"""
        return f"**Error:** {error_type}\n\n{message}"