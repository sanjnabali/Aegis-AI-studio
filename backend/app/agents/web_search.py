"""
Web Search Agent
================
Handles research queries using DuckDuckGo (free, no API key needed).
Includes smart result filtering and summarization.
"""

from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from loguru import logger

try:
    from duckduckgo_search import DDGS
except ImportError:
    logger.warning("duckduckgo_search not installed. Web search disabled.")
    DDGS = None


class WebSearchAgent:
    """
    Free-tier web search agent using DuckDuckGo.
    No API key required, unlimited searches.
    """
    
    def __init__(self):
        self.search_history = []
        self.max_results = 5
    
    async def search(
        self,
        query: str,
        max_results: int = None,
        region: str = "wt-wt",  # wt-wt = worldwide
    ) -> List[Dict[str, Any]]:
        """
        Search the web using DuckDuckGo.
        
        Args:
            query: Search query string
            max_results: Maximum number of results (default: 5)
            region: Search region (wt-wt = worldwide)
        
        Returns:
            List of search results with title, snippet, and URL
        """
        
        if DDGS is None:
            logger.error("DuckDuckGo search not available")
            return [{
                "error": "Web search not configured. Install: pip install duckduckgo-search"
            }]
        
        if max_results is None:
            max_results = self.max_results
        
        logger.info(f"Searching: '{query}' (max_results={max_results})")
        
        try:
            results = []
            
            # Perform search (runs in thread pool since DDGS is synchronous)
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None,
                self._sync_search,
                query,
                max_results,
                region,
            )
            
            # Format results
            for idx, result in enumerate(search_results, 1):
                results.append({
                    "position": idx,
                    "title": result.get("title", "No title"),
                    "snippet": result.get("body", "No description"),
                    "url": result.get("href", ""),
                    "timestamp": datetime.utcnow().isoformat(),
                })
            
            # Store in history
            self.search_history.append({
                "query": query,
                "results_count": len(results),
                "timestamp": datetime.utcnow().isoformat(),
            })
            
            logger.success(f"✓ Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return [{
                "error": str(e),
                "query": query,
            }]
    
    def _sync_search(self, query: str, max_results: int, region: str) -> List[Dict]:
        """Synchronous search helper"""
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results, region=region))
    
    async def search_news(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for news articles.
        
        Args:
            query: News search query
            max_results: Maximum results
        
        Returns:
            List of news results
        """
        
        if DDGS is None:
            return [{"error": "DuckDuckGo not available"}]
        
        logger.info(f"📰 News search: '{query}'")
        
        try:
            loop = asyncio.get_event_loop()
            news_results = await loop.run_in_executor(
                None,
                self._sync_news_search,
                query,
                max_results,
            )
            
            results = []
            for idx, article in enumerate(news_results, 1):
                results.append({
                    "position": idx,
                    "title": article.get("title", ""),
                    "snippet": article.get("body", ""),
                    "url": article.get("url", ""),
                    "source": article.get("source", "Unknown"),
                    "date": article.get("date", ""),
                })
            
            logger.success(f"Found {len(results)} news articles")
            return results
            
        except Exception as e:
            logger.error(f"News search error: {e}")
            return [{"error": str(e)}]
    
    def _sync_news_search(self, query: str, max_results: int) -> List[Dict]:
        """Synchronous news search helper"""
        with DDGS() as ddgs:
            return list(ddgs.news(query, max_results=max_results))
    
    def format_results_for_llm(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results for LLM consumption.
        Creates a clean, structured summary.
        """
        
        if not results:
            return "No search results found."
        
        if "error" in results[0]:
            return f"Search error: {results[0]['error']}"
        
        formatted = "# Web Search Results\n\n"
        
        for result in results:
            formatted += f"## {result['position']}. {result['title']}\n"
            formatted += f"{result['snippet']}\n"
            formatted += f"Source: {result['url']}\n\n"
        
        return formatted
    
    async def quick_answer(self, query: str) -> Optional[str]:
        """
        Get a quick answer/instant answer for simple queries.
        
        Examples: "weather in London", "2+2", "who is X"
        """
        
        if DDGS is None:
            return None
        
        try:
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                None,
                self._sync_instant_answer,
                query,
            )
            
            if answer:
                logger.info(f"✓ Quick answer found")
                return answer
            
        except Exception as e:
            logger.error(f"Quick answer error: {e}")
        
        return None
    
    def _sync_instant_answer(self, query: str) -> Optional[str]:
        """Get instant answer synchronously"""
        try:
            with DDGS() as ddgs:
                results = ddgs.answers(query)
                if results:
                    return results[0].get("text")
        except:
            pass
        return None
    
    def get_search_history(self) -> List[Dict[str, Any]]:
        """Get recent search history"""
        return self.search_history[-10:]  # Last 10 searches


# Global agent instance
web_search_agent = WebSearchAgent()