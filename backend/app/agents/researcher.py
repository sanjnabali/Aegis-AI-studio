"""
Deep Research Agent
===================
Combines web search + scraping for comprehensive research.
Performs multi-step analysis with source verification.
"""

from typing import List, Dict, Any
import asyncio

from loguru import logger

from app.agents.web_search import web_search_agent
from app.tools.web_scraper import web_scraper


class ResearchAgent:
    """
    Performs deep research by:
    1. Searching the web for relevant sources
    2. Scraping top results for detailed content
    3. Synthesizing findings into a comprehensive report
    """
    
    def __init__(self):
        self.max_sources = 3
        self.research_history = []
    
    async def research(
        self,
        query: str,
        num_sources: int = None,
        include_news: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive research on a topic.
        
        Args:
            query: Research question or topic
            num_sources: Number of sources to analyze (default: 3)
            include_news: Whether to include news search
        
        Returns:
            Dict with search results, scraped content, and summary
        """
        
        if num_sources is None:
            num_sources = self.max_sources
        
        logger.info(f"Starting research: '{query}'")
        
        research_result = {
            "query": query,
            "search_results": [],
            "scraped_content": [],
            "news_results": [],
            "status": "in_progress",
        }
        
        try:
            # Step 1: Web search
            logger.info("Step 1/3: Searching web...")
            search_results = await web_search_agent.search(
                query,
                max_results=num_sources + 2,  # Get extra in case some fail
            )
            research_result["search_results"] = search_results
            
            # Step 2: Search news (if requested)
            if include_news:
                logger.info("Step 2a/3: Searching news...")
                news_results = await web_search_agent.search_news(
                    query,
                    max_results=3,
                )
                research_result["news_results"] = news_results
            
            # Step 3: Scrape top sources
            logger.info(f"Step 3/3: Scraping {num_sources} sources...")
            
            # Extract URLs from search results
            urls = [
                result["url"]
                for result in search_results
                if "url" in result and "error" not in result
            ][:num_sources]
            
            # Scrape concurrently
            scrape_tasks = [
                web_scraper.scrape(url)
                for url in urls
            ]
            
            scraped_results = await asyncio.gather(*scrape_tasks)
            research_result["scraped_content"] = scraped_results
            
            # Count successful scrapes
            successful_scrapes = sum(
                1 for result in scraped_results
                if result.get("success")
            )
            
            research_result["status"] = "complete"
            research_result["sources_analyzed"] = successful_scrapes
            
            logger.success(
                f"Research complete: {successful_scrapes}/{num_sources} sources analyzed"
            )
            
            # Store in history
            self.research_history.append({
                "query": query,
                "sources_count": successful_scrapes,
                "timestamp": logger._core.handlers[0]._sink._file.name if logger._core.handlers else None,
            })
            
            return research_result
            
        except Exception as e:
            logger.error(f"Research error: {e}")
            research_result["status"] = "error"
            research_result["error"] = str(e)
            return research_result
    
    def format_research_report(self, research_result: Dict[str, Any]) -> str:
        """
        Format research results into a comprehensive report for LLM.
        """
        
        if research_result.get("status") == "error":
            return f"Research failed: {research_result.get('error')}"
        
        report = f"# Research Report: {research_result['query']}\n\n"
        
        # Add search overview
        report += "## Search Overview\n\n"
        search_results = research_result.get("search_results", [])
        
        if search_results and "error" not in search_results[0]:
            for result in search_results:
                report += f"**{result['position']}. {result['title']}**\n"
                report += f"{result['snippet']}\n"
                report += f"{result['url']}\n\n"
        else:
            report += "No search results available.\n\n"
        
        # Add news (if available)
        news_results = research_result.get("news_results", [])
        if news_results and "error" not in news_results[0]:
            report += "## Recent News\n\n"
            for article in news_results:
                report += f"**{article['title']}**\n"
                report += f"{article['snippet']}\n"
                report += f"Source: {article.get('source', 'Unknown')} | {article.get('date', '')}\n"
                report += f"{article['url']}\n\n"
        
        # Add detailed content from scraped sources
        report += "## Detailed Analysis\n\n"
        scraped_content = research_result.get("scraped_content", [])
        
        for idx, content in enumerate(scraped_content, 1):
            if content.get("success"):
                report += f"### Source {idx}: {content['title']}\n\n"
                report += f"**URL:** {content['url']}\n\n"
                report += f"{content['text']}\n\n"
                report += "---\n\n"
            else:
                report += f"### Source {idx}: Failed to load\n"
                report += f"Error: {content.get('error')}\n\n"
        
        return report
    
    async def quick_research(self, query: str) -> str:
        """
        Quick research with immediate formatting.
        
        Returns formatted report string ready for LLM.
        """
        
        research_result = await self.research(query, num_sources=2)
        return self.format_research_report(research_result)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get recent research history"""
        return self.research_history[-5:]


# Global researcher instance
research_agent = ResearchAgent()