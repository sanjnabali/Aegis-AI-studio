"""
Web Content Scraper
===================
Extracts clean text content from web pages.
Used in conjunction with web search for detailed analysis.
"""

import asyncio
from typing import Optional, Dict, Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from loguru import logger


class WebScraper:
    """
    Extracts and cleans text content from web pages.
    Handles various content types and error cases.
    """
    
    def __init__(self):
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        self.max_content_length = 10000  # characters
        self.timeout = 10  # seconds
    
    async def scrape(self, url: str) -> Dict[str, Any]:
        """
        Scrape and extract text from a URL.
        
        Args:
            url: URL to scrape
        
        Returns:
            Dict with title, text, metadata, and status
        """
        
        logger.info(f"📄 Scraping: {url}")
        
        # Validate URL
        if not self._is_valid_url(url):
            return {
                "success": False,
                "error": "Invalid URL format",
                "url": url,
            }
        
        try:
            # Fetch page
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": self.user_agent},
                    timeout=self.timeout,
                    follow_redirects=True,
                )
                
                response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup)
            
            # Extract text
            text = self._extract_text(soup)
            
            # Truncate if too long
            if len(text) > self.max_content_length:
                text = text[:self.max_content_length] + "\n\n[Content truncated...]"
            
            # Extract metadata
            metadata = self._extract_metadata(soup, url)
            
            logger.success(f"✓ Scraped {len(text)} characters")
            
            return {
                "success": True,
                "url": url,
                "title": title,
                "text": text,
                "metadata": metadata,
                "length": len(text),
            }
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {url}")
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}",
                "url": url,
            }
        
        except httpx.TimeoutException:
            logger.error(f"Timeout: {url}")
            return {
                "success": False,
                "error": "Request timeout",
                "url": url,
            }
        
        except Exception as e:
            logger.error(f"Scraping error: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url,
            }
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        
        # Try <title> tag
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        
        # Try <h1>
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
        
        return "No title"
    
    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text content"""
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 
                            'header', 'aside', 'iframe', 'noscript']):
            element.decompose()
        
        # Get text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        # Remove duplicate lines (common in navs/footers)
        seen = set()
        unique_lines = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        
        return '\n'.join(unique_lines)
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from page"""
        
        metadata = {
            "domain": urlparse(url).netloc,
        }
        
        # Try to get description
        desc_tag = soup.find('meta', attrs={'name': 'description'})
        if desc_tag:
            metadata['description'] = desc_tag.get('content', '')
        
        # Try to get author
        author_tag = soup.find('meta', attrs={'name': 'author'})
        if author_tag:
            metadata['author'] = author_tag.get('content', '')
        
        return metadata
    
    def format_for_llm(self, scrape_result: Dict[str, Any]) -> str:
        """Format scraped content for LLM"""
        
        if not scrape_result.get("success"):
            return f"Failed to scrape {scrape_result.get('url')}: {scrape_result.get('error')}"
        
        formatted = f"# {scrape_result['title']}\n\n"
        formatted += f"**Source:** {scrape_result['url']}\n"
        
        if scrape_result.get('metadata', {}).get('description'):
            formatted += f"**Description:** {scrape_result['metadata']['description']}\n"
        
        formatted += f"\n## Content\n\n{scrape_result['text']}\n"
        
        return formatted


# Global scraper instance
web_scraper = WebScraper()