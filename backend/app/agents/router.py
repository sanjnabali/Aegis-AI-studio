"""
Agent Router - Intelligent Task Routing
========================================
Determines which specialized agent should handle each request:
- Web Search Agent: For research, current events, fact-checking
- Code Agent: For programming, debugging, technical analysis
- Chat Agent: For general conversation
"""

from typing import Dict, Any, List
from loguru import logger

from app.core.config import get_settings

settings = get_settings()


class AgentRouter:
    """
    Intelligently routes requests to specialized agents based on:
    - Keywords in user query
    - Conversation context
    - Task complexity
    """
    
    def __init__(self):
        self.routing_stats = {
            "web_search": 0,
            "code": 0,
            "chat": 0,
        }
    
    def analyze_intent(self, messages: List[Dict[str, Any]]) -> str:
        """
        Analyze user intent from conversation messages.
        
        Returns:
            Agent type: "web_search", "code", or "chat"
        """
        
        if not messages:
            return "chat"
        
        # Get last user message
        last_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_message = msg.get("content", "")
                break
        
        if not last_message:
            return "chat"
        
        # Convert to lowercase for analysis
        query = last_message.lower()
        
        # Web Search Intent Detection
        web_search_triggers = [
            # Explicit search requests
            "search for", "look up", "find information", "google",
            "search the web", "what's the latest", "recent news",
            
            # Current information needs
            "today", "currently", "right now", "latest", "recent",
            "this week", "this month", "2024", "2025",
            
            # Fact-checking
            "is it true", "verify", "fact check", "according to",
            
            # Research queries
            "compare", "difference between", "best", "top",
            "reviews", "ratings", "recommendations",
            
            # Current events
            "news", "happened", "update on", "status of",
            "what's going on", "current situation",
        ]
        
        # Code Intent Detection
        code_triggers = [
            # Programming languages
            "python", "javascript", "java", "c++", "rust", "go",
            "typescript", "ruby", "php", "swift", "kotlin",
            
            # Code-related actions
            "code", "function", "class", "method", "algorithm",
            "debug", "error", "bug", "fix", "implement",
            "program", "script", "syntax", "compile",
            
            # Development concepts
            "api", "database", "sql", "query", "regex",
            "docker", "kubernetes", "git", "deploy",
            
            # Code analysis
            "refactor", "optimize", "review", "test",
            "performance", "benchmark",
        ]
        
        # Check for web search intent
        if any(trigger in query for trigger in web_search_triggers):
            logger.info("🔍 Routing to WEB SEARCH agent")
            self.routing_stats["web_search"] += 1
            return "web_search"
        
        # Check for code intent
        if any(trigger in query for trigger in code_triggers):
            logger.info("💻 Routing to CODE agent")
            self.routing_stats["code"] += 1
            return "code"
        
        # Check for code blocks in message
        if "```" in last_message or "def " in query or "function " in query:
            logger.info("💻 Routing to CODE agent (code detected)")
            self.routing_stats["code"] += 1
            return "code"
        
        # Default to chat
        logger.info("💬 Routing to CHAT agent")
        self.routing_stats["chat"] += 1
        return "chat"
    
    def should_use_web_search(self, messages: List[Dict[str, Any]]) -> bool:
        """Quick check if web search should be used"""
        return self.analyze_intent(messages) == "web_search"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        total = sum(self.routing_stats.values())
        
        return {
            "total_requests": total,
            "web_search": {
                "count": self.routing_stats["web_search"],
                "percentage": f"{self.routing_stats['web_search']/total*100:.1f}%" if total > 0 else "0%",
            },
            "code": {
                "count": self.routing_stats["code"],
                "percentage": f"{self.routing_stats['code']/total*100:.1f}%" if total > 0 else "0%",
            },
            "chat": {
                "count": self.routing_stats["chat"],
                "percentage": f"{self.routing_stats['chat']/total*100:.1f}%" if total > 0 else "0%",
            },
        }


# Global router instance
agent_router = AgentRouter()