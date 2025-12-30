"""
Tool Registry
=============
Central registry for all available tools and their metadata.
"""

from typing import Dict, Any, List, Callable
from enum import Enum


class ToolCategory(str, Enum):
    """Tool categories"""
    SEARCH = "search"
    SCRAPING = "scraping"
    CODE = "code"
    RESEARCH = "research"
    AUDIO = "audio"
    UTILITY = "utility"


class Tool:
    """Tool definition"""
    
    def __init__(
        self,
        name: str,
        category: ToolCategory,
        description: str,
        function: Callable,
        parameters: Dict[str, Any],
        enabled: bool = True,
    ):
        self.name = name
        self.category = category
        self.description = description
        self.function = function
        self.parameters = parameters
        self.enabled = enabled
        self.usage_count = 0
    
    async def execute(self, **kwargs):
        """Execute the tool"""
        self.usage_count += 1
        return await self.function(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "parameters": self.parameters,
            "enabled": self.enabled,
            "usage_count": self.usage_count,
        }


class ToolRegistry:
    """Central tool registry"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """Register a new tool"""
        self.tools[tool.name] = tool
    
    def get(self, name: str) -> Tool:
        """Get tool by name"""
        return self.tools.get(name)
    
    def list_tools(self, category: ToolCategory = None) -> List[Tool]:
        """List all tools, optionally filtered by category"""
        if category:
            return [t for t in self.tools.values() if t.category == category]
        return list(self.tools.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_tools": len(self.tools),
            "by_category": {
                cat.value: len([t for t in self.tools.values() if t.category == cat])
                for cat in ToolCategory
            },
            "total_usage": sum(t.usage_count for t in self.tools.values()),
        }


# Global registry
tool_registry = ToolRegistry()