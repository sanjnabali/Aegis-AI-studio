"""
Response Formatting Utilities
==============================
Formats responses for different contexts (voice, text, code, etc.)
"""

import re
from typing import Dict, Any, List
from datetime import datetime


class ResponseFormatter:
    """Formats responses for optimal presentation"""
    
    @staticmethod
    def format_for_voice(text: str, max_length: int = 500) -> str:
        """
        Format response for voice output.
        
        - Removes markdown formatting
        - Shortens sentences
        - Removes code blocks
        - Limits length
        """
        
        # Remove markdown headers
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        
        # Remove markdown bold/italic
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '[code block omitted]', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove bullet points
        text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Normalize whitespace
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Truncate if too long
        if len(text) > max_length:
            # Try to break at sentence
            sentences = text[:max_length].split('. ')
            if len(sentences) > 1:
                text = '. '.join(sentences[:-1]) + '.'
            else:
                text = text[:max_length] + '...'
        
        return text.strip()
    
    @staticmethod
    def format_search_results(
        results: List[Dict[str, Any]],
        format_type: str = "markdown"
    ) -> str:
        """Format search results for display"""
        
        if not results:
            return "No results found."
        
        if "error" in results[0]:
            return f"Search error: {results[0]['error']}"
        
        if format_type == "markdown":
            formatted = "# Search Results\n\n"
            
            for result in results:
                formatted += f"## {result.get('position', '?')}. {result.get('title', 'No title')}\n"
                formatted += f"{result.get('snippet', 'No description')}\n"
                formatted += f"🔗 [{result.get('url', '')}]({result.get('url', '')})\n\n"
            
            return formatted
        
        elif format_type == "plain":
            formatted = "Search Results:\n\n"
            
            for result in results:
                formatted += f"{result.get('position', '?')}. {result.get('title', 'No title')}\n"
                formatted += f"{result.get('snippet', '')}\n"
                formatted += f"URL: {result.get('url', '')}\n\n"
            
            return formatted
        
        elif format_type == "json":
            import json
            return json.dumps(results, indent=2)
        
        return str(results)
    
    @staticmethod
    def format_code_output(
        code: str,
        output: str,
        error: str = None,
        language: str = "python"
    ) -> str:
        """Format code execution results"""
        
        formatted = f"```{language}\n{code}\n```\n\n"
        
        if error:
            formatted += "**Error:**\n```\n"
            formatted += error
            formatted += "\n```\n"
        elif output:
            formatted += "**Output:**\n```\n"
            formatted += output
            formatted += "\n```\n"
        else:
            formatted += "*No output*\n"
        
        return formatted
    
    @staticmethod
    def format_timestamp(dt: datetime = None, format_str: str = "human") -> str:
        """Format timestamp in various formats"""
        
        if dt is None:
            dt = datetime.utcnow()
        
        if format_str == "human":
            # Human-readable format
            return dt.strftime("%B %d, %Y at %I:%M %p UTC")
        elif format_str == "iso":
            # ISO 8601
            return dt.isoformat()
        elif format_str == "unix":
            # Unix timestamp
            return str(int(dt.timestamp()))
        else:
            return dt.strftime(format_str)
    
    @staticmethod
    def truncate_text(
        text: str,
        max_length: int,
        suffix: str = "..."
    ) -> str:
        """Truncate text intelligently"""
        
        if len(text) <= max_length:
            return text
        
        # Try to break at word boundary
        truncated = text[:max_length - len(suffix)]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # If we can save >80% of text
            truncated = truncated[:last_space]
        
        return truncated + suffix
    
    @staticmethod
    def highlight_code(code: str, language: str = "python") -> str:
        """Add syntax highlighting markers"""
        return f"```{language}\n{code}\n```"
    
    @staticmethod
    def create_table(
        headers: List[str],
        rows: List[List[Any]]
    ) -> str:
        """Create markdown table"""
        
        # Create header
        table = "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        # Add rows
        for row in rows:
            table += "| " + " | ".join(str(cell) for cell in row) + " |\n"
        
        return table
    
    @staticmethod
    def format_error_message(
        error: Exception,
        include_traceback: bool = False
    ) -> str:
        """Format error message for user display"""
        
        error_type = type(error).__name__
        error_msg = str(error)
        
        formatted = f"**Error:** {error_type}\n"
        formatted += f"**Message:** {error_msg}\n"
        
        if include_traceback:
            import traceback
            formatted += "\n**Traceback:**\n```\n"
            formatted += traceback.format_exc()
            formatted += "\n```\n"
        
        return formatted


class VoiceOptimizer:
    """Optimizes responses specifically for voice output"""
    
    @staticmethod
    def detect_voice_query(text: str) -> bool:
        """Detect if query is likely from voice input"""
        
        # Voice queries tend to be:
        # - Shorter (< 50 words)
        # - Less formal punctuation
        # - Conversational tone
        
        word_count = len(text.split())
        
        if word_count > 50:
            return False
        
        # Check for voice indicators
        voice_indicators = [
            "hey", "hi", "hello", "um", "uh", "okay",
            "can you", "could you", "would you",
            "what's", "how's", "where's",
        ]
        
        text_lower = text.lower()
        has_indicator = any(indicator in text_lower for indicator in voice_indicators)
        
        # Check punctuation density
        punctuation_count = sum(1 for c in text if c in '.,!?;:')
        punctuation_ratio = punctuation_count / len(text) if text else 0
        
        # Voice input typically has less punctuation
        has_low_punctuation = punctuation_ratio < 0.05
        
        return has_indicator or (word_count < 30 and has_low_punctuation)
    
    @staticmethod
    def optimize_for_tts(text: str) -> str:
        """
        Optimize text for text-to-speech output.
        
        - Expands abbreviations
        - Spells out symbols
        - Adjusts numbers
        """
        
        # Expand common abbreviations
        abbreviations = {
            r'\bDr\.': 'Doctor',
            r'\bMr\.': 'Mister',
            r'\bMrs\.': 'Missus',
            r'\bMs\.': 'Miss',
            r'\betc\.': 'etcetera',
            r'\be\.g\.': 'for example',
            r'\bi\.e\.': 'that is',
            r'\bvs\.': 'versus',
        }
        
        for abbrev, expansion in abbreviations.items():
            text = re.sub(abbrev, expansion, text, flags=re.IGNORECASE)
        
        # Spell out symbols
        symbols = {
            '&': 'and',
            '@': 'at',
            '#': 'number',
            '%': 'percent',
            '$': 'dollars',
            '€': 'euros',
            '£': 'pounds',
            '+': 'plus',
            '=': 'equals',
            '<': 'less than',
            '>': 'greater than',
        }
        
        for symbol, word in symbols.items():
            text = text.replace(symbol, f' {word} ')
        
        # Format URLs for speaking
        text = re.sub(
            r'https?://([^\s]+)',
            r'\1',
            text
        )
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()