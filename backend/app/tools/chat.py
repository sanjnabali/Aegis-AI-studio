# --- The "General Chat" tool (wrapper for Mistral-7B).
"""
Enhanced Chat Tool
==================
Manages chat conversations with context management and personality adaptation.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import deque

from loguru import logger


class ConversationManager:
    """
    Manages conversation history and context.
    
    Features:
    - Context window management
    - Conversation summarization
    - Memory pruning
    - Multi-turn tracking
    """
    
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.conversations: Dict[str, deque] = {}
        self.summaries: Dict[str, str] = {}
        self.metadata: Dict[str, Dict] = {}
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ):
        """Add a message to conversation history"""
        
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = deque(maxlen=self.max_history)
            self.metadata[conversation_id] = {
                "created_at": datetime.utcnow().isoformat(),
                "message_count": 0,
                "last_updated": None,
            }
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        
        self.conversations[conversation_id].append(message)
        self.metadata[conversation_id]["message_count"] += 1
        self.metadata[conversation_id]["last_updated"] = message["timestamp"]
        
        logger.debug(f"Added message to conversation {conversation_id}")
    
    def get_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        include_system: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get conversation history"""
        
        if conversation_id not in self.conversations:
            return []
        
        messages = list(self.conversations[conversation_id])
        
        if not include_system:
            messages = [m for m in messages if m["role"] != "system"]
        
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_context(
        self,
        conversation_id: str,
        max_tokens: int = 4000,
    ) -> List[Dict[str, str]]:
        """
        Get conversation context optimized for token limits.
        
        Uses intelligent pruning to fit within token budget.
        """
        
        messages = self.get_history(conversation_id)
        
        # Estimate tokens (rough: 4 chars = 1 token)
        formatted_messages = []
        total_chars = 0
        max_chars = max_tokens * 4
        
        # Always include the last message (user's current query)
        if messages:
            last_msg = messages[-1]
            formatted_messages.append({
                "role": last_msg["role"],
                "content": last_msg["content"],
            })
            total_chars += len(last_msg["content"])
        
        # Add previous messages in reverse order until we hit limit
        for msg in reversed(messages[:-1]):
            msg_chars = len(msg["content"])
            
            if total_chars + msg_chars > max_chars:
                # If we have a summary, include it
                if conversation_id in self.summaries:
                    formatted_messages.insert(0, {
                        "role": "system",
                        "content": f"Previous conversation summary: {self.summaries[conversation_id]}",
                    })
                break
            
            formatted_messages.insert(0, {
                "role": msg["role"],
                "content": msg["content"],
            })
            total_chars += msg_chars
        
        return formatted_messages
    
    def summarize_conversation(
        self,
        conversation_id: str,
        summary: str,
    ):
        """Store a conversation summary"""
        
        self.summaries[conversation_id] = summary
        logger.info(f"Stored summary for conversation {conversation_id}")
    
    def clear_conversation(self, conversation_id: str):
        """Clear a conversation history"""
        
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            del self.metadata[conversation_id]
            if conversation_id in self.summaries:
                del self.summaries[conversation_id]
            
            logger.info(f"Cleared conversation {conversation_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        
        total_messages = sum(
            meta["message_count"]
            for meta in self.metadata.values()
        )
        
        return {
            "total_conversations": len(self.conversations),
            "total_messages": total_messages,
            "active_conversations": len([
                c for c, meta in self.metadata.items()
                if meta["message_count"] > 0
            ]),
        }


class PersonalityAdapter:
    """
    Adapts response style based on user preferences and context.
    
    Personality modes:
    - Professional: Formal, concise, technical
    - Friendly: Conversational, warm, helpful
    - Educational: Detailed, patient, explanatory
    - Creative: Imaginative, expressive, artistic
    """
    
    PERSONALITIES = {
        "professional": {
            "tone": "formal and concise",
            "style": "technical and precise",
            "emojis": False,
            "examples": True,
        },
        "friendly": {
            "tone": "warm and conversational",
            "style": "helpful and approachable",
            "emojis": True,
            "examples": True,
        },
        "educational": {
            "tone": "patient and clear",
            "style": "detailed and explanatory",
            "emojis": False,
            "examples": True,
        },
        "creative": {
            "tone": "imaginative and expressive",
            "style": "artistic and engaging",
            "emojis": True,
            "examples": False,
        },
    }
    
    def __init__(self, default_personality: str = "friendly"):
        self.user_preferences: Dict[str, str] = {}
        self.default_personality = default_personality
    
    def set_personality(self, user_id: str, personality: str):
        """Set personality for a user"""
        
        if personality not in self.PERSONALITIES:
            logger.warning(f"Invalid personality '{personality}', using default")
            personality = self.default_personality
        
        self.user_preferences[user_id] = personality
        logger.info(f"Set personality for {user_id}: {personality}")
    
    def get_personality(self, user_id: str) -> str:
        """Get personality for a user"""
        return self.user_preferences.get(user_id, self.default_personality)
    
    def get_system_prompt(self, user_id: str) -> str:
        """Generate system prompt based on personality"""
        
        personality = self.get_personality(user_id)
        config = self.PERSONALITIES[personality]
        
        prompt = f"You are a helpful AI assistant with a {config['tone']} tone. "
        prompt += f"Your responses should be {config['style']}. "
        
        if config["emojis"]:
            prompt += "You can use emojis to make responses more engaging. "
        else:
            prompt += "Avoid using emojis. "
        
        if config["examples"]:
            prompt += "Provide examples when helpful."
        
        return prompt
    
    def adapt_response(
        self,
        response: str,
        user_id: str,
        context: Optional[Dict] = None,
    ) -> str:
        """
        Adapt response based on personality and context.
        
        This is a post-processing step for fine-tuning responses.
        """
        
        personality = self.get_personality(user_id)
        
        # Voice mode detection
        if context and context.get("is_voice"):
            # Optimize for voice
            from app.utils.formatters import ResponseFormatter
            response = ResponseFormatter.format_for_voice(response)
        
        return response


class ChatTool:
    """
    Main chat tool combining conversation management and personality.
    """
    
    def __init__(self):
        self.conversation_manager = ConversationManager()
        self.personality_adapter = PersonalityAdapter()
        self.chat_stats = {
            "total_messages": 0,
            "by_personality": {p: 0 for p in PersonalityAdapter.PERSONALITIES},
        }
    
    async def process_message(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Process an incoming chat message.
        
        Returns:
            Processed message data ready for LLM
        """
        
        # Add to conversation history
        self.conversation_manager.add_message(
            conversation_id,
            "user",
            message,
            metadata=context,
        )
        
        # Get conversation context
        history = self.conversation_manager.get_context(conversation_id)
        
        # Get system prompt based on personality
        system_prompt = self.personality_adapter.get_system_prompt(user_id)
        
        # Combine into messages for LLM
        messages = [{"role": "system", "content": system_prompt}] + history
        
        # Update stats
        self.chat_stats["total_messages"] += 1
        personality = self.personality_adapter.get_personality(user_id)
        self.chat_stats["by_personality"][personality] += 1
        
        return {
            "messages": messages,
            "personality": personality,
            "conversation_id": conversation_id,
            "context": context or {},
        }
    
    async def process_response(
        self,
        user_id: str,
        conversation_id: str,
        response: str,
        context: Optional[Dict] = None,
    ) -> str:
        """
        Process LLM response before returning to user.
        """
        
        # Add to conversation history
        self.conversation_manager.add_message(
            conversation_id,
            "assistant",
            response,
        )
        
        # Adapt response based on personality
        adapted_response = self.personality_adapter.adapt_response(
            response,
            user_id,
            context,
        )
        
        return adapted_response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chat statistics"""
        
        return {
            **self.chat_stats,
            "conversations": self.conversation_manager.get_stats(),
        }


# Global chat tool instance
chat_tool = ChatTool()