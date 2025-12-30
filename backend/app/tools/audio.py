# --- Contains the STT (Speech-to-Text) and TTS (Text-to-Speech) logic.

"""
Audio Tools - Speech-to-Text and Text-to-Speech
================================================
Handles voice input/output using Web Speech API and OpenAI TTS.
"""

import asyncio
import base64
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile

from loguru import logger

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available. TTS disabled.")


class TextToSpeech:
    """
    Text-to-Speech using OpenAI API.
    
    Supports:
    - Multiple voices (alloy, echo, fable, onyx, nova, shimmer)
    - Speed control
    - Format selection (mp3, opus, aac, flac)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.api_url = "https://api.openai.com/v1/audio/speech"
        
        self.voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        self.models = ["tts-1", "tts-1-hd"]
        
        self.usage_stats = {
            "total_requests": 0,
            "total_characters": 0,
            "by_voice": {voice: 0 for voice in self.voices},
        }
    
    async def synthesize(
        self,
        text: str,
        voice: str = "nova",
        model: str = "tts-1",
        speed: float = 1.0,
        output_format: str = "mp3",
    ) -> Optional[bytes]:
        """
        Convert text to speech.
        
        Args:
            text: Text to synthesize
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            model: TTS model (tts-1 or tts-1-hd)
            speed: Speech speed (0.25 to 4.0)
            output_format: Audio format (mp3, opus, aac, flac)
        
        Returns:
            Audio data as bytes or None if error
        """
        
        if not HTTPX_AVAILABLE:
            logger.error("httpx not available for TTS")
            return None
        
        if not self.api_key:
            logger.error("OpenAI API key not configured")
            return None
        
        # Validate parameters
        if voice not in self.voices:
            logger.warning(f"Invalid voice '{voice}', using 'nova'")
            voice = "nova"
        
        if model not in self.models:
            logger.warning(f"Invalid model '{model}', using 'tts-1'")
            model = "tts-1"
        
        speed = max(0.25, min(4.0, speed))
        
        logger.info(f"🔊 Synthesizing speech: {len(text)} chars, voice={voice}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "input": text,
                        "voice": voice,
                        "speed": speed,
                        "response_format": output_format,
                    },
                    timeout=30.0,
                )
                
                response.raise_for_status()
                
                # Update stats
                self.usage_stats["total_requests"] += 1
                self.usage_stats["total_characters"] += len(text)
                self.usage_stats["by_voice"][voice] += 1
                
                logger.success(f"✓ Generated {len(response.content)} bytes of audio")
                return response.content
        
        except httpx.HTTPStatusError as e:
            logger.error(f"TTS API error {e.response.status_code}: {e.response.text}")
            return None
        
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None
    
    async def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        **kwargs
    ) -> bool:
        """
        Synthesize speech and save to file.
        
        Returns:
            True if successful, False otherwise
        """
        
        audio_data = await self.synthesize(text, **kwargs)
        
        if audio_data:
            try:
                Path(output_path).write_bytes(audio_data)
                logger.success(f"✓ Saved audio to {output_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to save audio: {e}")
                return False
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get TTS usage statistics"""
        return self.usage_stats


class SpeechToText:
    """
    Speech-to-Text using Web Speech API (client-side).
    This class provides utilities and validation for STT.
    
    Note: Actual STT happens in browser using Web Speech API,
    this class handles the server-side processing of results.
    """
    
    def __init__(self):
        self.supported_languages = [
            "en-US", "en-GB", "es-ES", "fr-FR", "de-DE",
            "it-IT", "pt-BR", "ru-RU", "ja-JP", "ko-KR",
            "zh-CN", "hi-IN", "ar-SA",
        ]
        
        self.transcription_stats = {
            "total_transcriptions": 0,
            "by_language": {},
        }
    
    def validate_language(self, language_code: str) -> bool:
        """Validate language code"""
        return language_code in self.supported_languages
    
    async def process_transcription(
        self,
        text: str,
        language: str = "en-US",
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Process and validate transcription from client.
        
        Args:
            text: Transcribed text
            language: Language code
            confidence: Confidence score (0.0 to 1.0)
        
        Returns:
            Processed transcription data
        """
        
        # Update stats
        self.transcription_stats["total_transcriptions"] += 1
        if language not in self.transcription_stats["by_language"]:
            self.transcription_stats["by_language"][language] = 0
        self.transcription_stats["by_language"][language] += 1
        
        # Clean and normalize text
        text = text.strip()
        
        # Detect if it's a voice query
        from app.utils.formatters import VoiceOptimizer
        is_voice = VoiceOptimizer.detect_voice_query(text)
        
        return {
            "text": text,
            "language": language,
            "confidence": confidence,
            "is_voice_query": is_voice,
            "word_count": len(text.split()),
            "char_count": len(text),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get STT statistics"""
        return self.transcription_stats


class AudioProcessor:
    """
    Audio processing utilities.
    
    Handles:
    - Format conversion
    - Audio validation
    - Metadata extraction
    """
    
    @staticmethod
    def validate_audio_data(data: bytes, max_size_mb: float = 25.0) -> bool:
        """
        Validate audio data.
        
        Args:
            data: Audio bytes
            max_size_mb: Maximum size in MB
        
        Returns:
            True if valid, False otherwise
        """
        
        size_mb = len(data) / (1024 * 1024)
        
        if size_mb > max_size_mb:
            logger.warning(f"Audio too large: {size_mb:.2f}MB (max: {max_size_mb}MB)")
            return False
        
        # Check for audio magic numbers
        audio_signatures = [
            b'\xFF\xFB',  # MP3
            b'\xFF\xF3',  # MP3
            b'\xFF\xF2',  # MP3
            b'ID3',       # MP3 with ID3
            b'OggS',      # OGG
            b'RIFF',      # WAV
            b'fLaC',      # FLAC
        ]
        
        is_valid = any(data.startswith(sig) for sig in audio_signatures)
        
        if not is_valid:
            logger.warning("Invalid audio format")
        
        return is_valid
    
    @staticmethod
    def encode_audio_base64(data: bytes) -> str:
        """Encode audio data to base64"""
        return base64.b64encode(data).decode('utf-8')
    
    @staticmethod
    def decode_audio_base64(encoded: str) -> bytes:
        """Decode base64 audio data"""
        return base64.b64decode(encoded)


# Global instances
tts_engine = TextToSpeech()
stt_engine = SpeechToText()
audio_processor = AudioProcessor()