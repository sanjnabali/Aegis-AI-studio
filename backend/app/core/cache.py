"""
Redis Caching Layer - Updated for Python 3.11
"""
import hashlib
import json
from typing import Optional
from datetime import datetime

from loguru import logger

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis not installed")

redis_client: Optional[aioredis.Redis] = None
cache_stats = {"hits": 0, "misses": 0, "errors": 0}

async def init_cache():
    """Initialize Redis connection"""
    global redis_client
    
    if not REDIS_AVAILABLE:
        logger.warning("Redis not available")
        return
    
    try:
        redis_client = aioredis.Redis(
            host="redis",
            port=6379,
            decode_responses=True
        )
        await redis_client.ping()
        logger.success("✓ Redis cache connected")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        redis_client = None

async def close_cache():
    """Close Redis connection"""
    if redis_client:
        await redis_client.close()

def generate_cache_key(messages: list, model: str) -> str:
    """Generate cache key"""
    content = json.dumps({"messages": messages, "model": model}, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()

async def get_cached_response(cache_key: str) -> Optional[str]:
    """Get cached response"""
    if not redis_client:
        return None
    
    try:
        cached = await redis_client.get(f"aegis:chat:{cache_key}")
        if cached:
            cache_stats["hits"] += 1
            return cached
        else:
            cache_stats["misses"] += 1
            return None
    except Exception as e:
        cache_stats["errors"] += 1
        logger.error(f"Cache error: {e}")
        return None

async def set_cached_response(cache_key: str, response: str, ttl: int = 3600):
    """Set cached response"""
    if redis_client:
        try:
            await redis_client.setex(f"aegis:chat:{cache_key}", ttl, response)
        except Exception as e:
            logger.error(f"Cache write error: {e}")

def get_cache_stats():
    """Get cache statistics"""
    total = cache_stats["hits"] + cache_stats["misses"]
    hit_rate = (cache_stats["hits"] / total * 100) if total > 0 else 0
    
    return {
        "enabled": redis_client is not None,
        "hits": cache_stats["hits"],
        "misses": cache_stats["misses"],
        "errors": cache_stats["errors"],
        "hit_rate": f"{hit_rate:.1f}%",
        "total_requests": total,
    }