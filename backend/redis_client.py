import redis
import json
from typing import Optional, Dict, List
from datetime import datetime
from config import settings
import logging

logger = logging.getLogger(__name__)

redis_client = None

PER_SESSION_HISTORY_LIMIT = 10
GLOBAL_HISTORY_PAIR_LIMIT = 1000


def init_redis():
    """Initialize Redis connection"""
    global redis_client
    redis_client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        decode_responses=True
    )
    try:
        redis_client.ping()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        raise


def get_session_history(session_id: str) -> Optional[Dict]:
    """Get conversation history for a session"""
    if not redis_client:
        return None
    
    key = f"session:{session_id}"
    data = redis_client.get(key)
    if data:
        return json.loads(data)
    return None


def update_session_history(session_id: str, user_query: str, system_response: str):
    """Update conversation history for a session"""
    if not redis_client:
        return
    
    key = f"session:{session_id}"
    history = get_session_history(session_id) or {
        "session_id": session_id,
        "conversation_history": [],
        "recent_qna_pairs": []
    }
    
    # add to conversation history
    history["conversation_history"].append({
        "user_query": user_query,
        "system_response": system_response,
        "timestamp": datetime.now().isoformat()
    })
    
    # trim session history
    if len(history["conversation_history"]) > PER_SESSION_HISTORY_LIMIT:
        history["conversation_history"] = history["conversation_history"][-PER_SESSION_HISTORY_LIMIT:]
    
    # add to recent QnA pairs
    history["recent_qna_pairs"].append({
        "question": user_query,
        "answer": system_response,
        "retrieved_at": datetime.now().isoformat()
    })
    
    # trim global QnA history
    if len(history["recent_qna_pairs"]) > GLOBAL_HISTORY_PAIR_LIMIT:
        history["recent_qna_pairs"] = history["recent_qna_pairs"][-GLOBAL_HISTORY_PAIR_LIMIT:]
    
    redis_client.setex(key, settings.SESSION_TTL, json.dumps(history))


def get_recent_qna_pairs(session_id: str) -> List[Dict]:
    """Get recent QnA pairs for a session"""
    history = get_session_history(session_id)
    if history:
        return history.get("recent_qna_pairs", [])
    return []


def clear_session(session_id: str):
    """Clear a session"""
    if redis_client:
        key = f"session:{session_id}"
        redis_client.delete(key)

