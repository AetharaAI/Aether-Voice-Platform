import json
import redis
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages user sessions using Redis"""
    
    def __init__(self, redis_url: str, ttl: int = 3600):
        try:
            self.redis = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis.ping()
            self.available = True
            logger.info("✅ SessionManager connected to Redis")
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable: {e}")
            self.redis = None
            self.available = False
        self.ttl = ttl
        
    def create_session(self, user_id: str, metadata: Optional[Dict] = None) -> str:
        """Create new session"""
        if not self.available:
            # Fallback: generate session ID without persistence
            return str(uuid.uuid4())
        
        session_id = str(uuid.uuid4())
        session_data = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_active": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
            "conversation_state": None
        }
        
        key = f"session:{session_id}"
        self.redis.setex(key, self.ttl, json.dumps(session_data))
        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        if not self.available:
            return None
        
        key = f"session:{session_id}"
        data = self.redis.get(key)
        if data:
            session = json.loads(data)
            # Update last active
            session["last_active"] = datetime.utcnow().isoformat()
            self.redis.setex(key, self.ttl, json.dumps(session))
            return session
        return None
    
    def update_session(self, session_id: str, updates: Dict[str, Any]):
        """Update session fields"""
        if not self.available:
            return False
        
        key = f"session:{session_id}"
        data = self.redis.get(key)
        if data:
            session = json.loads(data)
            session.update(updates)
            session["last_active"] = datetime.utcnow().isoformat()
            self.redis.setex(key, self.ttl, json.dumps(session))
            return True
        return False
    
    def delete_session(self, session_id: str):
        """Delete session"""
        if not self.available:
            return
        
        key = f"session:{session_id}"
        self.redis.delete(key)
        logger.info(f"Deleted session {session_id}")
    
    def session_exists(self, session_id: str) -> bool:
        """Check if session exists"""
        if not self.available:
            return False
        return self.redis.exists(f"session:{session_id}") > 0
