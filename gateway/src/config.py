from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    # Server
    port: int = 8080
    log_level: str = "info"
    
    # CORS
    cors_origins: str = "http://localhost:3000,http://localhost:16382"
    
    # Redis
    redis_url: str = "redis://redis:6379/0"
    session_ttl: int = 3600  # 1 hour
    
    # Upstream Services
    asr_url: str = "http://asr-service:8001"
    tts_url: str = "http://tts-service:8002"
    omni_url: str = "http://omni-service:8003"
    
    # Service Timeouts
    asr_timeout: int = 30
    tts_timeout: int = 60
    omni_timeout: int = 120
    
    # Auth (optional)
    jwt_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    @property
    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    class Config:
        env_file = ".env"


settings = Settings()
