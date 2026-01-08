from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "rag_db"
    
    # Redis
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # OpenAI
    OPENAI_API_KEY: str
    
    # Model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    WHISPER_MODEL: str = "small"
    TTS_MODEL: str = "tts_models/en/ljspeech/glow-tts"
    
    # Processing settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    SIMILARITY_THRESHOLD: float = 0.7
    QNA_CACHE_THRESHOLD: float = 0.85
    SESSION_TTL: int = 3600  # 1 hour
    SKIP_RECTIFICATION_FOR_SIMPLE_QUERIES: bool = True
    
    # API settings
    API_V1_PREFIX: str = "/api"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

