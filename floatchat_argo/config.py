from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    pg_host: str = os.getenv("POSTGRES_HOST", "localhost")
    pg_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    pg_db: str = os.getenv("POSTGRES_DB", "argo")
    pg_user: str = os.getenv("POSTGRES_USER", "argo")
    pg_password: str = os.getenv("POSTGRES_PASSWORD", "argo")
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8000"))
    chroma_dir: str = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
    env: str = os.getenv("ENV", "dev")

    @property
    def sqlalchemy_url(self) -> str:
        return f"postgresql+psycopg2://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}"

settings = Settings()
