import os

TEMPORAL_HOST = os.environ.get("TEMPORAL_HOST", "localhost:7233")
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:11435")
MODEL = os.environ.get("MODEL", "default")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://temporal:temporal@localhost:5432/churner")
