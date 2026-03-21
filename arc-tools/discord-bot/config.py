import os

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:11435")
MODEL = os.environ.get("MODEL", "default")
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "40"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))
COOLDOWN_SECONDS = float(os.environ.get("COOLDOWN_SECONDS", "2.0"))
