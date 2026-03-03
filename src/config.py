import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# SurrealDB configuration
SURREAL_URL = os.getenv("SURREAL_URL", "ws://localhost:8000")
SURREAL_USER = os.getenv("SURREAL_USER", "jarvis")
SURREAL_PASS = os.getenv("SURREAL_PASS", "memory")
SURREAL_NS = os.getenv("SURREAL_NS", "main")
SURREAL_DB = os.getenv("SURREAL_DB", "main")
