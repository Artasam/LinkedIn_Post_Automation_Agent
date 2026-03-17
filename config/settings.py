"""
config/settings.py
------------------
Central configuration for the LinkedIn AI Content Automation Agent.
Single source of truth for all environment variables and constants.

Model selection:
  llama-3.3-70b-versatile — default
  • Best prose quality for professional LinkedIn content
  • 6,000 TPM limit — pipeline uses ~4,500 TPM (5 LLM calls × ~900 TPM)
  • Stable production model (not preview)

  Other available Groq models:
  "llama-3.1-8b-instant"                      TPM: 6,000  — fastest, lower quality
  "meta-llama/llama-4-scout-17b-16e-instruct" TPM: 30,000 — multimodal preview
  "qwen/qwen3-32b"                             TPM: 6,000  — math/coding focus
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Groq / LLM ───────────────────────────────────────────────────────────────
GROQ_API_KEY:     str   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL:       str   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.7"))
# 700 tokens gives content agent room to write 130-300 words comfortably.
# 5 LLM calls × ~900 TPM = ~4,500 TPM — within 6,000 TPM limit.
GROQ_MAX_TOKENS:  int   = int(os.getenv("GROQ_MAX_TOKENS", "700"))

# ─── LinkedIn ─────────────────────────────────────────────────────────────────
LINKEDIN_ACCESS_TOKEN: str = os.getenv("LINKEDIN_ACCESS_TOKEN", "")
LINKEDIN_PERSON_ID:    str = os.getenv("LINKEDIN_PERSON_ID", "")
LINKEDIN_API_BASE:     str = "https://api.linkedin.com/v2"

# ─── News Sources ─────────────────────────────────────────────────────────────
# news_tool.py fetches from 6 independent REST APIs (no RSS).
# Sources: ArXiv, HackerNews, GitHub Trending, NewsAPI, Wikipedia, DuckDuckGo.
RSS_MAX_ARTICLES_PER_FEED: int = int(os.getenv("RSS_MAX_ARTICLES_PER_FEED", "8"))

# Optional: NewsAPI key — free tier 100 req/day (newsapi.org)
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

# Optional: GitHub token raises rate limit from 60 to 5,000 req/hr
# Get at https://github.com/settings/tokens (no scopes needed)
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")

# ─── Agent Settings ───────────────────────────────────────────────────────────
TOPIC_CANDIDATE_COUNT: int = int(os.getenv("TOPIC_CANDIDATE_COUNT", "5"))
MULTI_TOPIC_DRAFTS:    int = int(os.getenv("MULTI_TOPIC_DRAFTS", "1"))

# ─── Post Constraints ─────────────────────────────────────────────────────────
POST_MAX_WORDS:    int = 300  # hard upper limit enforced by content_agent
POST_MIN_HASHTAGS: int = 3    # LinkedIn recommends 3-5 hashtags
POST_MAX_HASHTAGS: int = 5    # more than 5 looks spammy

# ─── Content Safety ──────────────────────────────────────────────────────────
# Used by safety_tool.py — runs as a node in the pipeline after content generation
SAFETY_MAX_RETRIES:    int  = int(os.getenv("SAFETY_MAX_RETRIES", "3"))
ENABLE_LLM_MODERATION: bool = os.getenv("ENABLE_LLM_MODERATION", "true").lower() == "true"

# ─── Image Generation ─────────────────────────────────────────────────────────
# Engine waterfall (all live-tested 2026-03-17):
#   1. Pexels API    — FREE, 200 req/hr, needs PEXELS_API_KEY
#   2. Unsplash API  — FREE, 50 req/hr,  needs UNSPLASH_ACCESS_KEY
#   3. SVG Generator — FREE, always works, no key needed (guaranteed fallback)
ENABLE_IMAGE_GENERATION: bool = os.getenv("ENABLE_IMAGE_GENERATION", "false").lower() == "true"
PEXELS_API_KEY:     str = os.getenv("PEXELS_API_KEY", "")
UNSPLASH_ACCESS_KEY: str = os.getenv("UNSPLASH_ACCESS_KEY", "")

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL:    str  = os.getenv("LOG_LEVEL", "INFO")
LOG_TO_FILE:  bool = os.getenv("LOG_TO_FILE", "false").lower() == "true"
LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "logs/agent.log")


def validate_config() -> list:
    """
    Check all required environment variables are set.
    Returns list of missing variable names (empty list = all good).
    """
    required = {
        "GROQ_API_KEY":           GROQ_API_KEY,
        "LINKEDIN_ACCESS_TOKEN":  LINKEDIN_ACCESS_TOKEN,
        "LINKEDIN_PERSON_ID":     LINKEDIN_PERSON_ID,
    }
    return [name for name, value in required.items() if not value]