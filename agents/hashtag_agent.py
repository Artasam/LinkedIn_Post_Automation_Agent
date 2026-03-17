"""
agents/hashtag_agent.py
-----------------------
Hashtag Agent: Generates 3-5 optimised LinkedIn hashtags for an AI post.

Updated per LinkedIn best practices:
  - 3-5 hashtags (not 4-6) — LinkedIn's own recommendation for reach
  - Mix of broad reach + niche topic-specific tags
  - No banned/spammy hashtags
  - CamelCase formatting for readability
"""

import logging
import re

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings

logger = logging.getLogger(__name__)

# Evergreen base hashtags — always high-reach on LinkedIn AI content
BASE_AI_HASHTAGS = [
    "#ArtificialIntelligence",
    "#MachineLearning",
    "#GenerativeAI",
    "#DeepLearning",
    "#LLM",
    "#MLOps",
    "#DataScience",
    "#AIResearch",
    "#OpenSourceAI",
    "#NLP",
    "#AIAgents",
    "#RAG",
    "#ComputerScience",
    "#TechInnovation",
    "#AI",
]

# Hashtags to never use — too broad, spammy, or off-topic
BLOCKED_HASHTAGS = {
    "#viral", "#followme", "#follow", "#like", "#share",
    "#makemoneyonline", "#crypto", "#nft", "#marketing",
    "#business", "#success", "#motivation", "#hustle",
    "#entrepreneur", "#mindset",
}


def _clean_hashtag(tag: str) -> str:
    """Normalise a hashtag — ensure leading #, strip spaces."""
    tag = tag.strip().lstrip("#").replace(" ", "")
    if not tag or not tag.isalnum() and not re.match(r"^\w+$", tag):
        return ""
    return f"#{tag}"


def _is_valid_hashtag(tag: str) -> bool:
    """Return True if hashtag is safe and appropriate for LinkedIn AI content."""
    if not tag or len(tag) < 2:
        return False
    if tag.lower() in {b.lower() for b in BLOCKED_HASHTAGS}:
        return False
    # Must be alphanumeric (no special chars after #)
    if not re.match(r"^#[A-Za-z][A-Za-z0-9]*$", tag):
        return False
    # Not too long (LinkedIn hashtag best practice)
    if len(tag) > 30:
        return False
    return True


def _deduplicate(tags: list) -> list:
    """Remove duplicate hashtags (case-insensitive)."""
    seen = set()
    result = []
    for tag in tags:
        key = tag.lower()
        if key not in seen:
            seen.add(key)
            result.append(tag)
    return result


def generate_hashtags_with_llm(post_text: str, topic: dict) -> list:
    """
    Ask the LLM to generate contextual AI hashtags for the post.

    Instructs the LLM to:
    - Use 3-5 hashtags (LinkedIn recommended range)
    - Mix 1-2 broad reach tags with 2-3 niche topic tags
    - Use CamelCase formatting
    - Avoid spammy or off-topic tags
    """
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0.3,   # low temperature for consistent, safe hashtag output
        max_tokens=150,
    )

    messages = [
        SystemMessage(content=(
            "You are a LinkedIn hashtag expert specialising in AI and technology content. "
            "Generate professional hashtags that are relevant to the AI/tech community. "
            "Only use hashtags that are professionally appropriate for LinkedIn. "
            "Never use promotional, spammy, or off-topic hashtags."
        )),
        HumanMessage(content=(
            f"Generate {settings.POST_MIN_HASHTAGS} to {settings.POST_MAX_HASHTAGS} "
            f"LinkedIn hashtags for this AI post.\n\n"
            f"TOPIC: {topic.get('title', '')}\n"
            f"POST EXCERPT: {post_text[:150]}\n\n"
            f"RULES:\n"
            f"- Use CamelCase for multi-word tags (e.g. #GenerativeAI, #MachineLearning)\n"
            f"- Include 1-2 broad reach tags (#AI, #MachineLearning, #DataScience)\n"
            f"- Include 1-3 topic-specific niche tags\n"
            f"- Only professional, AI/tech relevant hashtags\n"
            f"- Do NOT use: #viral, #follow, #business, #motivation, or promotional tags\n"
            f"- Return ONLY hashtags separated by spaces — nothing else"
        )),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
    except Exception as exc:
        logger.error("Hashtag LLM call failed: %s", exc)
        return []

    # Extract and validate hashtags
    tokens = re.findall(r"#?\w+", raw)
    tags = []
    for token in tokens:
        tag = _clean_hashtag(token)
        if tag and _is_valid_hashtag(tag):
            tags.append(tag)

    logger.info("LLM generated %d valid hashtags: %s", len(tags), " ".join(tags))
    return tags


def select_final_hashtags(llm_tags: list, topic: dict) -> list:
    """
    Combine LLM-generated tags with evergreen base tags.
    Validates all tags, deduplicates, and enforces count limits.

    Strategy:
    1. LLM-generated contextual tags take priority
    2. Pad with evergreen base tags if below minimum
    3. Cap at POST_MAX_HASHTAGS
    """
    # Validate LLM tags
    valid_llm = [t for t in llm_tags if _is_valid_hashtag(t)]

    # Combine with base tags (LLM tags first)
    combined = _deduplicate(valid_llm + BASE_AI_HASHTAGS)

    # Filter out blocked hashtags
    combined = [t for t in combined if _is_valid_hashtag(t)]

    # Enforce count limits
    final = combined[:settings.POST_MAX_HASHTAGS]

    # Ensure minimum
    if len(final) < settings.POST_MIN_HASHTAGS:
        padding = [t for t in BASE_AI_HASHTAGS if t not in final]
        final += padding[:settings.POST_MIN_HASHTAGS - len(final)]

    final = final[:settings.POST_MAX_HASHTAGS]
    logger.info("Final hashtags (%d): %s", len(final), " ".join(final))
    return final


def run_hashtag_agent(post_text: str, topic: dict) -> str:
    """
    Entry point for the Hashtag Agent.
    Returns a space-separated string of 3-5 validated hashtags.
    """
    logger.info("Hashtag Agent: Generating hashtags…")
    llm_tags   = generate_hashtags_with_llm(post_text, topic)
    final_tags = select_final_hashtags(llm_tags, topic)
    return " ".join(final_tags)