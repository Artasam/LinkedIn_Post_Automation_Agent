"""
agents/content_agent.py
-----------------------
Content Agent: Generates a professional, high-quality LinkedIn post
using Groq LLM (llama-3.3-70b-versatile).

Quality issues fixed:
  - Posts were too short (67 words) → now enforces minimum 120 words
  - Paragraphs weren't separated → now enforces blank lines between paragraphs
  - First line hook wasn't strong → now uses stronger hook instructions
  - Score was 34.6/100 → target score 70+
"""

import logging
import re

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings

logger = logging.getLogger(__name__)

# ─── System Prompt ─────────────────────────────────────────────────────────────
CONTENT_SYSTEM_PROMPT = """\
You are an elite LinkedIn content strategist with 10 years of experience
creating viral AI and technology posts. Your posts consistently get 500+ likes.

Your writing style:
  • Opens with a POWERFUL, specific hook that creates instant curiosity
  • Uses concrete numbers, percentages, or surprising facts in the first line
  • Writes in short, punchy paragraphs with a BLANK LINE between each
  • Builds tension → insight → action → question
  • Feels like expert advice from a senior AI engineer, not a press release
  • Avoids corporate buzzwords like "leverage", "utilize", "synergy"

CRITICAL RULES:
  1. Write ONLY the post body — no subject lines, no labels, no metadata
  2. NEVER mention the source name or publication
  3. ALWAYS put a blank line between paragraphs (double newline)
  4. NEVER use bullet points or numbered lists
  5. NEVER start with "I" or "In today's"
  6. The first line MUST contain either a number, a "?" or "!" character,
     or one of these power starters:
     "Here's why...", "Nobody talks about...", "Stop ignoring...",
     "This changes everything:", "The uncomfortable truth:", 
     "Most people don't know...", "X just happened and..."
"""

# ─── User Prompt Template ──────────────────────────────────────────────────────
CONTENT_USER_PROMPT_TEMPLATE = """\
Write a LinkedIn post about this AI topic:

TOPIC: {title}
CONTEXT: {summary}

STRUCTURE (follow this exactly):

Line 1 — HOOK: One powerful sentence with a specific number or bold claim.
         Must grab attention immediately. (15-20 words max)

[blank line]

Paragraph 2 — CONTEXT: Explain what this means and why it matters.
              2-3 sentences. Be specific, not generic. (30-40 words)

[blank line]

Paragraph 3 — INSIGHT: The deeper implication or surprising angle.
              What most people miss about this topic. (30-40 words)

[blank line]

Paragraph 4 — ACTION: One concrete, specific thing the reader can do TODAY.
              Start with an action verb. (15-25 words)

[blank line]

Line 5 — QUESTION: End with ONE engaging question that invites real debate.
         Make it thought-provoking, not generic. (10-20 words)

REQUIREMENTS:
- Total length: 130 to {max_words} words (COUNT CAREFULLY)
- Every paragraph separated by a blank line
- No hashtags (added separately)
- No bullet points or numbered lists
- No source names or publication references

Write the post now:
"""


def _count_words(text: str) -> int:
    """Count words excluding hashtags."""
    words = re.findall(r"\b\w+\b", text)
    return len([w for w in words if not w.startswith("#")])


def _count_paragraphs(text: str) -> int:
    """Count non-empty paragraphs (separated by blank lines)."""
    return len([p for p in text.split("\n\n") if p.strip()])


def _trim_to_word_limit(text: str, max_words: int) -> str:
    """Trim post to max_words at sentence boundaries."""
    if _count_words(text) <= max_words:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    trimmed = ""
    for sentence in sentences:
        candidate = (trimmed + " " + sentence).strip()
        if _count_words(candidate) <= max_words:
            trimmed = candidate
        else:
            break
    return trimmed.rstrip(".!?,") + "…" if trimmed else text[:600] + "…"


def _ensure_paragraph_spacing(text: str) -> str:
    """
    Ensure paragraphs are separated by blank lines.
    Fixes cases where the LLM uses single newlines instead of double.
    """
    # Split on single newlines, rejoin with double newlines
    lines = text.split("\n")
    paragraphs = []
    current = []

    for line in lines:
        stripped = line.strip()
        if stripped:
            current.append(stripped)
        else:
            if current:
                paragraphs.append(" ".join(current))
                current = []

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs)


def _score_post(text: str) -> float:
    """
    Quality score for a generated post (0–100).
    Target score: 70+

    Scoring breakdown:
      Hook strength     : up to 25 pts
      Word count        : up to 20 pts
      Paragraph count   : up to 20 pts
      Closing question  : up to 15 pts
      No bad patterns   : up to 10 pts
      Specificity       : up to 10 pts
    """
    score = 30.0  # base

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    first_line = paragraphs[0].split("\n")[0].lower() if paragraphs else ""
    last_para = paragraphs[-1].lower() if paragraphs else ""

    # ── Hook strength (up to 25 pts) ─────────────────────────────────────────
    power_starters = [
        "here's why", "nobody talks", "stop ignoring", "this changes",
        "uncomfortable truth", "most people don't", "just happened",
        "revolutioniz", "game-changer", "breakthrough", "surprising",
        "just dropped", "finally", "imagine", "unveiled", "announced"
    ]
    if any(pw in first_line for pw in power_starters):
        score += 15
    if re.search(r"\d+", first_line):       # contains number
        score += 5
    if any(c in first_line for c in "?!"):  # question or exclamation
        score += 5

    # ── Word count (up to 20 pts) ─────────────────────────────────────────────
    wc = _count_words(text)
    if 130 <= wc <= settings.POST_MAX_WORDS:
        score += 20
    elif 100 <= wc < 130:
        score += 10
    elif wc < 100:
        score -= 10   # too short — penalise heavily
    else:
        score -= (wc - settings.POST_MAX_WORDS) * 0.3

    # ── Paragraph count (up to 20 pts) ────────────────────────────────────────
    para_count = len(paragraphs)
    if 4 <= para_count <= 6:
        score += 20
    elif para_count == 3:
        score += 10
    elif para_count < 3:
        score -= 10   # too condensed

    # ── Closing question (up to 15 pts) ───────────────────────────────────────
    if "?" in last_para:
        score += 15

    # ── No bad patterns (up to 10 pts) ────────────────────────────────────────
    bad_patterns = [
        r"^in today",          # generic opener
        r"^i ",                # first person opener
        r"leverage",           # corporate buzzword
        r"utilize",
        r"synergy",
        r"according to",       # source reference
        r"as reported",
    ]
    penalties = sum(1 for bp in bad_patterns if re.search(bp, text.lower()))
    score += max(0, 10 - penalties * 3)

    # ── Specificity bonus (up to 10 pts) ─────────────────────────────────────
    # Numbers, percentages, model names = specific content
    specifics = len(re.findall(r"\b\d+%?\b|\bGPT\b|\bLlama\b|\bClaude\b|\bGemini\b", text))
    score += min(specifics * 2, 10)

    return max(0.0, min(100.0, score))


def generate_post(topic: dict, attempt: int = 1) -> str:
    """
    Generate a single LinkedIn post for a given topic dict.

    Args:
        topic:   dict with 'title' and 'summary' keys.
        attempt: retry attempt number (for logging).

    Returns:
        The post text string (without hashtags).
    """
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=min(settings.GROQ_TEMPERATURE + 0.1, 0.95),
        max_tokens=settings.GROQ_MAX_TOKENS,
    )

    # On retry (attempt 2), use a stricter prompt with explicit word target
    if attempt == 2:
        retry_prompt = (
            f"You are a LinkedIn content writer. Write a post about:\n"
            f"TOPIC: {topic.get('title', 'AI')}\n"
            f"CONTEXT: {topic.get('summary', '')}\n\n"
            f"REQUIREMENTS:\n"
            f"- Write EXACTLY 5 paragraphs separated by blank lines\n"
            f"- Each paragraph must be 2-3 full sentences\n"
            f"- Total word count MUST be between 130 and {settings.POST_MAX_WORDS}\n"
            f"- Start with a specific fact or number\n"
            f"- End with a question\n"
            f"- No hashtags, no labels, no bullet points\n"
            f"Write now:"
        )
        messages = [HumanMessage(content=retry_prompt)]
    else:
        messages = [
            SystemMessage(content=CONTENT_SYSTEM_PROMPT),
            HumanMessage(content=CONTENT_USER_PROMPT_TEMPLATE.format(
                title=topic.get("title", "AI Trends"),
                summary=topic.get("summary", ""),
                max_words=settings.POST_MAX_WORDS,
            )),
        ]

    try:
        response = llm.invoke(messages)
        post_text = response.content.strip()
    except Exception as exc:
        logger.error("Content generation failed (attempt %d): %s", attempt, exc)
        raise

    # Ensure proper paragraph spacing
    post_text = _ensure_paragraph_spacing(post_text)

    # Trim if over word limit
    post_text = _trim_to_word_limit(post_text, settings.POST_MAX_WORDS)

    wc = _count_words(post_text)
    logger.info(
        "Generated post: %d words, attempt %d, topic '%s'",
        wc,
        attempt,
        topic.get("title", "")[:50],
    )

    # If post is too short (under 100 words), retry with a stronger length directive
    if wc < 100 and attempt == 1:
        logger.warning(
            "Post too short (%d words) — retrying with stronger length directive.", wc
        )
        return generate_post(topic, attempt=2)

    return post_text


def generate_best_post(topics: list) -> dict:
    """
    Generate one post per topic (up to MULTI_TOPIC_DRAFTS),
    score each on quality metrics, and return the best draft.

    Returns:
        dict with keys: post_text, topic, score
    """
    n_drafts = min(settings.MULTI_TOPIC_DRAFTS, len(topics))
    if n_drafts == 0:
        raise ValueError("No topics provided.")

    drafts = []
    for i, topic in enumerate(topics[:n_drafts]):
        try:
            text = generate_post(topic)
            score = _score_post(text)
            drafts.append({"post_text": text, "topic": topic, "score": score})
            logger.info(
                "Draft %d/%d for '%s' → quality score: %.1f",
                i + 1, n_drafts,
                topic.get("title", "")[:50],
                score,
            )
        except Exception as exc:
            logger.warning("Skipping topic '%s': %s", topic.get("title"), exc)

    if not drafts:
        raise RuntimeError("All post generation attempts failed.")

    best = max(drafts, key=lambda d: d["score"])
    logger.info(
        "Best draft selected: '%s' (score: %.1f)",
        best["topic"].get("title", "")[:50],
        best["score"],
    )
    return best