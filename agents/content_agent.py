"""
agents/content_agent.py
-----------------------
Content Agent: Generates LinkedIn posts optimised for the LinkedIn
algorithm and maximum human engagement.

Key design decisions based on LinkedIn algorithm research (2025):
  - DWELL TIME wins: short paragraphs, one idea per line, white space
  - COMMENTS > LIKES: divisive, specific questions beat generic ones
  - PERSONAL POV beats news summary: expert insight framing
  - SPECIFIC beats VAGUE: numbers, model names, percentages
  - FORMAT matters: every line break increases readability score
  - 5 rotating formats prevent audience fatigue
  - No promotional CTAs (flagged as spam by safety tools)
  - No "Revolutionizing", "Game-changing", "X% of..." openers
"""

import logging
import random
import re
from datetime import date

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings

logger = logging.getLogger(__name__)


# ─── Post Format Rotation ─────────────────────────────────────────────────────
# 5 formats rotate daily by day-of-week to prevent repetition.
# Research shows varied formats increase follower retention by 40%.

POST_FORMATS = {
    0: {  # Monday — HOT TAKE
        "name": "Hot Take",
        "instruction": (
            "Write a bold, contrarian opinion about this AI topic that will spark debate.\n"
            "Format:\n"
            "• Line 1: Controversial statement that will divide readers (no '%, no 'Revolutionizing')\n"
            "• Line 2: BLANK LINE\n"
            "• Lines 3-4: Why most people are wrong about this\n"
            "• Line 5: BLANK LINE\n"
            "• Lines 6-7: The nuanced reality that few people talk about\n"
            "• Line 8: BLANK LINE\n"
            "• Line 9: One specific insight practitioners can use\n"
            "• Line 10: BLANK LINE\n"
            "• Line 11: A divisive question that forces a YES or NO answer\n\n"
            "Style: Direct, confident, slightly provocative. No hedging."
        ),
    },
    1: {  # Tuesday — BREAKDOWN
        "name": "Breakdown",
        "instruction": (
            "Write a crisp breakdown of what this AI development actually means in practice.\n"
            "Format:\n"
            "• Line 1: Specific fact or number from the topic (concrete, not vague)\n"
            "• Line 2: BLANK LINE\n"
            "• Line 3: What this means for AI engineers (specific job impact)\n"
            "• Line 4: BLANK LINE\n"
            "• Line 5: What this means for businesses (specific ROI angle)\n"
            "• Line 6: BLANK LINE\n"
            "• Line 7: The one thing most people will miss about this\n"
            "• Line 8: BLANK LINE\n"
            "• Line 9: A question that invites people to share their experience\n\n"
            "Style: Analytical, expert, punchy. Each line is its own paragraph."
        ),
    },
    2: {  # Wednesday — LESSON LEARNED (highest engagement day)
        "name": "Lesson",
        "instruction": (
            "Write as an AI practitioner sharing a genuine insight from this topic.\n"
            "Use first person sparingly — write as a senior engineer sharing wisdom.\n"
            "Format:\n"
            "• Line 1: A surprising or counterintuitive insight (NOT starting with 'I')\n"
            "• Line 2: BLANK LINE\n"
            "• Lines 3-4: Context — why this matters right now\n"
            "• Line 5: BLANK LINE\n"
            "• Lines 6-7: The deeper lesson or pattern this reveals\n"
            "• Line 8: BLANK LINE\n"
            "• Line 9: One concrete action or observation\n"
            "• Line 10: BLANK LINE\n"
            "• Line 11: A reflective question about their own experience\n\n"
            "Style: Thoughtful, experienced, genuine. Feels like expert advice."
        ),
    },
    3: {  # Thursday — MYTH BUSTER
        "name": "Myth Buster",
        "instruction": (
            "Bust a common misconception related to this AI topic.\n"
            "Format:\n"
            "• Line 1: State the common myth (what people believe) — make it specific\n"
            "• Line 2: BLANK LINE\n"
            "• Line 3: 'The reality:' — what is actually true\n"
            "• Line 4: BLANK LINE\n"
            "• Lines 5-6: Why the myth persists and who it hurts\n"
            "• Line 7: BLANK LINE\n"
            "• Lines 8-9: What practitioners should actually know\n"
            "• Line 10: BLANK LINE\n"
            "• Line 11: Ask readers which myths they've encountered\n\n"
            "Style: Authoritative, clear, educational. Like correcting a smart friend."
        ),
    },
    4: {  # Friday — PREDICTION (end-of-week forward-looking)
        "name": "Prediction",
        "instruction": (
            "Make a specific, bold prediction about where this AI trend is heading.\n"
            "Format:\n"
            "• Line 1: The prediction — specific, time-bounded, bold\n"
            "• Line 2: BLANK LINE\n"
            "• Lines 3-4: The signal from this topic that supports the prediction\n"
            "• Line 5: BLANK LINE\n"
            "• Lines 6-7: What changes if this prediction is right\n"
            "• Line 8: BLANK LINE\n"
            "• Line 9: What could make this prediction wrong (credibility builder)\n"
            "• Line 10: BLANK LINE\n"
            "• Line 11: Ask readers if they agree or disagree with the prediction\n\n"
            "Style: Confident, forward-thinking, data-backed. Specific dates/numbers."
        ),
    },
    5: {  # Saturday — QUICK INSIGHT
        "name": "Quick Insight",
        "instruction": (
            "Write a punchy, ultra-short insight post. Maximum 80 words total.\n"
            "Format:\n"
            "• Line 1: One bold sentence (the whole point of the post)\n"
            "• Line 2: BLANK LINE\n"
            "• Lines 3-5: Three ultra-short supporting points, each on its own line\n"
            "• Line 6: BLANK LINE\n"
            "• Line 7: One sharp question\n\n"
            "Style: Twitter-like brevity. Every word earns its place. No fluff."
        ),
    },
    6: {  # Sunday — DEEP DIVE
        "name": "Deep Dive",
        "instruction": (
            "Write a thoughtful, in-depth post that AI professionals will save and share.\n"
            "Format:\n"
            "• Line 1: A bold opening claim about the state of this technology\n"
            "• Line 2: BLANK LINE\n"
            "• Lines 3-5: The technical context explained for senior practitioners\n"
            "• Line 6: BLANK LINE\n"
            "• Lines 7-9: Real-world implications and use cases\n"
            "• Line 10: BLANK LINE\n"
            "• Lines 11-13: What this means for the next 12 months\n"
            "• Line 14: BLANK LINE\n"
            "• Line 15: A thought-provoking question for experts\n\n"
            "Style: Authoritative, detailed, save-worthy. The kind experts share with teams."
        ),
    },
}

# ─── Banned Words/Phrases ──────────────────────────────────────────────────────
# These patterns make posts feel generic or AI-written.
BANNED_PATTERNS = [
    r"\bRevolutioniz\w*\b",
    r"\bGame.chang\w*\b",
    r"\bLeverage\b",
    r"\bUtiliz\w*\b",
    r"\bSynerg\w*\b",
    r"\bTapestry\b",
    r"\bLandscape\b",
    r"\bDelve\b",
    r"\bUnlock\w*\b",
    r"^I ",                          # starting with I
    r"In today's (fast-paced|digital|AI-driven|rapidly evolving)",
    r"As an AI",
    r"Implement .{1,20} today",      # promotional CTA
    r"Try .{1,20} today",
    r"\d+% of \w+ (are|is|have|fail|can't)",  # overused stat hook
]

# ─── System Prompt ─────────────────────────────────────────────────────────────
CONTENT_SYSTEM_PROMPT = """\
You are a senior AI engineer and LinkedIn thought leader with 50,000+ followers.
Your posts consistently get 500+ comments because they spark genuine debate and
provide real practitioner value — not generic news summaries.

YOUR VOICE:
  • Direct and confident — you have strong opinions backed by experience
  • Specific over vague — you use real numbers, model names, benchmarks
  • Practitioner-focused — you write for engineers and technical leads, not executives
  • Genuinely curious — your questions make people want to answer
  • Honest about uncertainty — you say "I think" not "it is certain"

LINKEDIN FORMAT RULES (critical for algorithm reach):
  • Every paragraph is 1-2 sentences MAX
  • ALWAYS put a blank line between every paragraph
  • Never write walls of text — white space is your friend
  • The hook (first line) must work as a standalone sentence in the feed preview
  • Formatting: bold is NOT supported in standard posts — don't use **bold**

BANNED WORDS AND PHRASES — never use these:
  Revolutionizing, game-changing, leverage, utilize, synergy, tapestry,
  landscape, delve, unlock, "In today's world", "As an AI", "X% of Y fail",
  "Implement X today", "Try X today", "Download X", any promotional CTA

CLOSING QUESTION RULES:
  • Must be SPECIFIC to the topic, not generic
  • Must force a real answer (YES/NO, or choose-a-side)
  • NEVER: "What do you think?" or "Share your thoughts"
  • GOOD: "Are you using X in production — and would you trust it with Y?"

Write ONLY the post body. No labels, no metadata, no hashtags."""


# ─── User Prompt ──────────────────────────────────────────────────────────────
def _build_user_prompt(topic: dict, fmt: dict) -> str:
    return f"""Write a LinkedIn post about this AI topic using the format below.

TOPIC: {topic.get('title', 'AI Trends')}
CONTEXT: {topic.get('summary', '')}

FORMAT TO USE — {fmt['name']}:
{fmt['instruction']}

HARD REQUIREMENTS:
- Total length: 80 to {settings.POST_MAX_WORDS} words
- Every paragraph separated by a BLANK LINE (double newline)
- No hashtags (added separately)
- No bold/italic markdown
- No source names mentioned
- No promotional language

Write the post now:"""


def _get_todays_format() -> dict:
    """Return today's post format based on day of week (0=Mon, 6=Sun)."""
    day = date.today().weekday()
    fmt = POST_FORMATS[day]
    logger.info("Post format today: %s (day %d)", fmt['name'], day)
    return fmt


def _count_words(text: str) -> int:
    words = re.findall(r"\b\w+\b", text)
    return len([w for w in words if not w.startswith("#")])


def _count_paragraphs(text: str) -> int:
    return len([p for p in text.split("\n\n") if p.strip()])


def _ensure_paragraph_spacing(text: str) -> str:
    """Ensure every paragraph is separated by a blank line."""
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


def _check_banned_patterns(text: str) -> list[str]:
    """Return list of banned patterns found in text."""
    found = []
    for pattern in BANNED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            found.append(pattern)
    return found


def _trim_to_word_limit(text: str, max_words: int) -> str:
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


def _score_post(text: str, fmt_name: str) -> float:
    """
    Quality score 0–100.
    Calibrated to reflect LinkedIn algorithm priorities:
      - Format/readability: 30 pts
      - Hook quality: 25 pts
      - Word count: 20 pts
      - Closing question: 15 pts
      - No banned patterns: 10 pts
    """
    score = 20.0  # lower baseline — harder to get 100

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    first_line = paragraphs[0].split("\n")[0].lower() if paragraphs else ""
    last_para  = paragraphs[-1].lower() if paragraphs else ""

    # ── Format / readability (30 pts) ────────────────────────────────────────
    para_count = len(paragraphs)
    if 4 <= para_count <= 8:
        score += 30
    elif 3 <= para_count <= 9:
        score += 20
    elif para_count < 3:
        score -= 10

    # ── Hook quality (25 pts) ─────────────────────────────────────────────────
    power_openers = [
        "nobody", "stop", "here's why", "the truth", "most people",
        "the real", "unpopular opinion", "hot take", "myth:", "reality:",
        "this changes", "just announced", "breaking:", "the future",
        "why", "how", "what if", "imagine", "the problem", "warning",
    ]
    if any(pw in first_line for pw in power_openers):
        score += 15
    if re.search(r"\d+", first_line):
        score += 5
    if any(c in first_line for c in "?!"):
        score += 5

    # ── Word count (20 pts) ───────────────────────────────────────────────────
    wc = _count_words(text)
    if 80 <= wc <= settings.POST_MAX_WORDS:
        score += 20
    elif 60 <= wc < 80:
        score += 10
    elif wc < 60:
        score -= 15

    # ── Closing question quality (15 pts) ─────────────────────────────────────
    if "?" in last_para:
        # Bonus if question is specific (contains nouns/names)
        specific_q = re.search(r"\b(your|you|team|company|production|using|tried)\b", last_para)
        score += 15 if specific_q else 8

    # ── No banned patterns (10 pts) ───────────────────────────────────────────
    banned_found = _check_banned_patterns(text)
    score += max(0, 10 - len(banned_found) * 5)

    return max(0.0, min(100.0, score))


def generate_post(topic: dict, attempt: int = 1, fmt: dict = None) -> str:
    """
    Generate a single LinkedIn post for the given topic.

    Args:
        topic:   dict with 'title' and 'summary' keys.
        attempt: retry attempt number.
        fmt:     post format dict (uses today's format if None).

    Returns:
        Post text string (without hashtags).
    """
    if fmt is None:
        fmt = _get_todays_format()

    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=min(settings.GROQ_TEMPERATURE + 0.1, 0.95),
        max_tokens=settings.GROQ_MAX_TOKENS,
    )

    if attempt == 2:
        # Retry with stricter length directive
        retry_prompt = (
            f"Write a LinkedIn post about: {topic.get('title', 'AI')}\n"
            f"Context: {topic.get('summary', '')}\n\n"
            f"STRICT REQUIREMENTS:\n"
            f"- Write EXACTLY 5 paragraphs\n"
            f"- BLANK LINE between every paragraph\n"
            f"- Each paragraph: 1-2 sentences only\n"
            f"- Total: 80 to {settings.POST_MAX_WORDS} words\n"
            f"- First line: bold claim or surprising fact\n"
            f"- Last line: specific question (not 'What do you think?')\n"
            f"- NO hashtags, NO bold markdown, NO source names\n"
            f"Write now:"
        )
        messages = [HumanMessage(content=retry_prompt)]
    else:
        messages = [
            SystemMessage(content=CONTENT_SYSTEM_PROMPT),
            HumanMessage(content=_build_user_prompt(topic, fmt)),
        ]

    try:
        response = llm.invoke(messages)
        post_text = response.content.strip()
    except Exception as exc:
        logger.error("Content generation failed (attempt %d): %s", attempt, exc)
        raise

    post_text = _ensure_paragraph_spacing(post_text)
    post_text = _trim_to_word_limit(post_text, settings.POST_MAX_WORDS)

    # Check for banned patterns
    banned = _check_banned_patterns(post_text)
    if banned:
        logger.warning("Banned patterns found: %s", banned)

    wc = _count_words(post_text)
    logger.info(
        "Generated post: %d words, attempt %d, format '%s', topic '%s'",
        wc, attempt, fmt['name'], topic.get("title", "")[:50],
    )

    if wc < 60 and attempt == 1:
        logger.warning("Post too short (%d words) — retrying.", wc)
        return generate_post(topic, attempt=2, fmt=fmt)

    return post_text


def generate_best_post(topics: list) -> dict:
    """
    Generate posts for up to MULTI_TOPIC_DRAFTS topics,
    score each, return the highest quality draft.

    Returns dict: {post_text, topic, score, format_name}
    """
    n_drafts = min(settings.MULTI_TOPIC_DRAFTS, len(topics))
    if n_drafts == 0:
        raise ValueError("No topics provided.")

    fmt = _get_todays_format()  # Same format for all drafts in one run
    drafts = []

    for i, topic in enumerate(topics[:n_drafts]):
        try:
            text  = generate_post(topic, fmt=fmt)
            score = _score_post(text, fmt['name'])
            drafts.append({
                "post_text":   text,
                "topic":       topic,
                "score":       score,
                "format_name": fmt['name'],
            })
            logger.info(
                "Draft %d/%d — '%s' → score %.1f",
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
        "Best draft: '%s' (score %.1f, format: %s)",
        best["topic"].get("title", "")[:50],
        best["score"],
        best["format_name"],
    )
    return best