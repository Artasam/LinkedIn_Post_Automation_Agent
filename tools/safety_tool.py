"""
tools/safety_tool.py
--------------------
Content Safety Filter — guards every generated post before it reaches LinkedIn.

Enforces:
  • LinkedIn Professional Community Guidelines
  • Factual accuracy (no false/exaggerated AI claims)
  • No harmful, offensive, or misleading content
  • No political propaganda, hate speech, or misinformation
  • Professional, educational, and informative tone only

Pipeline position: runs between generate_content and generate_hashtags.
If a post fails safety checks it is regenerated (up to MAX_RETRIES times).
If all retries fail the pipeline falls back to a guaranteed-safe template post.

Safety checks run in two tiers:
  Tier 1 — Rule-based (instant, no API call):
    Checks for banned phrases, prohibited patterns, word limits
  Tier 2 — LLM-based moderation (only if Tier 1 passes):
    Asks the same LLM to self-moderate the post against guidelines
"""

import logging
import re
from dataclasses import dataclass, field

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 3          # Max regeneration attempts before using fallback
MIN_WORDS   = 40         # Post too short = likely an error
MAX_WORDS   = 300        # Hard ceiling (matches POST_MAX_WORDS)
MIN_HASHTAGS_IN_POST = 0 # Hashtags are added separately — post body should have 0


# ─── Tier 1: Rule-Based Banned Patterns ───────────────────────────────────────

# Phrases that indicate harmful, misleading or off-topic content
BANNED_PHRASES: list[str] = [
    # False / exaggerated AI claims
    "ai is sentient", "ai is conscious", "ai has feelings", "ai is alive",
    "ai will replace all humans", "ai will destroy humanity",
    "ai is smarter than all humans", "artificial general intelligence is here",
    "agi has been achieved", "ai has achieved agi",
    "100% accurate", "never makes mistakes", "perfect ai",

    # Misinformation / conspiracy
    "ai controls the government", "ai is a hoax", "ai is fake",
    "big tech is hiding", "they don't want you to know",
    "secret algorithm", "suppressed by",

    # Political / hate speech
    "hate speech", "political propaganda",
    "white supremac", "black lives don't", "islamophob",
    "antisemit", "racist",

    # Inappropriate / adult content markers
    "nsfw", "explicit content", "adult content",

    # Spam / scam markers
    "click here to make money", "earn $", "get rich quick",
    "dm me for details", "link in bio", "limited time offer",
    "act now", "buy now",

    # Dangerous content
    "how to hack", "how to exploit", "bypass security",
    "steal data", "create malware", "create virus",
]

# Regex patterns for structural safety violations
BANNED_PATTERNS: list[str] = [
    r"\b(100|99\.9)\s*%\s*(accurate|correct|perfect|reliable)\b",
    r"\bguarantee[sd]?\s+(results?|success|accuracy)\b",
    r"\b(will|can)\s+(definitely|certainly|always)\s+(solve|fix|replace)\b",
    r"\bAI\s+(is|will be)\s+(god|divine|omniscient|infallible)\b",
    r"\bmake\s+\$[\d,]+\b",
    r"\bfollow\s+for\s+more\b",
    r"\blike\s+and\s+share\b",
    r"\bgo\s+viral\b",
    r"(\bDM\b|\bdirect\s+message\b).{0,20}(me|us|now)",
]


@dataclass
class SafetyResult:
    """Result from the content safety check."""
    passed:   bool
    reasons:  list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.passed:
            return f"SAFE (warnings: {len(self.warnings)})"
        return f"UNSAFE — {'; '.join(self.reasons)}"


# ─── Tier 1: Rule-Based Check ─────────────────────────────────────────────────

def check_rules(text: str) -> SafetyResult:
    """
    Fast rule-based safety check. Runs in microseconds, no API call needed.

    Checks:
    - Word count (too short / too long)
    - Banned phrases (case-insensitive)
    - Banned regex patterns
    - Promotional / spam signals
    - All-caps abuse
    """
    reasons: list[str] = []
    warnings: list[str] = []
    text_lower = text.lower()

    # ── Word count ────────────────────────────────────────────────────────────
    word_count = len(re.findall(r"\b\w+\b", text))
    if word_count < MIN_WORDS:
        reasons.append(f"Post too short ({word_count} words, minimum {MIN_WORDS})")
    if word_count > MAX_WORDS:
        warnings.append(f"Post exceeds word limit ({word_count}/{MAX_WORDS})")

    # ── Banned phrases ────────────────────────────────────────────────────────
    for phrase in BANNED_PHRASES:
        if phrase in text_lower:
            reasons.append(f"Banned phrase detected: '{phrase}'")

    # ── Banned patterns ───────────────────────────────────────────────────────
    for pattern in BANNED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            reasons.append(f"Banned pattern detected: '{pattern[:40]}'")

    # ── Excessive capitalisation (shouting / spam) ────────────────────────────
    words = text.split()
    caps_words = [w for w in words if w.isupper() and len(w) > 2]
    if len(caps_words) > 5:
        warnings.append(
            f"Excessive ALL-CAPS detected ({len(caps_words)} words) — "
            "may appear unprofessional"
        )

    # ── Emoji overload ────────────────────────────────────────────────────────
    emoji_count = len(re.findall(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]",
        text
    ))
    if emoji_count > 6:
        warnings.append(
            f"High emoji count ({emoji_count}) — "
            "LinkedIn professional tone recommends < 6 emojis"
        )

    # ── Repetitive exclamation marks ─────────────────────────────────────────
    if text.count("!") > 4:
        warnings.append(
            f"Excessive exclamation marks ({text.count('!')}) — "
            "reduces professional credibility"
        )

    # ── Must contain a question (engagement requirement) ─────────────────────
    if "?" not in text:
        warnings.append("No closing question found — reduces engagement potential")

    passed = len(reasons) == 0
    return SafetyResult(passed=passed, reasons=reasons, warnings=warnings)


# ─── Tier 2: LLM Self-Moderation ──────────────────────────────────────────────

SAFETY_SYSTEM_PROMPT = """\
You are a strict LinkedIn Content Safety Moderator.
Your job is to evaluate whether a LinkedIn post meets professional standards.

You must check for:
1. FACTUAL ACCURACY — no false or exaggerated AI claims
2. PROFESSIONALISM — suitable for a business audience
3. LINKEDIN POLICY — no spam, hate speech, propaganda, or adult content
4. EDUCATIONAL VALUE — the post must inform or educate the reader
5. TONE — professional, respectful, and constructive

Respond ONLY in this exact format:
VERDICT: SAFE or UNSAFE
REASON: <one sentence explaining your verdict>
SUGGESTION: <one sentence on how to improve, or "None" if already good>
"""

SAFETY_USER_PROMPT = """\
Evaluate this LinkedIn post for content safety and professional quality:

---
{post_text}
---

Topic: {topic_title}

Apply the 5 evaluation criteria strictly. Return VERDICT, REASON, and SUGGESTION only.
"""


def check_llm_moderation(post_text: str, topic_title: str) -> SafetyResult:
    """
    LLM-based self-moderation. Asks the model to evaluate its own output.
    Returns SafetyResult with LLM verdict.
    """
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0.0,    # deterministic moderation — no creativity
        max_tokens=200,
    )

    messages = [
        SystemMessage(content=SAFETY_SYSTEM_PROMPT),
        HumanMessage(content=SAFETY_USER_PROMPT.format(
            post_text=post_text,
            topic_title=topic_title,
        )),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
    except Exception as exc:
        logger.warning("LLM moderation call failed: %s — assuming SAFE", exc)
        return SafetyResult(passed=True, warnings=["LLM moderation skipped (API error)"])

    # Parse the structured response
    verdict = "SAFE"
    reason = ""
    suggestion = ""

    for line in raw.splitlines():
        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip().upper()
        elif line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()
        elif line.startswith("SUGGESTION:"):
            suggestion = line.replace("SUGGESTION:", "").strip()

    passed = verdict == "SAFE"

    result = SafetyResult(passed=passed)
    if not passed:
        result.reasons.append(f"LLM moderation: {reason}")
    if suggestion and suggestion.lower() != "none":
        result.warnings.append(f"Suggestion: {suggestion}")

    logger.info(
        "LLM moderation verdict: %s | Reason: %s",
        verdict, reason or "N/A"
    )
    return result


# ─── Combined Safety Check ────────────────────────────────────────────────────

def run_safety_check(post_text: str, topic_title: str = "") -> SafetyResult:
    """
    Run both tiers of safety checks on generated post content.

    Tier 1 (rule-based) runs first — fast and free.
    Tier 2 (LLM moderation) runs only if Tier 1 passes.

    Args:
        post_text:    The generated LinkedIn post text.
        topic_title:  The source topic title (used in LLM moderation context).

    Returns:
        SafetyResult with passed=True if all checks pass.
    """
    logger.info("Running content safety check…")

    # Tier 1: Rule-based
    tier1 = check_rules(post_text)
    if tier1.warnings:
        for w in tier1.warnings:
            logger.warning("Safety warning: %s", w)

    if not tier1.passed:
        for r in tier1.reasons:
            logger.error("Safety violation (Tier 1): %s", r)
        return tier1

    logger.info("Tier 1 (rule-based): PASSED")

    # Tier 2: LLM moderation
    tier2 = check_llm_moderation(post_text, topic_title)
    if tier2.warnings:
        for w in tier2.warnings:
            logger.warning("Safety warning (LLM): %s", w)

    if not tier2.passed:
        for r in tier2.reasons:
            logger.error("Safety violation (Tier 2): %s", r)
        # Merge reasons from both tiers
        combined = SafetyResult(
            passed=False,
            reasons=tier1.reasons + tier2.reasons,
            warnings=tier1.warnings + tier2.warnings,
        )
        return combined

    logger.info("Tier 2 (LLM moderation): PASSED")

    # All clear
    return SafetyResult(
        passed=True,
        reasons=[],
        warnings=tier1.warnings + tier2.warnings,
    )


# ─── Safe Fallback Template ───────────────────────────────────────────────────

SAFE_FALLBACK_POSTS: list[str] = [
    """\
AI is transforming how we build software — and the pace is accelerating.

Retrieval-Augmented Generation (RAG) has become the default architecture for \
enterprise AI applications. Instead of retraining large models, teams connect \
LLMs to live knowledge bases — cutting costs and keeping outputs up to date.

The practical takeaway: start small. Pick one internal knowledge base, \
build a RAG pipeline around it, and measure the accuracy improvement. \
Most teams see results within a sprint.

What is the biggest challenge your team faces when deploying AI in production?""",

    """\
Open-source AI is closing the gap with proprietary models faster than anyone expected.

Models like Llama 3, Mistral, and Qwen now benchmark within striking distance of \
GPT-4 on reasoning tasks — and they run on hardware your team already owns. \
The barrier to building custom AI solutions has never been lower.

One actionable step: evaluate an open-source model on your specific use case this \
week. The results may surprise you.

Which open-source AI model has impressed you the most recently?""",

    """\
AI agents are moving from demos into production — and the architecture choices matter.

The most reliable agentic systems share three traits: clear tool boundaries, \
structured output validation, and graceful failure handling. Getting any one \
of these wrong compounds into unpredictable behavior at scale.

Start with the simplest possible agent that solves a real problem. \
Complexity can always be added later.

What is the hardest engineering challenge you have encountered when building AI agents?""",
]


def get_safe_fallback(index: int = 0) -> str:
    """
    Return a guaranteed-safe, pre-written LinkedIn post.
    Used when all LLM generation + retry attempts fail safety checks.
    """
    idx = index % len(SAFE_FALLBACK_POSTS)
    logger.warning(
        "Using safe fallback post #%d — all regeneration attempts failed safety.",
        idx + 1,
    )
    return SAFE_FALLBACK_POSTS[idx]
