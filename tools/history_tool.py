"""
tools/history_tool.py
---------------------
Post History Tracker — prevents the agent from repeating a topic.

Saves a rolling log of the last 30 published post topics to post_history.json.
Before topic selection the agent calls filter_fresh_topics() which loads the
history ONCE and filters the entire list in a single pass — not once per article.

Performance fix: history file is loaded ONCE per filter operation,
not once per article (which caused 26 redundant file reads in logs).
"""

import json
import logging
import re
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

HISTORY_FILE    = Path(os.getenv("HISTORY_FILE", "post_history.json"))
MAX_HISTORY     = 30
SIMILARITY_THRESHOLD = 0.6


def _title_key(title: str) -> str:
    """Normalise title for comparison — lowercase, no punctuation, no source prefix."""
    title = re.sub(r"[\[\]():#\-–—]", " ", title.lower())
    title = re.sub(r"^\s*(arxiv|hn|github|ddg|wikipedia)[^\]]*\]\s*", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title[:120]


def _similarity(a: str, b: str) -> float:
    """Word-overlap Jaccard similarity between two title strings (0.0–1.0)."""
    stops = {
        "the","a","an","in","of","for","on","with","and","or","is","are",
        "was","to","how","why","what","new","using","via","from","by","as",
        "at","its","it","this","that","these","those","can","will","be",
    }
    words_a = set(_title_key(a).split()) - stops
    words_b = set(_title_key(b).split()) - stops
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def load_history() -> list[dict]:
    """
    Load post history from disk.
    Returns empty list if file doesn't exist or is corrupted.
    """
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            logger.info("Loaded %d entries from post history.", len(data))
            return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read post history: %s", exc)
    return []


def save_history(history: list[dict]) -> None:
    """Save history to disk, keeping only the last MAX_HISTORY entries."""
    try:
        trimmed = history[-MAX_HISTORY:]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(trimmed, f, indent=2, default=str)
        logger.info("Post history saved (%d entries).", len(trimmed))
    except OSError as exc:
        logger.warning("Could not save post history: %s", exc)


def record_post(topic: dict, post_id: str = "") -> None:
    """
    Add a successfully published topic to the history file.
    Called by agent_graph.py after a successful LinkedIn publish.
    """
    history = load_history()
    entry = {
        "title":        topic.get("title", ""),
        "source":       topic.get("source", ""),
        "post_id":      post_id,
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    history.append(entry)
    save_history(history)
    logger.info("Recorded post to history: '%s'", entry["title"][:80])


def was_used_recently(title: str, history: list[dict] | None = None) -> bool:
    """
    Check if a title is too similar to any recently published post.

    Args:
        title:   The candidate topic title.
        history: Pre-loaded history list. If None, loads from disk.
                 Pass the pre-loaded list to avoid repeated file reads.

    Returns:
        True if the topic should be skipped (similarity ≥ 60%).
    """
    if history is None:
        history = load_history()

    for entry in history:
        past_title = entry.get("title", "")
        sim = _similarity(title, past_title)
        if sim >= SIMILARITY_THRESHOLD:
            logger.info(
                "Skipping '%s' (%.0f%% similar to past post '%s')",
                title[:60],
                sim * 100,
                past_title[:60],
            )
            return True
    return False


def filter_fresh_topics(topics: list[dict]) -> list[dict]:
    """
    Filter topic list, removing any recently-published topics.

    PERFORMANCE FIX: Loads history ONCE, then passes it to every
    was_used_recently() call. This prevents the N×file-read problem
    that caused 26 redundant log lines in the previous version.

    Args:
        topics: List of topic dicts with at least a 'title' key.

    Returns:
        Filtered list. Falls back to full list if all topics are used.
    """
    # ── Load history ONCE for the entire filter operation ─────────────────────
    history = load_history()

    if not history:
        logger.info("No post history found — all %d topics are fresh.", len(topics))
        return topics

    # ── Single-pass filter using the pre-loaded history ───────────────────────
    fresh = [
        t for t in topics
        if not was_used_recently(t.get("title", ""), history=history)
    ]

    if not fresh:
        logger.warning(
            "All %d candidate topics were recently used. "
            "Using full list to avoid empty pipeline.",
            len(topics),
        )
        return topics

    skipped = len(topics) - len(fresh)
    if skipped > 0:
        logger.info(
            "History filter: removed %d recently-used topic(s), "
            "%d fresh topics remain.",
            skipped,
            len(fresh),
        )
    return fresh


def get_history_summary() -> str:
    """Return a human-readable summary of the last 10 published posts."""
    history = load_history()
    if not history:
        return "No post history yet."
    lines = [f"Last {min(len(history), 10)} published topics:"]
    for i, entry in enumerate(reversed(history[-10:]), 1):
        date  = entry.get("published_at", "")[:10]
        title = entry.get("title", "")[:70]
        lines.append(f"  {i:2}. [{date}] {title}")
    return "\n".join(lines)