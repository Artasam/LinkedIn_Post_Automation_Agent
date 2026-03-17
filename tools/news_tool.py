"""
tools/news_tool.py
------------------
SMART NEWS FETCHER — Replaces RSS entirely.

Uses 5 independent, highly reliable sources that work even when
RSS feeds are blocked. Sources are tried in priority order and
results are merged, deduplicated, and ranked.

Sources (in priority order):
  1. ArXiv API          — Official XML API, never blocked, no key needed
  2. NewsAPI            — Real-time AI news (free tier: 100 req/day)
  3. HackerNews API     — Algolia search API, always accessible
  4. Wikipedia Recent   — Recent AI changes via MediaWiki API
  5. DuckDuckGo Search  — Zero-click info API, no key needed

All sources use direct REST APIs (not RSS/HTML scraping),
making them resilient to network restrictions.
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from typing import Optional
from urllib.parse import quote_plus

import requests

from config import settings

logger = logging.getLogger(__name__)

# ── Shared HTTP session with retry-friendly headers ───────────────────────────
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/xml, */*",
    "Accept-Language": "en-US,en;q=0.9",
})

TIMEOUT = 12  # seconds per request


def _clean(text: Optional[str]) -> str:
    """Strip HTML tags, extra whitespace, and unicode junk."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    return text.strip()


def _make_article(title: str, summary: str, url: str,
                  source: str, weight: int,
                  published: Optional[datetime] = None) -> dict:
    """Standardised article dict matching the shape expected by topic_agent."""
    return {
        "title":     _clean(title)[:200],
        "summary":   _clean(summary)[:400],
        "url":       url,
        "source":    source,
        "weight":    weight,
        "published": published or datetime.now(timezone.utc),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — ArXiv API (Official XML API — most reliable, no key needed)
# ══════════════════════════════════════════════════════════════════════════════

ARXIV_API = "https://export.arxiv.org/api/query"

ARXIV_QUERIES = [
    "cat:cs.AI+AND+ti:large+language+model",
    "cat:cs.AI+AND+ti:agent",
    "cat:cs.LG+AND+ti:transformer",
    "cat:cs.AI+AND+ti:multimodal",
    "cat:cs.AI+AND+ti:reasoning",
]


def fetch_arxiv(max_results: int = 8) -> list[dict]:
    """
    Fetch latest AI research papers from ArXiv official API.
    Returns papers from the last 7 days sorted by submission date.
    """
    articles = []
    query = (
        "search_query=cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CL"
        "&sortBy=submittedDate&sortOrder=descending"
        f"&max_results={max_results}"
    )
    url = f"{ARXIV_API}?{query}"

    try:
        resp = SESSION.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("ArXiv API failed: %s", exc)
        return articles

    try:
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns):
            title = entry.findtext("atom:title", "", ns).replace("\n", " ").strip()
            summary = entry.findtext("atom:summary", "", ns).replace("\n", " ").strip()
            link_el = entry.find("atom:id", ns)
            link = link_el.text if link_el is not None else ""
            published_str = entry.findtext("atom:published", "", ns)

            # Parse ISO date
            try:
                pub = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            except Exception:
                pub = datetime.now(timezone.utc)

            # Skip papers older than 7 days
            if (datetime.now(timezone.utc) - pub).days > 7:
                continue

            # Get author names
            authors = [
                a.findtext("atom:name", "", ns)
                for a in entry.findall("atom:author", ns)
            ]
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."

            articles.append(_make_article(
                title=f"[ArXiv] {title}",
                summary=f"{summary[:300]} — Authors: {author_str}",
                url=link,
                source="ArXiv Research",
                weight=4,          # highest weight — peer-reviewed research
                published=pub,
            ))

    except ET.ParseError as exc:
        logger.warning("ArXiv XML parse error: %s", exc)

    logger.info("ArXiv: fetched %d papers", len(articles))
    return articles


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — HackerNews via Algolia API (always accessible, no key needed)
# ══════════════════════════════════════════════════════════════════════════════

HN_API = "https://hn.algolia.com/api/v1/search"

HN_QUERIES = [
    "LLM",
    "GPT",
    "Claude",
    "Llama",
    "machine learning",
    "neural",
    "AI",
]


def fetch_hackernews(max_results: int = 10) -> list[dict]:
    """
    Fetch top AI stories from HackerNews via Algolia search API.
    Filters for stories with significant upvotes (score > 50).
    """
    articles = []
    seen_titles: set[str] = set()

    # Search multiple queries, collect unique high-score stories
    for query in HN_QUERIES[:3]:
        try:
            resp = SESSION.get(
                HN_API,
                params={
                    "query": query,
                    "tags": "story",
                    "numericFilters": "points>20",   # lowered from 50 — more results
                    "hitsPerPage": 5,
                    "attributesToRetrieve": "title,url,points,num_comments,created_at,objectID",
                },
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            for hit in data.get("hits", []):
                title = hit.get("title", "")
                if not title or title.lower() in seen_titles:
                    continue
                seen_titles.add(title.lower())

                url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
                points = hit.get("points", 0)
                comments = hit.get("num_comments", 0)

                # Parse date
                try:
                    pub = datetime.fromisoformat(
                        hit.get("created_at", "").replace("Z", "+00:00")
                    )
                except Exception:
                    pub = datetime.now(timezone.utc)

                # Skip items older than 7 days
                if (datetime.now(timezone.utc) - pub).total_seconds() > 604800:
                    continue

                articles.append(_make_article(
                    title=f"[HN] {title}",
                    summary=f"HackerNews discussion — {points} points, {comments} comments. {title}",
                    url=url,
                    source="Hacker News",
                    weight=3,
                    published=pub,
                ))

                if len(articles) >= max_results:
                    break

        except requests.RequestException as exc:
            logger.warning("HackerNews query '%s' failed: %s", query, exc)
            continue

        if len(articles) >= max_results:
            break

    logger.info("HackerNews: fetched %d stories", len(articles))
    return articles


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 3 — NewsAPI (free tier: 100 req/day, key optional)
# ══════════════════════════════════════════════════════════════════════════════

NEWS_API = "https://newsapi.org/v2/everything"

NEWS_QUERIES = [
    "artificial intelligence AI 2025",
    "large language model LLM",
    "OpenAI OR Anthropic OR Google DeepMind",
    "machine learning breakthrough",
]


def fetch_newsapi(max_results: int = 8) -> list[dict]:
    """
    Fetch AI news from NewsAPI.
    Requires NEWS_API_KEY in environment (free tier: 100 req/day).
    Skipped silently if key not configured.
    """
    import os
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key:
        logger.info("NEWS_API_KEY not set — skipping NewsAPI source.")
        return []

    articles = []
    seen: set[str] = set()
    since = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

    for query in NEWS_QUERIES[:2]:
        try:
            resp = SESSION.get(
                NEWS_API,
                params={
                    "q": query,
                    "from": since,
                    "sortBy": "publishedAt",
                    "language": "en",
                    "pageSize": 5,
                    "apiKey": api_key,
                },
                timeout=TIMEOUT,
            )
            if resp.status_code == 401:
                logger.warning("NewsAPI: Invalid API key.")
                break
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("articles", []):
                title = item.get("title", "") or ""
                if not title or "[Removed]" in title or title.lower() in seen:
                    continue
                seen.add(title.lower())

                try:
                    pub = datetime.fromisoformat(
                        item.get("publishedAt", "").replace("Z", "+00:00")
                    )
                except Exception:
                    pub = datetime.now(timezone.utc)

                articles.append(_make_article(
                    title=title,
                    summary=item.get("description") or item.get("content") or title,
                    url=item.get("url", ""),
                    source=item.get("source", {}).get("name", "News"),
                    weight=3,
                    published=pub,
                ))

        except requests.RequestException as exc:
            logger.warning("NewsAPI query failed: %s", exc)
            continue

    logger.info("NewsAPI: fetched %d articles", len(articles))
    return articles


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 4 — Wikipedia Current Events + Recent AI Articles (always accessible)
# ══════════════════════════════════════════════════════════════════════════════

WIKI_API = "https://en.wikipedia.org/w/api.php"

WIKI_SEARCH_TERMS = [
    "artificial intelligence 2025",
    "large language model",
    "generative AI",
    "AI safety",
    "machine learning",
]


def fetch_wikipedia(max_results: int = 5) -> list[dict]:
    """
    Fetch recently updated AI-related Wikipedia articles.
    Uses MediaWiki API — always accessible, no key needed.
    """
    articles = []
    seen: set[str] = set()

    for term in WIKI_SEARCH_TERMS[:3]:
        try:
            resp = SESSION.get(
                WIKI_API,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": term,
                    "srnamespace": 0,
                    "srlimit": 3,
                    "srprop": "snippet|timestamp|wordcount",
                    "format": "json",
                    "utf8": 1,
                },
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("query", {}).get("search", []):
                title = item.get("title", "")
                if not title or title.lower() in seen:
                    continue
                seen.add(title.lower())

                snippet = _clean(item.get("snippet", ""))
                page_url = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"

                try:
                    pub = datetime.fromisoformat(
                        item.get("timestamp", "").replace("Z", "+00:00")
                    )
                except Exception:
                    pub = datetime.now(timezone.utc)

                articles.append(_make_article(
                    title=f"[Wikipedia] {title}",
                    summary=f"Wikipedia: {snippet}",
                    url=page_url,
                    source="Wikipedia",
                    weight=2,
                    published=pub,
                ))

                if len(articles) >= max_results:
                    break

        except requests.RequestException as exc:
            logger.warning("Wikipedia API query failed: %s", exc)
            continue

        if len(articles) >= max_results:
            break

    logger.info("Wikipedia: fetched %d articles", len(articles))
    return articles


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 5 — DuckDuckGo Instant Answer API (no key, no rate limit)
# ══════════════════════════════════════════════════════════════════════════════

DDG_API = "https://api.duckduckgo.com/"

DDG_QUERIES = [
    "AI artificial intelligence",
    "large language model LLM",
    "machine learning deep learning",
    "OpenAI Anthropic Google AI",
    "neural network research",
]


def fetch_duckduckgo(max_results: int = 5) -> list[dict]:
    """
    Fetch AI topic summaries from DuckDuckGo Instant Answer API.
    No API key required, no rate limits, always accessible.
    Returns related topics and instant answers about AI.
    """
    articles = []
    seen: set[str] = set()

    for query in DDG_QUERIES[:3]:
        try:
            resp = SESSION.get(
                DDG_API,
                params={
                    "q": query,
                    "format": "json",
                    "no_redirect": 1,
                    "no_html": 1,
                    "skip_disambig": 1,
                },
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            # Main abstract
            abstract = data.get("AbstractText", "")
            abstract_title = data.get("Heading", "")
            abstract_url = data.get("AbstractURL", "")

            if abstract and abstract_title and abstract_title.lower() not in seen:
                seen.add(abstract_title.lower())
                articles.append(_make_article(
                    title=f"[DDG] {abstract_title}",
                    summary=abstract[:400],
                    url=abstract_url,
                    source="DuckDuckGo Knowledge",
                    weight=2,
                ))

            # Related topics
            for topic in data.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict) and topic.get("Text"):
                    text = topic["Text"]
                    first_sentence = text.split(".")[0]
                    topic_url = topic.get("FirstURL", "")
                    key = first_sentence.lower()[:50]

                    if key and key not in seen:
                        seen.add(key)
                        articles.append(_make_article(
                            title=f"[DDG Topic] {first_sentence[:100]}",
                            summary=text[:300],
                            url=topic_url,
                            source="DuckDuckGo Knowledge",
                            weight=1,
                        ))

            if len(articles) >= max_results:
                break

            time.sleep(0.3)  # gentle rate limiting

        except requests.RequestException as exc:
            logger.warning("DuckDuckGo query '%s' failed: %s", query, exc)
            continue

    logger.info("DuckDuckGo: fetched %d items", len(articles))
    return articles


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 6 — GitHub Trending (AI repos via GitHub API — no key for public data)
# ══════════════════════════════════════════════════════════════════════════════

GITHUB_API = "https://api.github.com/search/repositories"


def fetch_github_trending(max_results: int = 5) -> list[dict]:
    """
    Fetch trending AI/ML GitHub repositories created in the last 7 days.
    Uses GitHub public search API — no authentication needed for basic search.
    """
    articles = []
    since_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

    ai_topics = [
        "large-language-model",
        "AI agent",
        "machine-learning 2025",
    ]

    for topic in ai_topics[:2]:
        try:
            resp = SESSION.get(
                GITHUB_API,
                params={
                    "q": f"{topic} created:>{since_date}",
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 3,
                },
                headers={"Accept": "application/vnd.github.v3+json"},
                timeout=TIMEOUT,
            )

            if resp.status_code == 403:
                logger.info("GitHub API rate limit reached — skipping.")
                break
            resp.raise_for_status()
            data = resp.json()

            for repo in data.get("items", []):
                name = repo.get("full_name", "")
                description = repo.get("description") or ""
                stars = repo.get("stargazers_count", 0)
                lang = repo.get("language") or "Unknown"
                url = repo.get("html_url", "")

                try:
                    pub = datetime.fromisoformat(
                        repo.get("created_at", "").replace("Z", "+00:00")
                    )
                except Exception:
                    pub = datetime.now(timezone.utc)

                if stars < 50:
                    continue

                articles.append(_make_article(
                    title=f"[GitHub] {name} — {stars:,} stars",
                    summary=(
                        f"New AI open-source project: {description}. "
                        f"Language: {lang}. Stars: {stars:,}. "
                        f"This repository has gained significant community attention."
                    ),
                    url=url,
                    source="GitHub Trending",
                    weight=3,
                    published=pub,
                ))

        except requests.RequestException as exc:
            logger.warning("GitHub API failed: %s", exc)
            break

    logger.info("GitHub Trending: fetched %d repos", len(articles))
    return articles


# ══════════════════════════════════════════════════════════════════════════════
# MASTER FETCHER — Tries all sources, merges and ranks results
# ══════════════════════════════════════════════════════════════════════════════

def fetch_all_news(max_per_source: int = 8) -> list[dict]:
    """
    Fetch AI news from all available sources in parallel-friendly sequence.

    Source priority order:
      1. ArXiv API          (weight 4) — research papers
      2. HackerNews Algolia (weight 3) — tech community discussions
      3. GitHub Trending    (weight 3) — new AI repositories
      4. NewsAPI            (weight 3) — news articles (key optional)
      5. Wikipedia          (weight 2) — encyclopedic context
      6. DuckDuckGo         (weight 1-2) — instant answers

    Returns deduplicated, recency+weight sorted article list.
    Falls back gracefully — if a source fails, others fill in.
    """
    all_articles: list[dict] = []
    source_results: dict[str, int] = {}

    fetchers = [
        ("ArXiv",       fetch_arxiv,           max_per_source),
        ("HackerNews",  fetch_hackernews,       max_per_source),
        ("GitHub",      fetch_github_trending,  5),
        ("NewsAPI",     fetch_newsapi,          max_per_source),
        ("Wikipedia",   fetch_wikipedia,        5),
        ("DuckDuckGo",  fetch_duckduckgo,       5),
    ]

    for name, fetcher, limit in fetchers:
        try:
            results = fetcher(limit)
            all_articles.extend(results)
            source_results[name] = len(results)
        except Exception as exc:
            logger.warning("Source '%s' raised unexpected error: %s", name, exc)
            source_results[name] = 0

    # Log summary
    total = len(all_articles)
    summary = " | ".join(f"{k}:{v}" for k, v in source_results.items() if v > 0)
    logger.info("News fetch complete: %d total articles [%s]", total, summary)

    if total == 0:
        logger.warning("ALL news sources returned 0 articles — using fallback topics.")
        return []

    # Deduplicate by title similarity (first 60 chars, lowercased)
    seen_keys: set[str] = set()
    unique: list[dict] = []
    for article in all_articles:
        key = re.sub(r"[\[\]()]", "", article["title"]).lower().strip()[:60]
        if key and key not in seen_keys:
            seen_keys.add(key)
            unique.append(article)

    # Sort by: weight (desc) then recency (desc)
    now = datetime.now(timezone.utc)
    def sort_key(a: dict) -> tuple:
        pub = a.get("published", now)
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        age_hours = (now - pub).total_seconds() / 3600
        recency = max(0.0, 1.0 - age_hours / 72)
        return (a.get("weight", 1) * 2 + recency, recency)

    unique.sort(key=sort_key, reverse=True)

    logger.info("Unique articles after deduplication: %d", len(unique))
    return unique