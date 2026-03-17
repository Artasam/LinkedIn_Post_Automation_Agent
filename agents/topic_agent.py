"""
agents/topic_agent.py
---------------------
Topic Agent: Fetches trending AI topics from multiple reliable sources
(ArXiv, HackerNews, GitHub, NewsAPI, Wikipedia, DuckDuckGo), ranks them
by relevance and recency, filters out recently-published topics using
post history, then uses the Groq LLM to select the best candidate.

Key anti-repetition features:
  - Post history filter (history_tool.py) skips recently used topics
  - Weighted random selection adds variety even among top-ranked topics
  - Daily rotating fallback topics (10 unique topics, day-based rotation)
  - LLM instructed to prefer novel/specific research over generic AI topics
"""

import logging
import random
from datetime import datetime, timezone

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from tools.news_tool import fetch_all_news
from tools.history_tool import filter_fresh_topics, get_history_summary
from config import settings

logger = logging.getLogger(__name__)

# ─── AI Relevance Keywords ─────────────────────────────────────────────────────
HIGH_VALUE_KEYWORDS = [
    "llm", "large language model", "gpt", "claude", "gemini", "llama",
    "agent", "multimodal", "transformer", "fine-tun", "rag",
    "retrieval", "reasoning", "benchmark", "open-source", "research",
    "breakthrough", "launch", "release", "model", "dataset",
    "alignment", "safety", "robotics", "automation", "arxiv",
    "deepmind", "openai", "anthropic", "mistral", "hugging",
    "diffusion", "embedding", "inference", "quantization", "lora",
    "chain-of-thought", "vision", "speech", "agentic", "rlhf",
]

# ─── 10 Rotating Fallback Topics ──────────────────────────────────────────────
FALLBACK_TOPICS = [
    {
        "title": "How Agentic AI Is Reshaping Enterprise Workflows in 2025",
        "summary": (
            "AI agents that can plan, use tools, and execute multi-step tasks "
            "are moving from research demos into production enterprise systems. "
            "Companies are deploying agents for code review, data analysis, "
            "and customer support automation."
        ),
        "source": "AI Engineering Community",
    },
    {
        "title": "The Rise of Small Language Models: Efficiency Over Scale",
        "summary": (
            "Researchers are finding that smaller, fine-tuned models often "
            "outperform giant general-purpose LLMs on specific tasks. "
            "Models under 10B parameters are becoming the go-to choice for "
            "edge deployment and cost-sensitive applications."
        ),
        "source": "Machine Learning Research",
    },
    {
        "title": "Retrieval-Augmented Generation Is Now the Industry Standard",
        "summary": (
            "RAG has become the default architecture for enterprise AI applications, "
            "allowing LLMs to access up-to-date domain-specific knowledge without "
            "expensive retraining. Vector databases and hybrid search are now "
            "central to modern AI stacks."
        ),
        "source": "AI Engineering Weekly",
    },
    {
        "title": "Open-Source LLMs Are Closing the Gap with Proprietary Models",
        "summary": (
            "Models like Llama 3, Mistral, and Qwen are benchmarking within "
            "striking distance of GPT-4 class models on reasoning tasks. "
            "The gap between open and closed models is narrowing faster "
            "than the industry predicted."
        ),
        "source": "Open Source AI Community",
    },
    {
        "title": "Multimodal AI: When Models Can See, Hear, and Reason Together",
        "summary": (
            "The latest generation of AI models natively processes text, images, "
            "audio, and video in a single unified architecture. This unlocks "
            "powerful applications in healthcare imaging, document processing, "
            "and real-time video understanding."
        ),
        "source": "AI Technology Review",
    },
    {
        "title": "AI Safety and Alignment: Why It Matters More Than Ever",
        "summary": (
            "As AI systems become more capable and agentic, ensuring they behave "
            "predictably and align with human values is a critical research priority. "
            "Techniques like RLHF, Constitutional AI, and interpretability are "
            "at the forefront of this work."
        ),
        "source": "AI Safety Research",
    },
    {
        "title": "The Prompt Engineering Era Is Ending — Fine-Tuning Takes Over",
        "summary": (
            "As foundation models mature, organizations are moving beyond "
            "prompt engineering toward fine-tuning on proprietary datasets. "
            "This shift creates a new competitive moat: unique training data, "
            "not just clever prompts."
        ),
        "source": "AI Engineering Insights",
    },
    {
        "title": "LLMs in Production: Lessons from Real-World Deployments",
        "summary": (
            "Engineering teams deploying LLMs at scale are learning hard lessons "
            "about latency, cost, hallucination rates, and monitoring. "
            "Observability tools and evaluation frameworks are now essential "
            "parts of the modern ML stack."
        ),
        "source": "MLOps Community",
    },
    {
        "title": "AI Coding Assistants Are Changing How Software Gets Built",
        "summary": (
            "Tools like GitHub Copilot, Cursor, and Claude Code are dramatically "
            "accelerating developer productivity. Studies show developers complete "
            "tasks 55% faster on average, fundamentally shifting the engineering role."
        ),
        "source": "Developer Technology News",
    },
    {
        "title": "Vector Databases: The Unsung Heroes of the Modern AI Stack",
        "summary": (
            "As RAG-based applications proliferate, vector databases like Pinecone, "
            "Weaviate, and pgvector are becoming critical infrastructure. Efficiently "
            "storing and retrieving semantic embeddings at scale is now a core "
            "engineering competency."
        ),
        "source": "Data Engineering Weekly",
    },
]


def _relevance_score(article: dict) -> float:
    """
    Score an article 0–1 for AI relevance:
      50% keyword density, 30% recency, 20% source authority.
    """
    text = (article.get("title", "") + " " + article.get("summary", "")).lower()

    hits = sum(1 for kw in HIGH_VALUE_KEYWORDS if kw in text)
    keyword_score = min(hits / 6, 1.0)

    now = datetime.now(timezone.utc)
    pub = article.get("published", now)
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)
    age_hours = (now - pub).total_seconds() / 3600
    recency_score = max(0.0, 1.0 - age_hours / 72)

    weight = article.get("weight", 1)
    source_score = min((weight - 1) / 3, 1.0)

    return round(0.5 * keyword_score + 0.3 * recency_score + 0.2 * source_score, 4)


def rank_articles(articles: list) -> list:
    """Score and sort articles by AI relevance (descending)."""
    for article in articles:
        article["relevance_score"] = _relevance_score(article)
    ranked = sorted(articles, key=lambda a: a["relevance_score"], reverse=True)

    logger.info("Top 3 articles after ranking:")
    for i, a in enumerate(ranked[:3]):
        logger.info(
            "  #%d [score %.3f] [%s] %s",
            i + 1,
            a["relevance_score"],
            a.get("source", "?"),
            a["title"][:80],
        )
    return ranked


def _weighted_random_pick(articles: list, pool_size: int = 10) -> list:
    """
    Instead of always picking the top-N articles in strict order,
    use weighted random sampling from the top pool_size articles.

    This ensures variety: even if ArXiv paper #1 is always top-ranked,
    the agent may pick #2, #3, or #4 instead on different runs.

    Weight = relevance_score^2 (emphasises high scores but allows lower ones).
    """
    pool = articles[:pool_size]
    if len(pool) <= 1:
        return pool

    weights = [max(a.get("relevance_score", 0.1) ** 2, 0.01) for a in pool]

    # Sample without replacement — pick min(pool_size, 15) diverse candidates
    sample_size = min(len(pool), 15)
    try:
        sampled = random.choices(pool, weights=weights, k=sample_size)
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for a in sampled:
            key = a["title"][:60]
            if key not in seen:
                seen.add(key)
                unique.append(a)
        logger.info(
            "Weighted random selection: picked %d diverse candidates from top %d.",
            len(unique), len(pool)
        )
        return unique
    except Exception:
        return pool


def select_topics_with_llm(articles: list, n: int = 3) -> list:
    """
    Use Groq LLM to pick the N most compelling and LinkedIn-worthy topics.
    Instructs the LLM to prefer novel and specific topics over generic AI content.
    """
    if not articles:
        return []

    catalogue = "\n".join(
        f"{i+1}. [{a['source']}] {a['title']} — {a['summary'][:120]}"
        for i, a in enumerate(articles[:20])
    )

    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0.5,   # slightly higher → more variety in topic selection
        max_tokens=600,
    )

    messages = [
        SystemMessage(content=(
            "You are an expert AI researcher and LinkedIn content strategist. "
            "Select the most compelling, timely, and professionally relevant AI topics "
            "for a LinkedIn audience of engineers, researchers, and business leaders. "
            "\n\nPriority rules:\n"
            "1. Prefer SPECIFIC research findings over generic AI discussions\n"
            "2. Prefer topics with concrete numbers, benchmarks, or named models\n"
            "3. AVOID selecting topics that sound generic like 'AI trends' or 'future of AI'\n"
            "4. Each selected topic must be DISTINCTLY different from the others"
        )),
        HumanMessage(content=(
            f"Select the {n} most compelling, varied topics from this list.\n"
            f"Choose topics that would generate strong LinkedIn engagement.\n\n"
            f"For each return EXACTLY:\n"
            f"TOPIC: <compelling LinkedIn-friendly title>\n"
            f"SUMMARY: <2-3 sentences with specific details>\n"
            f"SOURCE: <source name>\n\n"
            f"Articles:\n{catalogue}\n\n"
            f"IMPORTANT: Make sure the {n} topics are distinctly different "
            f"from each other — no two should cover the same concept.\n"
            f"Return ONLY the {n} topics in the format above."
        )),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
    except Exception as exc:
        logger.error("LLM topic selection failed: %s", exc)
        return [
            {"title": a["title"], "summary": a["summary"], "source": a["source"]}
            for a in articles[:n]
        ]

    selected = []
    for block in raw.strip().split("\n\n"):
        topic, summary, source = "", "", ""
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("TOPIC:"):
                topic = line.replace("TOPIC:", "").strip()
            elif line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
            elif line.startswith("SOURCE:"):
                source = line.replace("SOURCE:", "").strip()
        if topic:
            selected.append({"title": topic, "summary": summary, "source": source})

    if not selected:
        logger.warning("LLM returned unparseable response — using top ranked articles.")
        selected = [
            {"title": a["title"], "summary": a["summary"], "source": a["source"]}
            for a in articles[:n]
        ]

    logger.info("LLM selected %d topic(s).", len(selected))
    return selected


def _get_fallback_topic() -> dict:
    """
    Return a daily rotating fallback topic based on day-of-year.
    Also checks history so even fallbacks don't repeat.
    """
    day_index = datetime.now().timetuple().tm_yday % len(FALLBACK_TOPICS)

    # Try day_index first, then cycle through if that was recently used
    for offset in range(len(FALLBACK_TOPICS)):
        idx = (day_index + offset) % len(FALLBACK_TOPICS)
        topic = FALLBACK_TOPICS[idx]
        from tools.history_tool import was_used_recently
        if not was_used_recently(topic["title"]):
            logger.info(
                "Using fallback topic #%d: '%s'", idx + 1, topic["title"]
            )
            return topic

    # All fallbacks were used recently — just use today's index anyway
    topic = FALLBACK_TOPICS[day_index]
    logger.warning("All fallback topics recently used — reusing today's fallback.")
    return topic


def run_topic_agent() -> list:
    """
    Entry point for the Topic Agent.

    Flow:
      1. Log current post history summary
      2. Fetch from all 6 news sources
      3. Score articles by keyword density + recency + source authority
      4. Filter out recently-published topics (history_tool)
      5. Weighted random sampling from top pool (prevents same #1 every time)
      6. LLM picks the top N most compelling, varied topics
      7. If ALL sources fail → use daily rotating fallback topic

    Returns:
        List of selected topic dicts (title, summary, source).
    """
    # Show history so we know what was published recently
    logger.info("Post History:\n%s", get_history_summary())

    logger.info("Topic Agent: Fetching from multi-source news tool…")
    articles = fetch_all_news(max_per_source=settings.RSS_MAX_ARTICLES_PER_FEED)

    if not articles:
        logger.warning(
            "All news sources returned 0 articles. "
            "Using daily rotating fallback topic."
        )
        return [_get_fallback_topic()]

    logger.info("Ranking %d total articles by AI relevance…", len(articles))
    ranked = rank_articles(articles)

    # Filter out recently used topics
    fresh_articles = filter_fresh_topics(ranked)
    logger.info("%d fresh (unused) articles available.", len(fresh_articles))

    # Weighted random sampling for variety — prevents always picking #1
    diverse_candidates = _weighted_random_pick(fresh_articles, pool_size=12)

    return select_topics_with_llm(diverse_candidates, n=settings.TOPIC_CANDIDATE_COUNT)