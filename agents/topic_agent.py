"""
agents/topic_agent.py
---------------------
Topic Agent: Fetches trending AI topics from 6 sources, ranks by
relevance, filters history, and uses LLM to select LinkedIn-optimised
topics — preferring practitioner-relevant over pure academic content.

Key improvement: topics are translated from academic paper titles into
LinkedIn-friendly angles before passing to the content agent.
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
    "code", "programming", "developer", "api", "deployment",
]

# ─── 10 Rotating Fallback Topics (practitioner-focused) ───────────────────────
FALLBACK_TOPICS = [
    {
        "title": "Why RAG Is Replacing Fine-Tuning for Most Enterprise AI Teams",
        "summary": (
            "Enterprise AI teams are abandoning fine-tuning in favour of "
            "retrieval-augmented generation. The economics are compelling: "
            "RAG updates in hours, costs 90% less, and outperforms fine-tuned "
            "models on domain knowledge tasks. Vector databases like Pinecone "
            "and pgvector are now standard infrastructure."
        ),
        "source": "AI Engineering Community",
    },
    {
        "title": "The Real Reason Most LLM Production Deployments Fail",
        "summary": (
            "80% of LLM pilots never reach production. The culprit is rarely "
            "the model — it's evaluation frameworks, latency budgets, and "
            "hallucination rates that teams discover too late. Building "
            "proper evals before deployment is the new non-negotiable."
        ),
        "source": "MLOps Practitioners",
    },
    {
        "title": "Llama 3 vs GPT-4: What the Benchmarks Don't Tell You",
        "summary": (
            "Benchmark scores mislead practitioners. Llama 3 70B outperforms "
            "GPT-4 on coding tasks, matches it on reasoning, but struggles "
            "with multi-turn instruction following. The model choice depends "
            "on your specific task, not general leaderboard rankings."
        ),
        "source": "Open Source AI Community",
    },
    {
        "title": "AI Agents Are Breaking in Production — Here's Why",
        "summary": (
            "Agentic AI systems fail in production for predictable reasons: "
            "tool call loops, context window exhaustion, and inability to "
            "recover from partial failures. Teams building reliable agents "
            "need structured state management, not just bigger context windows."
        ),
        "source": "AI Engineering Weekly",
    },
    {
        "title": "The Hidden Cost of Running LLMs That Nobody Talks About",
        "summary": (
            "API costs are the visible cost. Hidden costs include: prompt "
            "engineering time (50+ hours per use case), evaluation pipelines, "
            "output validation, and the 3am incident when the model starts "
            "hallucinating in production. Total AI cost is 3-5x the API bill."
        ),
        "source": "AI Cost Analysis Report",
    },
    {
        "title": "Why AI Safety Research Finally Matters to Practitioners",
        "summary": (
            "AI safety is no longer just philosophy. RLHF, Constitutional AI, "
            "and interpretability tools are now standard parts of model "
            "training pipelines. Teams ignoring alignment are shipping models "
            "that behave unpredictably in edge cases — with real consequences."
        ),
        "source": "AI Safety Research",
    },
    {
        "title": "The Prompt Engineering Era Is Over — What Comes Next",
        "summary": (
            "Prompt engineering is table stakes now. The teams winning in 2025 "
            "are those with proprietary fine-tuned models, unique datasets, "
            "and systematic evaluation pipelines. The competitive moat has "
            "shifted from clever prompts to data and evaluation infrastructure."
        ),
        "source": "AI Strategy Insights",
    },
    {
        "title": "AI Code Assistants Changed My Team's Workflow — Not How You Think",
        "summary": (
            "GitHub Copilot and Cursor increased velocity by 40%, but changed "
            "the role of senior engineers. They now spend more time on "
            "architecture, code review, and AI output validation — less on "
            "implementation. Junior engineers are writing more code than ever."
        ),
        "source": "Developer Productivity Research",
    },
    {
        "title": "The Vector Database Landscape in 2025: What Actually Works",
        "summary": (
            "Pinecone, Weaviate, Qdrant, and pgvector each have real tradeoffs "
            "that benchmarks hide. Pinecone wins on managed simplicity, "
            "pgvector wins on existing PostgreSQL stacks, Qdrant wins on "
            "self-hosted performance. The right choice depends on your query patterns."
        ),
        "source": "Database Engineering Community",
    },
    {
        "title": "Multimodal AI Is Changing What 'Senior AI Engineer' Means",
        "summary": (
            "GPT-4V, Gemini 1.5, and Claude 3 process images, audio, and "
            "video natively. This unlocks document processing, quality control, "
            "and medical imaging use cases that weren't viable 18 months ago. "
            "AI engineers who only know text models are already behind."
        ),
        "source": "AI Technology Review",
    },
]


def _relevance_score(article: dict) -> float:
    """Score article for AI relevance: 50% keyword, 30% recency, 20% source weight."""
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
            i + 1, a["relevance_score"],
            a.get("source", "?"), a["title"][:80],
        )
    return ranked


def _weighted_random_pick(articles: list, pool_size: int = 12) -> list:
    """Weighted random sample from top pool — prevents always picking #1."""
    pool = articles[:pool_size]
    if len(pool) <= 1:
        return pool
    weights = [max(a.get("relevance_score", 0.1) ** 2, 0.01) for a in pool]
    sample_size = min(len(pool), 15)
    try:
        sampled = random.choices(pool, weights=weights, k=sample_size)
        seen, unique = set(), []
        for a in sampled:
            key = a["title"][:60]
            if key not in seen:
                seen.add(key)
                unique.append(a)
        logger.info("Weighted random: picked %d diverse candidates from top %d.", len(unique), len(pool))
        return unique
    except Exception:
        return pool


def select_topics_with_llm(articles: list, n: int = 3) -> list:
    """
    Use Groq LLM to select and REFRAME the N best topics for LinkedIn.

    Key improvement: LLM translates academic paper titles into
    practitioner-friendly angles that resonate with LinkedIn audience.
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
        temperature=0.5,
        max_tokens=700,
    )

    messages = [
        SystemMessage(content=(
            "You are a LinkedIn content strategist for a senior AI engineer "
            "with 50,000 followers. Your audience is AI practitioners, engineers, "
            "and technical leads — NOT executives or academics.\n\n"
            "When selecting topics:\n"
            "1. Translate academic titles into practitioner-relevant angles\n"
            "2. Prefer topics with real-world impact over theoretical research\n"
            "3. Avoid topics that are too niche or too academic\n"
            "4. Each topic must be distinctly different from the others\n"
            "5. Reframe the title as a LinkedIn-friendly hook, not a paper title"
        )),
        HumanMessage(content=(
            f"From these AI articles, select the {n} most compelling topics "
            f"for a LinkedIn post aimed at senior AI practitioners.\n\n"
            f"For each, return EXACTLY:\n"
            f"TOPIC: <LinkedIn-friendly title — not the paper title>\n"
            f"SUMMARY: <2-3 sentences explaining the practitioner impact>\n"
            f"SOURCE: <source name>\n\n"
            f"Articles:\n{catalogue}\n\n"
            f"CRITICAL: Rewrite academic titles as practitioner insights.\n"
            f"Example: 'EndoCoT: Scaling Endogenous Chain-of-Thought' → "
            f"'Why Diffusion Models Are Getting Better at Reasoning'\n\n"
            f"Return ONLY the {n} topics in the exact format above."
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
        logger.warning("LLM returned unparseable response — using raw articles.")
        selected = [
            {"title": a["title"], "summary": a["summary"], "source": a["source"]}
            for a in articles[:n]
        ]

    logger.info("LLM selected and reframed %d topic(s).", len(selected))
    return selected


def _get_fallback_topic() -> dict:
    """Rotating fallback: practitioner-focused, checks history before returning."""
    day_index = datetime.now().timetuple().tm_yday % len(FALLBACK_TOPICS)
    for offset in range(len(FALLBACK_TOPICS)):
        idx = (day_index + offset) % len(FALLBACK_TOPICS)
        topic = FALLBACK_TOPICS[idx]
        from tools.history_tool import was_used_recently, load_history
        if not was_used_recently(topic["title"], history=load_history()):
            logger.info("Using fallback topic #%d: '%s'", idx + 1, topic["title"])
            return topic
    return FALLBACK_TOPICS[day_index]


def run_topic_agent() -> list:
    """
    Entry point for the Topic Agent.
    Returns list of LinkedIn-reframed topic dicts.
    """
    logger.info("Post History:\n%s", get_history_summary())
    logger.info("Topic Agent: Fetching from multi-source news tool…")

    articles = fetch_all_news(max_per_source=settings.RSS_MAX_ARTICLES_PER_FEED)

    if not articles:
        logger.warning("All news sources returned 0 articles — using fallback.")
        return [_get_fallback_topic()]

    logger.info("Ranking %d total articles by AI relevance…", len(articles))
    ranked = rank_articles(articles)

    fresh = filter_fresh_topics(ranked)
    logger.info("%d fresh (unused) articles available.", len(fresh))

    diverse = _weighted_random_pick(fresh, pool_size=12)

    return select_topics_with_llm(diverse, n=settings.TOPIC_CANDIDATE_COUNT)