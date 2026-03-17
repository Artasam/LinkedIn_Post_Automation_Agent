"""
tools/
------
External integration tools for the LinkedIn AI Content Automation Agent.

Modules
-------
news_tool       — Smart multi-source AI news fetcher. Replaces RSS entirely.
                  Uses 6 independent REST APIs that work even when RSS feeds
                  are blocked by DNS or network restrictions:

                    1. ArXiv API          (weight 4) — Official XML API, no key needed.
                                                       Fetches latest CS.AI/CS.LG/CS.CL
                                                       research papers from last 7 days.

                    2. HackerNews Algolia (weight 3) — Top AI stories filtered by
                                                       score > 50 and comments > 10.
                                                       No key required.

                    3. GitHub Trending    (weight 3) — New AI/ML repositories with
                                                       50+ stars created in last 7 days.
                                                       No key required (optional token
                                                       raises limit to 5,000 req/hr).

                    4. NewsAPI            (weight 3) — Real-time AI news articles.
                                                       Optional key (free tier: 100 req/day).
                                                       Skipped silently if key not set.

                    5. Wikipedia          (weight 2) — MediaWiki REST API, always
                                                       accessible. Returns recently
                                                       updated AI-related articles.

                    6. DuckDuckGo         (weight 1) — Instant Answer API. No key,
                                                       no rate limits. Returns topic
                                                       summaries and related concepts.

                  Entry point: fetch_all_news(max_per_source=8)
                  Returns: deduplicated, recency+weight sorted article list.

history_tool    — Post history tracker. Prevents the agent from ever repeating
                  a topic it already published.

                  How it works:
                    - Saves every published topic to post_history.json (last 30)
                    - Before topic selection, filters out articles that are
                      60%+ similar (word-overlap) to any past post title
                    - Rotates through fallback topics using day-of-year index
                      so even offline fallbacks never repeat

                  Key functions:
                    record_post(topic, post_id)      — call after successful publish
                    filter_fresh_topics(topics)      — remove recently-used topics
                    was_used_recently(title)         — check single title
                    get_history_summary()            — human-readable log string

safety_tool     — Content safety filter. Two-tier moderation system:

                    Tier 1 (Rule-based, instant):
                      • Banned phrase detection (50+ patterns)
                      • Banned regex patterns (spam, false claims)
                      • Word count validation (120–150 words)
                      • Emoji overload detection (> 6 emojis)
                      • ALL-CAPS abuse detection (> 5 caps words)
                      • Closing question check

                    Tier 2 (LLM moderation, only if Tier 1 passes):
                      • Self-moderation via Groq LLM (temp=0.0)
                      • Checks: factual accuracy, professionalism,
                        LinkedIn policy, educational value, tone

                    On failure: auto-regenerate (up to 3 retries).
                    Final fallback: 3 guaranteed-safe pre-written posts.

                  Key functions:
                    run_safety_check(post_text, topic_title) → SafetyResult
                    get_safe_fallback(index)                 → str

linkedin_tool   — LinkedIn API v2 client (API version: 202304).
                  Handles OAuth bearer token validation, plain-text post
                  publishing, and 3-step image-attached post publishing:

                    Step 1: POST /v2/assets?action=registerUpload  → upload URL + asset URN
                    Step 2: PUT  <upload_url>                       → upload PNG binary
                    Step 3: POST /v2/ugcPosts                       → publish with asset URN

                  Falls back to text-only if image upload fails at any step.

image_tool      — AI image generation using Stability AI REST API v2beta.
                  Endpoint: https://api.stability.ai/v2beta/stable-image/generate/core
                  Output: 16:9 PNG image (aspect_ratio="16:9", output_format="png").
                  Only active when ENABLE_IMAGE_GENERATION=true in environment.
                  Generates topic-aware prompts from 10 keyword-to-style mappings.
                  Falls back gracefully — image failure never breaks the pipeline.

Usage
-----
    from tools.news_tool     import fetch_all_news
    from tools.history_tool  import record_post, filter_fresh_topics, get_history_summary
    from tools.safety_tool   import run_safety_check, get_safe_fallback
    from tools.linkedin_tool import publish_text_post, publish_image_post, validate_token
    from tools.image_tool    import generate_image        # optional — needs STABILITY_API_KEY

All tools are called internally by the agents and workflow graph.
Direct imports are always preferred over importing from this __init__.py
to avoid circular import chains at module load time.
"""

# No eager imports — prevents circular import chain:
# main → scheduler.__init__ → workflow.__init__ → agents.__init__
#      → topic_agent → news_tool / history_tool → (safe, no back-references)