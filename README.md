# AI LinkedIn Content Automation Agent

> A production-ready multi-agent AI system that automatically generates and publishes
> unique daily LinkedIn posts about Artificial Intelligence — powered by Groq LLM,
> 6 live news sources, post-history deduplication, and GitHub Actions automation.
> **Zero manual effort after initial setup.**

![Python](https://img.shields.io/badge/Python-3.13-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.2+-green)
![LangGraph](https://img.shields.io/badge/LangGraph-1.1+-green)
![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b--versatile-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Table of Contents

- [Overview](#overview)
- [Complete Project Flow](#complete-project-flow)
- [Architecture — LangGraph Pipeline](#architecture--langgraph-pipeline)
- [Anti-Repetition System](#anti-repetition-system)
- [News Sources](#news-sources)
- [Image Generation](#image-generation)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Configuring the Groq API](#configuring-the-groq-api)
- [Configuring the LinkedIn API](#configuring-the-linkedin-api)
- [GitHub Actions Automation](#github-actions-automation)
- [Environment Variables Reference](#environment-variables-reference)
- [Running Locally](#running-locally)
- [Live Output Example](#live-output-example)
- [Troubleshooting](#troubleshooting)
- [Extending the Agent](#extending-the-agent)

---

## Overview

This project is a **production-ready multi-agent AI system** that runs fully on autopilot:

1. Fetches live AI news from **6 independent sources** (ArXiv, HackerNews, GitHub, NewsAPI, Wikipedia, DuckDuckGo)
2. Scores and ranks articles by keyword density, recency, and source authority
3. Filters out recently-published topics using **post history deduplication**
4. Uses **weighted random sampling** to guarantee topic variety
5. Uses **Llama 3.3 70B Versatile** via Groq to select and write a compelling post
6. Generates a professional, engagement-optimised post (max **300 words**)
7. Adds **4–6** contextual AI hashtags
8. Attaches a **professional image** via Pexels → Unsplash → SVG fallback
9. Publishes to LinkedIn via the official **API v2**
10. Runs automatically every day at **09:00 UTC** via GitHub Actions

---

## Complete Project Flow

```
╔══════════════════════════════════════════════════════════════════════════════╗
║              TRIGGER: GitHub Actions (09:00 UTC daily)                      ║
║              OR: python main.py  (manual local run)                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  scheduler/run_pipeline.py                                                  ║
║  • Validate 3 required env vars  • Setup logging  • Parse --dry-run flag    ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  workflow/agent_graph.py  ── LangGraph StateGraph DAG                       ║
║  AgentState: topics · best_draft · hashtags · full_post ·                   ║
║              publish_result · error · image_path                            ║
║  Every edge: error set? → END immediately. No error? → next node.          ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                      │
                        ┌─────────────▼─────────────┐
                        │      NODE 1: fetch_topics   │
                        │      agents/topic_agent.py  │
                        │                             │
                        │  1. Log post history        │
                        │  2. fetch_all_news()        │◄── 6 news sources
                        │  3. Rank by relevance       │
                        │  4. History filter          │◄── history_tool.py
                        │  5. Weighted random pick    │
                        │  6. LLM selects 5 topics    │
                        └─────────────┬───────────────┘
                                      │
             ╔════════════════════════╧═══════════════════════╗
             ║         NEWS SOURCES  (tools/news_tool.py)      ║
             ║                                                  ║
             ║  ArXiv(4) · HackerNews(3) · GitHub(3)           ║
             ║  NewsAPI(3) · Wikipedia(2) · DuckDuckGo(1-2)    ║
             ║                                                  ║
             ║  → Merge → Deduplicate → Sort by weight+recency ║
             ╚══════════════════════════════════════════════════╝
                                      │
                        ┌─────────────▼─────────────┐
                        │   NODE 2: generate_content  │
                        │   agents/content_agent.py   │
                        │                             │
                        │  Groq LLM call:             │
                        │  • Model: llama-3.3-70b     │
                        │  • Temp: 0.8 (boosted)      │
                        │  • Max tokens: 700          │
                        │  • Min words: 130           │
                        │  • Max words: 300           │
                        │  • Auto-retry if < 100 wds  │
                        └─────────────┬───────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │   NODE 3: generate_hashtags │
                        │   agents/hashtag_agent.py   │
                        │  • Temp: 0.4               │
                        │  • Output: 4-6 hashtags     │
                        └─────────────┬───────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │   NODE 4: generate_image    │
                        │   tools/image_tool.py       │
                        │                             │
                        │  Engine 1: Pexels API       │
                        │  Engine 2: Unsplash API     │
                        │  Engine 3: SVG Generator    │
                        │  (SVG always succeeds)      │
                        └─────────────┬───────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │   NODE 5: assemble_post     │
                        │   post_text + hashtags      │
                        └─────────────┬───────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │   NODE 6: publish_post      │
                        │   tools/linkedin_tool.py    │
                        │                             │
                        │  Text post: POST ugcPosts   │
                        │  Image post (3 steps):      │
                        │  1. Register upload         │
                        │  2. PUT binary image        │
                        │  3. POST ugcPosts + URN     │
                        │                             │
                        │  On success:                │
                        │  → record_post() called     │
                        │  → post_history.json saved  │
                        └─────────────┬───────────────┘
                                      │
                                    [END]
                             Log result + exit code
                     0=success · 1=missing env · 2=exception
```

---

## Architecture — LangGraph Pipeline

```python
class AgentState(TypedDict):
    topics:         list[dict]    # selected topic candidates
    best_draft:     dict          # {post_text, topic, score}
    hashtags:       str           # final hashtag string
    full_post:      str           # assembled post + hashtags
    publish_result: dict          # LinkedIn API response
    error:          Optional[str] # set by any node on failure
    image_path:     Optional[str] # path to image file
```

Every edge checks `state["error"]` — failure in any node routes to `END` immediately, preventing cascading failures.

---

## Anti-Repetition System

### 1. Post History Filter (`tools/history_tool.py`)

```
Every published post → saved to post_history.json (last 30 entries)
                                    │
Next run: load history ONCE (single file read, not per-article)
          ↓
For each candidate: word-overlap similarity vs every past title
  ≥ 60% similar → SKIP   |   < 60% → KEEP
```

### 2. Weighted Random Selection (`agents/topic_agent.py`)

Samples from top 12 fresh articles using `weight = score²` probability — high-scored articles are preferred but lower-ranked ones get a chance, guaranteeing variety even when ArXiv publishes the same top paper multiple days.

### 3. LLM Diversity Instruction

The topic selection prompt explicitly requires:
> *"Make sure the 5 topics are distinctly different from each other."*

### 4. Daily Rotating Fallbacks

10 pre-written high-quality fallback topics, selected by `day_of_year % 10`. Each fallback also checks history — cycles to the next one if recently used.

---

## News Sources

All 6 sources use direct **REST APIs** (not RSS). Works even when RSS feeds are DNS-blocked:

| # | Source | Key | Weight | Filter |
|---|--------|-----|--------|--------|
| 1 | **ArXiv** | ❌ None | 4 | CS.AI + CS.LG + CS.CL, last 7 days |
| 2 | **HackerNews** | ❌ None | 3 | Score > 20, last 7 days |
| 3 | **GitHub Trending** | ❌ Optional | 3 | Stars > 50, created last 7 days |
| 4 | **NewsAPI** | ✅ Free | 3 | 100 req/day at newsapi.org |
| 5 | **Wikipedia** | ❌ None | 2 | Recent AI article updates |
| 6 | **DuckDuckGo** | ❌ None | 1–2 | Instant answers, no rate limit |

**Relevance scoring:**
```
score = 0.5 × keyword_score  (38 AI keywords)
      + 0.3 × recency_score  (decay over 72 hours)
      + 0.2 × source_score   (weight 1–4 normalised)
```

---

## Image Generation

All engines **live-tested and verified working** (2026-03-17).

### Removed Engines (failed live testing)

| Engine | Status | Reason |
|--------|--------|--------|
| Pollinations AI | ❌ Removed | HTTP 500 — server unstable |
| HuggingFace FLUX.1-dev | ❌ Removed | HTTP 410 — moved to paid tier |
| HuggingFace FLUX.1-schnell | ❌ Removed | HTTP 410 — moved to paid tier |
| Together AI | ❌ Removed | HTTP 402 — credits required |
| Stability AI | ❌ Removed | HTTP 402 — paid subscription |

### Current Engines (working)

| # | Engine | Cost | Key | Quality |
|---|--------|------|-----|---------|
| 1 | **Pexels API** | Free 200 req/hr | `PEXELS_API_KEY` | Professional photography |
| 2 | **Unsplash API** | Free 50 req/hr | `UNSPLASH_ACCESS_KEY` | Curated professional photos |
| 3 | **SVG Generator** | Always free | None needed | Clean branded header |

**Waterfall:** Pexels → Unsplash → SVG. The SVG engine is pure Python (no network, no key) so **images are always generated** — the post is never text-only due to image failure.

### Getting Free API Keys

**Pexels** (recommended first):
1. Go to [https://www.pexels.com/api/](https://www.pexels.com/api/)
2. Click **Get Started** → free signup
3. Copy your API key → add `PEXELS_API_KEY=your_key` to `.env`

**Unsplash:**
1. Go to [https://unsplash.com/developers](https://unsplash.com/developers)
2. Click **Register as a developer** → create app
3. Copy your Access Key → add `UNSPLASH_ACCESS_KEY=your_key` to `.env`

---

## Project Structure

```
linkedin-ai-agent/
│
├── agents/
│   ├── __init__.py
│   ├── topic_agent.py       # News fetch → rank → history filter → LLM selection
│   ├── content_agent.py     # Groq LLM post (temp 0.8, max 700 tokens, min 130 words)
│   └── hashtag_agent.py     # 4–6 hashtags (temp 0.4, max 256 tokens)
│
├── tools/
│   ├── __init__.py
│   ├── news_tool.py         # 6-source AI news fetcher
│   ├── history_tool.py      # Post deduplication — post_history.json, 60% threshold
│   ├── linkedin_tool.py     # LinkedIn API v2 — ugcPosts, version 202304
│   └── image_tool.py        # Pexels → Unsplash → SVG (all live-tested)
│
├── workflow/
│   ├── __init__.py
│   └── agent_graph.py       # LangGraph DAG — 6 nodes + conditional error routing
│
├── scheduler/
│   ├── __init__.py
│   └── run_pipeline.py      # CLI — logging, validation, --dry-run
│
├── config/
│   ├── __init__.py
│   └── settings.py          # All env vars + constants
│
├── .github/
│   └── workflows/
│       └── daily_post.yml   # 8 steps, cron 0 9 * * *, persists post history
│
├── post_history.json        # Auto-created — last 30 published topics
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.13 |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| AI Framework | LangChain (`langchain-core`, `langchain-groq`) |
| Workflow | LangGraph `StateGraph` |
| News Sources | ArXiv XML · HackerNews Algolia · GitHub REST · NewsAPI · Wikipedia · DuckDuckGo |
| Post History | `history_tool.py` — JSON file, 60% word-overlap similarity |
| Images | Pexels REST · Unsplash REST · SVG Generator (Python) |
| LinkedIn API | REST v2 — `ugcPosts`, version `202304` |
| Automation | GitHub Actions — cron `0 9 * * *` |

---

## Quick Start

```powershell
# 1. Clone
git clone https://github.com/YOUR_USERNAME/linkedin-ai-agent.git
cd linkedin-ai-agent

# 2. Virtual environment (Windows)
python -m venv LA
LA\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Configure
copy .env.example .env
# Edit .env — fill in GROQ_API_KEY, LINKEDIN_ACCESS_TOKEN, LINKEDIN_PERSON_ID

# 5. Test (no publishing)
python main.py --dry-run

# 6. Run live
python main.py
```

---

## Configuring the Groq API

1. Sign up at [https://console.groq.com](https://console.groq.com)
2. **API Keys** → **Create API Key**
3. Add to `.env`: `GROQ_API_KEY=gsk_xxxx`

### LLM Calls Per Pipeline Run

| Agent | Temperature | Max Tokens | Purpose |
|-------|-------------|------------|---------|
| Topic Agent | 0.5 | 600 | Select 5 varied topics |
| Content Agent | 0.8 | 700 | Write LinkedIn post |
| Hashtag Agent | 0.4 | 256 | Generate 4–6 hashtags |

**Total TPM per run:** ~2,000–2,500 — well within the 6,000 TPM limit.

### Available Models

| Model | TPM | Best for |
|-------|-----|----------|
| `llama-3.3-70b-versatile` ✅ default | 6,000 | Best prose quality |
| `llama-3.1-8b-instant` | 6,000 | Fastest |
| `meta-llama/llama-4-scout-17b-16e-instruct` | 30,000 | Multimodal |
| `qwen/qwen3-32b` | 6,000 | Math, coding |

---

## Configuring the LinkedIn API

### Step 1 — Create App
[https://www.linkedin.com/developers/apps/new](https://www.linkedin.com/developers/apps/new) → fill in details → **Create App**

### Step 2 — Request Products
App → **Products** tab → request **"Share on LinkedIn"** and **"Sign In with LinkedIn using OpenID Connect"**

### Step 3 — Generate Token
[https://www.linkedin.com/developers/tools/oauth/token-generator](https://www.linkedin.com/developers/tools/oauth/token-generator) → select app → check `w_member_social` → **Request access token**

```env
LINKEDIN_ACCESS_TOKEN=AQVt_ZbBxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> ⚠️ Token expires after **60 days** — regenerate and update GitHub Secret monthly.

### Step 4 — Get Person ID

```powershell
$token = "YOUR_ACCESS_TOKEN"
Invoke-RestMethod -Uri "https://api.linkedin.com/v2/userinfo" `
  -Headers @{ Authorization = "Bearer $token" }
```

Copy the `sub` field → `LINKEDIN_PERSON_ID=abc123XYZ`

> Use `/v2/userinfo` not `/v2/me` — LinkedIn removed `r_liteprofile` scope in 2024.

---

## GitHub Actions Automation

### Required Secrets

**Settings → Secrets and variables → Actions → New repository secret:**

| Secret | Required | Description |
|--------|----------|-------------|
| `GROQ_API_KEY` | ✅ | Groq API key |
| `LINKEDIN_ACCESS_TOKEN` | ✅ | LinkedIn OAuth token (expires 60 days) |
| `LINKEDIN_PERSON_ID` | ✅ | LinkedIn profile `sub` field |
| `PEXELS_API_KEY` | Recommended | Free photos — pexels.com/api |
| `UNSPLASH_ACCESS_KEY` | Recommended | Free photos — unsplash.com/developers |
| `NEWS_API_KEY` | Optional | 100 req/day — newsapi.org |
| `GITHUB_TOKEN` | Optional | Raises GitHub API to 5,000 req/hr |

### Workflow — 8 Steps

| # | Step | Purpose |
|---|------|---------|
| 1 | Checkout | Pull latest code |
| 2 | Set up Python 3.13 | Install Python with pip cache |
| 3 | Install dependencies | `pip install -r requirements.txt` |
| 4 | Verify required secrets | Fail fast before API calls |
| 5 | Smoke-test imports | Validate packages load |
| 6 | Run LinkedIn AI Agent | Full pipeline → publish post |
| 7 | Persist post history | Commit `post_history.json` → dedup works across runs |
| 8 | Write job summary | Log result to Actions dashboard |

### Key: Step 7 — Persisting Post History

GitHub Actions runners are stateless — `post_history.json` would reset each run without Step 7. The workflow commits the file back to the repo so the anti-repetition system works across all daily runs:

```yaml
- name: "Persist post history"
  if: success()
  run: |
    git config user.name "github-actions[bot]"
    git add post_history.json
    git diff --staged --quiet || git commit -m "chore: update post history [skip ci]"
    git push || true
```

### Schedule

```yaml
schedule:
  - cron: "0 9 * * *"    # Every day 09:00 UTC = 14:00 PKT
```

---

## Environment Variables Reference

```env
# ── REQUIRED ─────────────────────────────────────────────────────────────────
GROQ_API_KEY=
LINKEDIN_ACCESS_TOKEN=
LINKEDIN_PERSON_ID=

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.7
GROQ_MAX_TOKENS=700          # Content agent needs 700 for full 130-300 word posts

# ── NEWS SOURCES ─────────────────────────────────────────────────────────────
RSS_MAX_ARTICLES_PER_FEED=8  # Max articles per source (8 sources × 8 = 64 max)
NEWS_API_KEY=                # Optional — newsapi.org
GITHUB_TOKEN=                # Optional — higher GitHub rate limit

# ── AGENT BEHAVIOUR ───────────────────────────────────────────────────────────
TOPIC_CANDIDATE_COUNT=5      # Topics LLM evaluates
MULTI_TOPIC_DRAFTS=1         # Set to 3 for multi-draft best-of selection

# ── POST CONSTRAINTS (hardcoded in settings.py) ───────────────────────────────
# POST_MAX_WORDS=300          # Max words per post
# POST_MIN_HASHTAGS=4
# POST_MAX_HASHTAGS=6

# ── IMAGE GENERATION ─────────────────────────────────────────────────────────
ENABLE_IMAGE_GENERATION=true    # Recommended: true (SVG always works as fallback)
PEXELS_API_KEY=                 # Free — pexels.com/api (Engine 1)
UNSPLASH_ACCESS_KEY=            # Free — unsplash.com/developers (Engine 2)
# Engine 3 (SVG) needs no key — always works

# ── LOGGING ───────────────────────────────────────────────────────────────────
LOG_LEVEL=INFO
LOG_TO_FILE=false
LOG_FILE_PATH=logs/agent.log
```

---

## Running Locally

```powershell
# Dry run — no publishing
python main.py --dry-run

# Live mode
python main.py
```

### Real Console Output (Healthy Run)

```
[INFO] Loaded 8 entries from post history.
[INFO] Topic Agent: Fetching from multi-source news tool…
[INFO] ArXiv: fetched 8 papers
[INFO] GitHub Trending: fetched 3 repos
[INFO] NewsAPI: fetched 10 articles
[INFO] Wikipedia: fetched 5 articles
[INFO] DuckDuckGo: fetched 4 items
[INFO] News fetch complete: 30 total articles
[INFO] Ranking 30 total articles by AI relevance…
[INFO] History filter: removed 1 recently-used topic(s), 29 fresh remain.
[INFO] Weighted random selection: picked 8 diverse candidates from top 12.
[INFO] LLM selected 5 topic(s).
[INFO] Generated post: 134 words, attempt 2 (score: 89.0)
[INFO] Final hashtags (5): #AI #MachineLearning ...
[INFO] Trying Engine 1: Pexels API…
[INFO] ✅ Engine 1 (Pexels) succeeded.
[INFO] Image post published. ID: urn:li:share:xxxx
[INFO] ✅ Post published! LinkedIn Post ID: urn:li:share:xxxx
```

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success — post published |
| `1` | Missing environment variables |
| `2` | Unhandled exception |
| `3` | Pipeline error state |

---

## Live Output Example

Real post generated on 2026-03-17:

```
90% of neural networks are vulnerable to privacy attacks.

Researchers have discovered that learnability and privacy are fundamentally
entangled. This challenges prior approaches to privacy preservation and
demands a completely new paradigm for secure neural network design.

The implications are significant: retraining all weights is no longer a
viable solution. Teams building production AI systems need to account for
this entanglement from the architecture design phase, not as an afterthought.

Start auditing your neural network training pipelines today for
privacy-leakability exposure.

What's your team's current approach to balancing model performance
with privacy guarantees?

#AI #MachineLearning #NeuralNetworkSecurity #DeepLearningPrivacy
#ArtificialIntelligence
```

> **Post ID:** `urn:li:share:7439402776255848448`

---

## Troubleshooting

### Post keeps repeating topics

Reset post history:
```powershell
del post_history.json
```

### Post is too short (under 100 words)

The agent auto-retries once with a stricter prompt. If still short, increase token budget:
```env
GROQ_MAX_TOKENS=800
```

### No image attached (text-only post)

Set `ENABLE_IMAGE_GENERATION=true` and add at least one key:
```env
ENABLE_IMAGE_GENERATION=true
PEXELS_API_KEY=your_free_key   # pexels.com/api — free signup
```
If no keys are set, the SVG engine generates a branded header automatically.

### HackerNews always returns 0 stories

Known issue — HN Algolia sometimes throttles anonymous queries. The other 5 sources compensate. ArXiv and Wikipedia are always reliable.

### LinkedIn token expired (60 days)

Regenerate at [linkedin.com/developers/tools/oauth/token-generator](https://www.linkedin.com/developers/tools/oauth/token-generator) → update `LINKEDIN_ACCESS_TOKEN` in GitHub Secrets.

### LinkedIn 403 on `/v2/me`

Use `/v2/userinfo` instead — `r_liteprofile` scope was removed in 2024.

---

## Extending the Agent

### Add a New News Source

In `tools/news_tool.py`:
```python
def fetch_my_source(max_results: int = 8) -> list[dict]:
    articles = []
    # ... fetch logic ...
    articles.append(_make_article(
        title="...", summary="...", url="...",
        source="My Source", weight=3
    ))
    return articles
```
Then add to the `fetchers` list in `fetch_all_news()`.

### Add a New Agent Node

```python
# 1. In workflow/agent_graph.py:
def your_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return state
    # your logic
    return state

graph.add_node("your_node", your_node)
graph.add_conditional_edges(
    "previous_node", should_continue,
    {"continue": "your_node", "error": END}
)
```

### Change Posting Schedule

```yaml
# .github/workflows/daily_post.yml
schedule:
  - cron: "0 4 * * 1-5"   # 9:00 AM PKT weekdays (highest engagement)
  - cron: "0 9 * * *"      # 2:00 PM PKT daily (current)
```

---

## License

MIT — free to use, modify, and distribute.

---

*Built with LangChain · LangGraph · Groq API · LinkedIn API v2 · GitHub Actions*
*Tested: Python 3.13 · Windows 11 · Ubuntu (GitHub Actions)*