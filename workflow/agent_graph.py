"""
workflow/agent_graph.py
-----------------------
LangGraph Workflow Orchestrator.

Pipeline (7 nodes):

  [fetch_topics] → [generate_content] → [safety_check]
                                              ↓
                                     [generate_hashtags]
                                              ↓
                                      [generate_image]
                                              ↓
                                       [assemble_post]
                                              ↓
                                       [publish_post]

State flows via AgentState TypedDict.
Every edge is conditional — errors route immediately to END.
Safety check runs after content generation and before hashtag/publish.
"""

import logging
from typing import TypedDict, Optional, Any

from langgraph.graph import StateGraph, END

from agents.topic_agent import run_topic_agent
from agents.content_agent import generate_best_post
from agents.hashtag_agent import run_hashtag_agent
from tools.linkedin_tool import publish_text_post, publish_image_post, LinkedInAPIError
from tools.safety_tool import run_safety_check, get_safe_fallback
from tools.history_tool import record_post
from config import settings

logger = logging.getLogger(__name__)


# ─── Shared State Schema ───────────────────────────────────────────────────────

class AgentState(TypedDict):
    """State passed between all nodes in the workflow graph."""
    topics:          list[dict]    # Selected topic candidates
    best_draft:      dict          # {post_text, topic, score, safety_passed}
    hashtags:        str           # Final hashtag string
    full_post:       str           # post_text + newlines + hashtags
    publish_result:  dict          # LinkedIn API response
    error:           Optional[str] # Error message if any node fails
    image_path:      Optional[str] # Path to generated image (bonus)
    safety_retries:  int           # Number of safety regenerations done


# ─── Node 1: Fetch Topics ─────────────────────────────────────────────────────

def fetch_topics_node(state: AgentState) -> AgentState:
    """Fetch and rank trending AI topics from 6 news sources."""
    logger.info("▶ Node: fetch_topics")
    try:
        topics = run_topic_agent()
        state["topics"] = topics
        logger.info("Topics fetched: %d", len(topics))
    except Exception as exc:
        state["error"] = f"fetch_topics failed: {exc}"
        logger.error(state["error"])
    return state


# ─── Node 2: Generate Content ─────────────────────────────────────────────────

def generate_content_node(state: AgentState) -> AgentState:
    """Generate LinkedIn post content using the Content Agent + safety filter."""
    logger.info("▶ Node: generate_content")
    if state.get("error"):
        return state
    try:
        best_draft = generate_best_post(state["topics"])
        state["best_draft"] = best_draft
    except Exception as exc:
        state["error"] = f"generate_content failed: {exc}"
        logger.error(state["error"])
    return state


# ─── Node 3: Safety Check ─────────────────────────────────────────────────────

def safety_check_node(state: AgentState) -> AgentState:
    """
    Final content safety gate before hashtags and publishing.

    Runs both rule-based and LLM moderation checks.
    If post fails, replaces it with a guaranteed-safe fallback post.
    This node never blocks the pipeline — it either passes or substitutes.
    """
    logger.info("▶ Node: safety_check")
    if state.get("error"):
        return state

    post_text   = state["best_draft"].get("post_text", "")
    topic_title = state["best_draft"].get("topic", {}).get("title", "")
    retries     = state.get("safety_retries", 0)

    safety = run_safety_check(post_text, topic_title)

    if safety.passed:
        logger.info("Safety check PASSED for post on '%s'", topic_title[:60])
        state["best_draft"]["safety_passed"] = True
    else:
        logger.warning(
            "Safety check FAILED (retries so far: %d). "
            "Substituting with guaranteed-safe fallback post.",
            retries,
        )
        fallback_text = get_safe_fallback(index=retries)
        state["best_draft"]["post_text"]    = fallback_text
        state["best_draft"]["safety_passed"] = False
        state["safety_retries"]             = retries + 1
        logger.info("Safe fallback post substituted.")

    return state


# ─── Node 4: Generate Hashtags ────────────────────────────────────────────────

def generate_hashtags_node(state: AgentState) -> AgentState:
    """Generate 3–5 optimised LinkedIn hashtags for the post."""
    logger.info("▶ Node: generate_hashtags")
    if state.get("error"):
        return state
    try:
        post_text = state["best_draft"]["post_text"]
        topic     = state["best_draft"]["topic"]
        hashtags  = run_hashtag_agent(post_text, topic)
        state["hashtags"] = hashtags
    except Exception as exc:
        state["error"] = f"generate_hashtags failed: {exc}"
        logger.error(state["error"])
    return state


# ─── Node 5: Generate Image ───────────────────────────────────────────────────

def generate_image_node(state: AgentState) -> AgentState:
    """
    Generate an AI image using Stability AI REST API (optional).
    Only runs when ENABLE_IMAGE_GENERATION=true.
    Failure is non-fatal — pipeline continues with text-only post.
    """
    logger.info("▶ Node: generate_image")
    if state.get("error") or not settings.ENABLE_IMAGE_GENERATION:
        return state

    try:
        from tools.image_tool import generate_image
        topic_title = state["best_draft"]["topic"].get("title", "AI trends")
        image_path  = generate_image(topic_title)
        state["image_path"] = image_path
        logger.info("Image generated at: %s", image_path)
    except ImportError:
        logger.info("image_tool not available — skipping image generation.")
    except Exception as exc:
        logger.warning("Image generation failed (non-fatal): %s", exc)
    return state


# ─── Node 6: Assemble Post ────────────────────────────────────────────────────

def assemble_post_node(state: AgentState) -> AgentState:
    """Combine post text and hashtags into the final publishable post."""
    logger.info("▶ Node: assemble_post")
    if state.get("error"):
        return state

    post_text = state["best_draft"]["post_text"]
    hashtags  = state.get("hashtags", "")
    full_post = f"{post_text}\n\n{hashtags}".strip()
    state["full_post"] = full_post

    safety_flag = "✓ safety-checked" if state["best_draft"].get("safety_passed") else "⚠ fallback used"
    logger.info(
        "Assembled post (%d chars) [%s]:\n%s",
        len(full_post), safety_flag, full_post,
    )
    return state


# ─── Node 7: Publish Post ─────────────────────────────────────────────────────

def publish_post_node(state: AgentState) -> AgentState:
    """Publish the final post to LinkedIn and record to post history."""
    logger.info("▶ Node: publish_post")
    if state.get("error"):
        return state

    try:
        full_post  = state["full_post"]
        image_path = state.get("image_path")

        if image_path and settings.ENABLE_IMAGE_GENERATION:
            result = publish_image_post(full_post, image_path)
        else:
            result = publish_text_post(full_post)

        state["publish_result"] = result
        logger.info("Post published successfully: %s", result)

        # Record to history — prevents future topic repetition
        if result.get("success"):
            topic   = state.get("best_draft", {}).get("topic", {})
            post_id = result.get("post_id", "")
            try:
                record_post(topic, post_id)
            except Exception as hist_exc:
                logger.warning("Could not record post history: %s", hist_exc)

    except LinkedInAPIError as exc:
        state["error"] = f"publish_post failed: {exc}"
        logger.error(state["error"])
    except Exception as exc:
        state["error"] = f"publish_post unexpected error: {exc}"
        logger.error(state["error"])

    return state


# ─── Conditional Edge ─────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """Route to END on error, continue otherwise."""
    return "error" if state.get("error") else "continue"


# ─── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph() -> Any:
    """Build and compile the 7-node LangGraph workflow."""
    graph = StateGraph(AgentState)

    graph.add_node("fetch_topics",      fetch_topics_node)
    graph.add_node("generate_content",  generate_content_node)
    graph.add_node("safety_check",      safety_check_node)
    graph.add_node("generate_hashtags", generate_hashtags_node)
    graph.add_node("generate_image",    generate_image_node)
    graph.add_node("assemble_post",     assemble_post_node)
    graph.add_node("publish_post",      publish_post_node)

    graph.set_entry_point("fetch_topics")

    for src, dst in [
        ("fetch_topics",      "generate_content"),
        ("generate_content",  "safety_check"),
        ("safety_check",      "generate_hashtags"),
        ("generate_hashtags", "generate_image"),
        ("generate_image",    "assemble_post"),
        ("assemble_post",     "publish_post"),
    ]:
        graph.add_conditional_edges(src, should_continue, {"continue": dst, "error": END})

    graph.add_edge("publish_post", END)
    return graph.compile()


# ─── Pipeline Entry Point ─────────────────────────────────────────────────────

def run_pipeline(dry_run: bool = False) -> AgentState:
    """
    Execute the full multi-agent pipeline.

    Args:
        dry_run: If True, assembles and safety-checks the post but skips publishing.

    Returns:
        Final AgentState after all nodes have run.
    """
    initial_state: AgentState = {
        "topics":         [],
        "best_draft":     {},
        "hashtags":       "",
        "full_post":      "",
        "publish_result": {},
        "error":          None,
        "image_path":     None,
        "safety_retries": 0,
    }

    if dry_run:
        logger.info("DRY RUN MODE: Post will be generated and safety-checked but NOT published.")

        def dry_publish(state: AgentState) -> AgentState:
            logger.info("[DRY RUN] Skipping LinkedIn publish.")
            state["publish_result"] = {"success": True, "dry_run": True}
            return state

        graph = StateGraph(AgentState)
        graph.add_node("fetch_topics",      fetch_topics_node)
        graph.add_node("generate_content",  generate_content_node)
        graph.add_node("safety_check",      safety_check_node)
        graph.add_node("generate_hashtags", generate_hashtags_node)
        graph.add_node("generate_image",    generate_image_node)
        graph.add_node("assemble_post",     assemble_post_node)
        graph.add_node("publish_post",      dry_publish)
        graph.set_entry_point("fetch_topics")

        for src, dst in [
            ("fetch_topics",      "generate_content"),
            ("generate_content",  "safety_check"),
            ("safety_check",      "generate_hashtags"),
            ("generate_hashtags", "generate_image"),
            ("generate_image",    "assemble_post"),
            ("assemble_post",     "publish_post"),
        ]:
            graph.add_conditional_edges(src, should_continue, {"continue": dst, "error": END})

        graph.add_edge("publish_post", END)
        compiled = graph.compile()
    else:
        compiled = build_graph()

    return compiled.invoke(initial_state)