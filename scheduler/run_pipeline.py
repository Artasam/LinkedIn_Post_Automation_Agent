"""
scheduler/run_pipeline.py
--------------------------
Entry point called by GitHub Actions (and optionally cron).
Sets up logging, validates config, and runs the full agent pipeline.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

# Ensure the project root is on the path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import validate_config, LOG_LEVEL, LOG_TO_FILE, LOG_FILE_PATH
from workflow.agent_graph import run_pipeline


def setup_logging() -> None:
    """Configure logging to stdout (and optionally a file)."""
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]

    if LOG_TO_FILE:
        os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
        handlers.append(logging.FileHandler(LOG_FILE_PATH))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LinkedIn AI Content Automation Agent"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and log the post WITHOUT publishing to LinkedIn.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    logger = logging.getLogger("scheduler")

    # ── Banner ────────────────────────────────────────────────────────────────
    run_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    logger.info("=" * 60)
    logger.info("  LinkedIn AI Content Agent — %s", run_time)
    logger.info("  Mode: %s", "DRY RUN" if args.dry_run else "LIVE")
    logger.info("=" * 60)

    # ── Config Validation ─────────────────────────────────────────────────────
    missing = validate_config()
    if missing and not args.dry_run:
        logger.error(
            "Missing required environment variables: %s\n"
            "Set them in your .env file or GitHub Secrets.",
            ", ".join(missing),
        )
        sys.exit(1)
    elif missing:
        logger.warning(
            "Missing env vars (ignored in dry-run): %s", ", ".join(missing)
        )

    # ── Run Pipeline ──────────────────────────────────────────────────────────
    try:
        final_state = run_pipeline(dry_run=args.dry_run)
    except Exception as exc:
        logger.exception("Unhandled exception in pipeline: %s", exc)
        sys.exit(2)

    # ── Results ───────────────────────────────────────────────────────────────
    if final_state.get("error"):
        logger.error("Pipeline ended with error: %s", final_state["error"])
        sys.exit(3)

    logger.info("=" * 60)
    logger.info("GENERATED POST:")
    logger.info("-" * 60)
    logger.info("\n%s", final_state.get("full_post", "(empty)"))
    logger.info("-" * 60)

    result = final_state.get("publish_result", {})
    if result.get("dry_run"):
        logger.info("✅ DRY RUN complete. Post was NOT published.")
    elif result.get("success"):
        logger.info("✅ Post published! LinkedIn Post ID: %s", result.get("post_id"))
    else:
        logger.warning("⚠️  Unexpected result: %s", result)

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
