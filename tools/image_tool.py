"""
tools/image_tool.py
-------------------
Professional Image Fetcher & Generator for LinkedIn Posts.

UPDATED ENGINES (2026-07-27):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ HuggingFace FLUX       — HTTP 410 (models moved to paid tier)
❌ Together AI            — HTTP 402 (credits required)
❌ Stability AI           — HTTP 402 (paid subscription required)
❌ imagen-3.0-generate-002 — HTTP 404 (restricted to paid Vertex AI only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ ENGINE 0 — Gemini 2.5 Flash Image : FREE via AI Studio, AI infographics
✅ ENGINE 1 — Pollinations AI        : FREE, enhanced AI generated pictures
✅ ENGINE 2 — Pexels API             : FREE, 200 req/hr, professional photos
✅ ENGINE 3 — Unsplash API           : FREE, 50 req/hr, curated photos
✅ ENGINE 4 — SVG Generator          : ALWAYS works, no network, branded visuals
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ENGINE DESCRIPTIONS:
  Gemini   — Google's latest multimodal model generates bespoke AI infographics
             that align with the post topic. Uses generate_content() API.
  Pexels   — Real professional photography, free API, 200 req/hr limit,
             searches by keyword to find relevant tech/AI imagery.
  Unsplash — Curated high-quality photos, free API, 50 req/hr limit,
             perfect for technology and abstract professional images.
  SVG      — Pure Python, zero dependencies, zero network, always works.
             Generates a clean branded LinkedIn header with topic title,
             gradient background, and subtle AI-themed design elements.

SETUP:
  Engine 0 (Gemini):       Get free key at https://aistudio.google.com/
                           Set GEMINI_API_KEY in .env
  Engine 1 (Pollinations): No key needed — free AI generation
  Engine 2 (Pexels):       Get free key at https://www.pexels.com/api/
                           Set PEXELS_API_KEY in .env
  Engine 3 (Unsplash):     Get free key at https://unsplash.com/developers
                           Set UNSPLASH_ACCESS_KEY in .env
  Engine 4 (SVG):          No key needed — always works as guaranteed fallback

Only active when ENABLE_IMAGE_GENERATION=true.
"""

import logging
import os
import random
import tempfile
from typing import Optional

import requests

from config import settings

logger = logging.getLogger(__name__)


# ── Topic → Search Keywords mapping ──────────────────────────────────────────
# Maps AI topic keywords to photo search terms that return
# professional, relevant, LinkedIn-appropriate images.

TOPIC_SEARCH_MAP = {
    "language model":    ["artificial intelligence neural network", "machine learning technology", "data science computing"],
    "llm":               ["artificial intelligence", "machine learning server", "neural network visualization"],
    "agent":             ["artificial intelligence robot", "autonomous technology", "AI automation"],
    "rag":               ["database search technology", "information retrieval", "knowledge management AI"],
    "neural network":    ["neural network technology", "artificial intelligence brain", "deep learning visualization"],
    "deep learning":     ["deep learning AI", "artificial intelligence computing", "machine learning research"],
    "machine learning":  ["machine learning", "data science technology", "AI algorithm"],
    "transformer":       ["AI architecture technology", "transformer neural network", "attention mechanism computing"],
    "diffusion":         ["AI generated art technology", "generative AI", "computational creativity"],
    "robotics":          ["robotics technology", "AI robot", "autonomous robot machine"],
    "computer vision":   ["computer vision AI", "image recognition technology", "visual AI computing"],
    "chip":              ["computer chip processor", "semiconductor technology", "CPU circuit board"],
    "infrastructure":    ["cloud computing infrastructure", "server data center", "cloud technology"],
    "openai":            ["artificial intelligence technology", "AI research computing", "language model AI"],
    "research":          ["AI research laboratory", "scientific computing technology", "data research"],
    "safety":            ["AI safety technology", "cybersecurity AI", "secure computing"],
    "automation":        ["automation technology", "AI workflow", "robotic process automation"],
    "benchmark":         ["AI performance metrics", "technology testing", "computing benchmark"],
    "trading":           ["financial technology AI", "algorithmic trading", "fintech computing"],
    "materials":         ["scientific research laboratory", "materials science technology", "computational chemistry"],
    "privacy":           ["data privacy technology", "cybersecurity computing", "secure AI"],
    "default":           ["artificial intelligence technology", "machine learning computing", "AI innovation"],
}


def _get_search_terms(topic_title: str) -> list[str]:
    """Match topic to relevant photo search terms."""
    title_lower = topic_title.lower()
    for keyword, terms in TOPIC_SEARCH_MAP.items():
        if keyword != "default" and keyword in title_lower:
            logger.info("Photo search matched keyword: '%s'", keyword)
            return terms
    return TOPIC_SEARCH_MAP["default"]


def _save_to_temp(image_bytes: bytes, engine: str = "photo") -> Optional[str]:
    """Save raw image bytes to a temporary file and return the path."""
    try:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False,
            prefix=f"linkedin_ai_{engine}_",
        )
        tmp.write(image_bytes)
        tmp.close()
        size_kb = len(image_bytes) // 1024
        logger.info("[%s] Image saved: %s (%d KB)", engine.upper(), tmp.name, size_kb)
        return tmp.name
    except OSError as exc:
        logger.error("Failed to save image: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 0 — Gemini Flash Image (FREE tier via AI Studio, premium infographics)
# Model: gemini-2.5-flash-image
# Get your free key: https://aistudio.google.com/
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_gemini_imagen(topic_title: str) -> Optional[str]:
    """
    Generate a high-quality infographic image using Gemini Flash Image.

    Uses gemini-2.5-flash-image via generate_content(),
    which is the correct free-tier approach (the old generate_images() +
    imagen-3.0-generate-002 are deprecated and unavailable on the free API).

    Requires GEMINI_API_KEY from Google AI Studio (free at aistudio.google.com).
    """
    api_key = getattr(settings, "GEMINI_API_KEY", "")
    if not api_key:
        logger.info("GEMINI_API_KEY not set — skipping Gemini engine.")
        return None

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning("google-genai not installed. Run: pip install google-genai")
        return None

    infographic_prompt = (
        f"Create a professional, minimalist LinkedIn infographic about: {topic_title}. "
        "Use a premium blue and gray corporate color palette. "
        "Include a clear main title, 3 bullet points with key insights, and minimal icons. "
        "Make the text highly legible with clean typography."
    )


    try:
        logger.info("Gemini Flash Image: generating infographic for '%s'…", topic_title[:50])
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=infographic_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        # Extract raw image bytes from the response parts
        image_data = None
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                image_data = part.inline_data.data
                break

        if not image_data:
            logger.warning("Gemini returned no image data in response parts.")
            return None

        return _save_to_temp(image_data, engine="gemini")

    except Exception as exc:
        logger.warning("Gemini image generation failed: %s", exc)

    return None



# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 1 — Pexels API (FREE, 200 req/hr, professional photography)
# Get your free key: https://www.pexels.com/api/
# ══════════════════════════════════════════════════════════════════════════════

PEXELS_API = "https://api.pexels.com/v1/search"


def _fetch_pexels(topic_title: str) -> Optional[str]:
    """
    Fetch a professional photo from Pexels matching the AI topic.

    Free tier: 200 requests/hour, 20,000/month.
    Requires PEXELS_API_KEY (free at pexels.com/api).

    Returns path to downloaded image, or None if unavailable.
    """
    api_key = os.getenv("PEXELS_API_KEY", "")
    if not api_key:
        logger.info("PEXELS_API_KEY not set — skipping Pexels engine.")
        return None

    search_terms = _get_search_terms(topic_title)

    for query in search_terms:
        try:
            logger.info("Pexels: searching '%s'…", query)
            resp = requests.get(
                PEXELS_API,
                headers={"Authorization": api_key},
                params={
                    "query":       query,
                    "orientation": "landscape",
                    "size":        "large",
                    "per_page":    15,
                },
                timeout=15,
            )

            if resp.status_code == 401:
                logger.warning("Pexels: invalid API key.")
                return None
            if resp.status_code == 429:
                logger.warning("Pexels: rate limit reached.")
                return None
            if resp.status_code != 200:
                logger.warning("Pexels returned HTTP %d.", resp.status_code)
                continue

            photos = resp.json().get("photos", [])
            if not photos:
                logger.info("Pexels: no photos for '%s', trying next query.", query)
                continue

            # Pick a random photo from results for variety
            photo = random.choice(photos[:10])
            photo_url = photo.get("src", {}).get("large2x") or photo.get("src", {}).get("original", "")

            if not photo_url:
                continue

            # Download the actual image
            img_resp = requests.get(photo_url, timeout=30)
            if img_resp.status_code == 200 and len(img_resp.content) > 10000:
                logger.info(
                    "Pexels: downloaded photo by %s",
                    photo.get("photographer", "unknown"),
                )
                return _save_to_temp(img_resp.content, engine="pexels")

        except requests.RequestException as exc:
            logger.warning("Pexels request failed for '%s': %s", query, exc)
            continue

    return None


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 2 — Unsplash API (FREE, 50 req/hr, curated professional photos)
# Get your free key: https://unsplash.com/developers
# ══════════════════════════════════════════════════════════════════════════════

UNSPLASH_API = "https://api.unsplash.com/search/photos"


def _fetch_unsplash(topic_title: str) -> Optional[str]:
    """
    Fetch a professional photo from Unsplash matching the AI topic.

    Free tier: 50 requests/hour.
    Requires UNSPLASH_ACCESS_KEY (free at unsplash.com/developers).

    Returns path to downloaded image, or None if unavailable.
    """
    access_key = os.getenv("UNSPLASH_ACCESS_KEY", "")
    if not access_key:
        logger.info("UNSPLASH_ACCESS_KEY not set — skipping Unsplash engine.")
        return None

    search_terms = _get_search_terms(topic_title)

    for query in search_terms:
        try:
            logger.info("Unsplash: searching '%s'…", query)
            resp = requests.get(
                UNSPLASH_API,
                params={
                    "query":       query,
                    "orientation": "landscape",
                    "per_page":    15,
                    "content_filter": "high",
                    "client_id":   access_key,
                },
                timeout=15,
            )

            if resp.status_code == 401:
                logger.warning("Unsplash: invalid access key.")
                return None
            if resp.status_code == 403:
                logger.warning("Unsplash: rate limit or access denied.")
                return None
            if resp.status_code != 200:
                logger.warning("Unsplash returned HTTP %d.", resp.status_code)
                continue

            results = resp.json().get("results", [])
            if not results:
                logger.info("Unsplash: no results for '%s', trying next.", query)
                continue

            # Pick a random result for variety
            photo = random.choice(results[:10])
            photo_url = (
                photo.get("urls", {}).get("regular")
                or photo.get("urls", {}).get("full", "")
            )

            if not photo_url:
                continue

            img_resp = requests.get(photo_url, timeout=30)
            if img_resp.status_code == 200 and len(img_resp.content) > 10000:
                logger.info(
                    "Unsplash: downloaded photo by %s",
                    photo.get("user", {}).get("name", "unknown"),
                )
                return _save_to_temp(img_resp.content, engine="unsplash")

        except requests.RequestException as exc:
            logger.warning("Unsplash request failed for '%s': %s", query, exc)
            continue

    return None


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 3 — Pollinations AI (FREE, AI generated enhanced pictures)
# Generates realistic AI images from text prompts.
# ══════════════════════════════════════════════════════════════════════════════

def _refine_topic_for_image(topic_title: str) -> str:
    """Use Groq LLM to refine the topic title into a visual scene description."""
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage
        
        llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model="openai/gpt-oss-120b",
            temperature=0.7,
            max_tokens=150,
        )
        
        messages = [
        SystemMessage(content=(
            "You are a world-class visual prompt engineer and cinematic art director with 20 years of "
            "experience creating award-winning LinkedIn banner imagery for Fortune 500 companies. "
            "Your sole job is to transform a topic title into a breathtaking, photorealistic visual scene "
            "that perfectly represents the topic's essence through imagery alone — no text, ever. "

            "STRICT OUTPUT RULES: "
            "- Return ONLY the scene description. No preamble, no explanation, no quotes, no extra text. "
            "- Maximum 2 crisp, vivid sentences. "
            "- Every word must serve a visual purpose. "

            "YOUR SCENE MUST INCLUDE ALL OF THESE LAYERS: "
            "1. SUBJECT: The dominant visual metaphor or object that instantly communicates the topic. "
            "2. HUMAN ELEMENT: Professional human figures, hands, silhouettes, or interactions where relevant. "
            "3. ENVIRONMENT: A specific, detailed setting (boardroom, data center, sky, cityscape, lab). "
            "4. LIGHTING MOOD: Cinematic lighting style (golden hour, cool blue neon, dramatic rim light, soft diffused). "
            "5. COLOR PALETTE: 2-3 dominant colors that reflect the topic's emotional tone. "
            "6. DEPTH & COMPOSITION: Foreground detail, midground action, background atmosphere. "
            "7. TEXTURE & DETAIL: Surface materials, particle effects, atmospheric haze, reflections. "

            "STYLE REFERENCES TO EMULATE: "
            "Shot on Hasselblad medium format, Unreal Engine 5 cinematic realism, "
            "Behance top-rated commercial photography, Netflix documentary visual style. "

            "ABSOLUTE PROHIBITIONS: "
            "Never include: text, words, letters, numbers, signs, labels, watermarks, "
            "logos, captions, symbols, or any readable inscription anywhere in the scene. "

            "THINK LIKE THIS: If a viewer saw this image with zero context, "
            "they should instantly understand the topic through visuals alone. "
            "Make it so powerful, relevant, and stunning that it stops the scroll."
        )),
            HumanMessage(content=topic_title)
        ]
        
        response = llm.invoke(messages)
        refined_topic = response.content.strip()
        
        if refined_topic:
            return refined_topic
    except Exception as exc:
        logger.warning("Groq image topic refinement failed: %s", exc)
        
    return topic_title


def _fetch_pollinations(topic_title: str) -> Optional[str]:
    """
    Generate an enhanced and realistic picture using Pollinations AI.
    Free, no API key required.
    """
    from urllib.parse import quote_plus

    refined_topic = _refine_topic_for_image(topic_title)
    logger.info("Original topic: '%s'", topic_title)
    logger.info("Refined visual topic: '%s'", refined_topic)

    prompt = (
        # 🎯 Core Subject
        f"LinkedIn professional banner, subject: {refined_topic}. "

        # 📷 Camera & Lens
        f"Shot on Hasselblad H6D-100c medium format camera, "
        f"85mm f/1.4 prime lens, razor-sharp focus, zero motion blur, "

        # 🌟 Lighting
        f"volumetric cinematic lighting, dramatic rim lighting, "
        f"HDR global illumination, ray-traced reflections, "
        f"golden ratio light diffusion, god rays, "

        # 🎨 Rendering & Quality
        f"Unreal Engine 5 render quality, Octane render, "
        f"8K ultra-high resolution, hyper-detailed textures, "
        f"masterpiece quality, award-winning commercial photography, "

        # 🖼️ Composition
        f"wide cinematic panoramic composition, rule of thirds, "
        f"foreground-midground-background depth layers, "
        f"ultra-sharp foreground with subtle bokeh depth of field, "
        f"dynamic symmetrical layout perfect for LinkedIn banner, "

        # 🎭 Style & Aesthetic
        f"Fortune 500 corporate brand aesthetic, "
        f"modern sleek futuristic professional design, "
        f"premium editorial magazine cover quality, "
        f"trending on Behance and ArtStation, "
        f"DaVinci Resolve cinematic color grade, "
        f"sophisticated vibrant color palette with deep contrast, "

        # 🚫 Strict No-Text
        f"absolutely no text, no words, no letters, no numbers, "
        f"no typography, no captions, no labels, no watermarks, "
        f"no logos, no symbols, no inscriptions, text-free image only."
    )

    url = (
        f"https://image.pollinations.ai/prompt/{quote_plus(prompt)}"
        f"?width=2400&height=1200&nologo=true&model=flux&enhance=true&seed=42"
    )

    try:
        logger.info("Pollinations AI: generating image for '%s'…", topic_title[:50])
        resp = requests.get(url, timeout=90)  # ⬆️ increased timeout for high-res

        if resp.status_code == 200 and len(resp.content) > 10000:
            logger.info("Pollinations AI: successfully generated image.")
            return _save_to_temp(resp.content, engine="pollinations")
        else:
            logger.warning("Pollinations AI returned HTTP %d or empty content.", resp.status_code)
    except requests.RequestException as exc:
        logger.warning("Pollinations AI request failed: %s", exc)

    return None


def generate_image(topic_title: str) -> Optional[str]:
    """
    Fetch or generate a professional LinkedIn header image for the given topic.

    Engine waterfall (updated):
      0. Gemini 2.5 Flash Image — FREE via AI Studio, AI infographics, needs GEMINI_API_KEY
      1. Pollinations AI        — FREE, AI generated pictures, no key needed
      2. Pexels API             — FREE, professional photos, needs PEXELS_API_KEY
      3. Unsplash API           — FREE, curated photos, needs UNSPLASH_ACCESS_KEY

    Returns:
        Absolute path to the image file (jpg), or None if all fail.
        None → pipeline publishes post as text-only (never crashes).
    """
    logger.info("Image generation started for topic: '%s'", topic_title[:80])

    # Engine 0: Gemini Flash Image
    logger.info("Trying Engine 0: Gemini 2.5 Flash Image (FREE AI infographics)…")
    result = _fetch_gemini_imagen(topic_title)
    if result:
        logger.info("✅ Engine 0 (Gemini) succeeded.")
        return result

    # Engine 1: Pollinations AI
    logger.info("Trying Engine 1: Pollinations AI (FREE AI generated photos)…")
    result = _fetch_pollinations(topic_title)
    if result:
        logger.info("✅ Engine 1 (Pollinations AI) succeeded.")
        return result

    # Engine 2: Pexels
    logger.info("Trying Engine 2: Pexels API (FREE professional photos)…")
    result = _fetch_pexels(topic_title)
    if result:
        logger.info("✅ Engine 2 (Pexels) succeeded.")
        return result

    # Engine 3: Unsplash
    logger.info("Trying Engine 3: Unsplash API (FREE curated photos)…")
    result = _fetch_unsplash(topic_title)
    if result:
        logger.info("✅ Engine 3 (Unsplash) succeeded.")
        return result

    logger.error("❌ All image engines failed.")
    return None


def cleanup_image(image_path: Optional[str]) -> None:
    """Remove the temporary image file after LinkedIn upload."""
    if image_path and os.path.exists(image_path):
        try:
            os.remove(image_path)
            logger.info("Cleaned up temp image: %s", image_path)
        except OSError as exc:
            logger.warning("Could not delete temp image: %s", exc)