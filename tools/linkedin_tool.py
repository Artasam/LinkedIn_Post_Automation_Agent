"""
tools/linkedin_tool.py
----------------------
Handles all interactions with the LinkedIn API v2.
Supports text-only posts and (optionally) image-attached posts.
"""

import logging
import os
from typing import Optional

import requests

from config import settings

logger = logging.getLogger(__name__)


class LinkedInAPIError(Exception):
    """Raised when the LinkedIn API returns an error response."""
    pass


def _auth_headers() -> dict:
    """Build the authorization headers required by LinkedIn API v2."""
    return {
        "Authorization": f"Bearer {settings.LINKEDIN_ACCESS_TOKEN}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
        "LinkedIn-Version": "202304",
    }


def validate_token() -> bool:
    """
    Ping the /me endpoint to confirm the access token is valid.
    Returns True if valid, False otherwise.
    """
    url = f"{settings.LINKEDIN_API_BASE}/me"
    try:
        resp = requests.get(url, headers=_auth_headers(), timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            logger.info(
                "LinkedIn token valid for user: %s %s",
                data.get("localizedFirstName", ""),
                data.get("localizedLastName", ""),
            )
            return True
        logger.warning("LinkedIn token validation failed: %s", resp.text)
        return False
    except requests.RequestException as exc:
        logger.error("LinkedIn token check request failed: %s", exc)
        return False


def publish_text_post(text: str) -> dict:
    """
    Publish a plain-text post to LinkedIn as the authenticated user.

    Args:
        text: The full post content (including hashtags).

    Returns:
        The LinkedIn API response dict.

    Raises:
        LinkedInAPIError: On any non-2xx response.
    """
    url = f"{settings.LINKEDIN_API_BASE}/ugcPosts"

    payload = {
        "author": f"urn:li:person:{settings.LINKEDIN_PERSON_ID}",
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": text},
                "shareMediaCategory": "NONE",
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        },
    }

    try:
        resp = requests.post(url, json=payload, headers=_auth_headers(), timeout=15)
    except requests.RequestException as exc:
        raise LinkedInAPIError(f"Request to LinkedIn failed: {exc}") from exc

    if resp.status_code in (200, 201):
        result = resp.json() if resp.text else {}
        post_id = resp.headers.get("X-RestLi-Id", result.get("id", "unknown"))
        logger.info("Post published successfully. Post ID: %s", post_id)
        return {"success": True, "post_id": post_id, "status_code": resp.status_code}

    raise LinkedInAPIError(
        f"LinkedIn API returned {resp.status_code}: {resp.text}"
    )


def upload_image_for_post(image_path: str) -> Optional[str]:
    """
    Upload an image asset to LinkedIn and return the asset URN.
    Used when ENABLE_IMAGE_GENERATION is True.

    Returns:
        The asset URN string, or None on failure.
    """
    # Step 1: Register the image upload
    register_url = f"{settings.LINKEDIN_API_BASE}/assets?action=registerUpload"
    register_payload = {
        "registerUploadRequest": {
            "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
            "owner": f"urn:li:person:{settings.LINKEDIN_PERSON_ID}",
            "serviceRelationships": [
                {
                    "relationshipType": "OWNER",
                    "identifier": "urn:li:userGeneratedContent",
                }
            ],
        }
    }

    try:
        reg_resp = requests.post(
            register_url,
            json=register_payload,
            headers=_auth_headers(),
            timeout=15,
        )
        reg_resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Image registration failed: %s", exc)
        return None

    reg_data = reg_resp.json()
    upload_url = (
        reg_data.get("value", {})
        .get("uploadMechanism", {})
        .get("com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest", {})
        .get("uploadUrl", "")
    )
    asset_urn = reg_data.get("value", {}).get("asset", "")

    if not upload_url or not asset_urn:
        logger.error("Could not extract upload URL or asset URN from response.")
        return None

    # Step 2: Upload the binary image
    try:
        with open(image_path, "rb") as img_file:
            img_headers = {
                "Authorization": f"Bearer {settings.LINKEDIN_ACCESS_TOKEN}",
                "Content-Type": "application/octet-stream",
            }
            put_resp = requests.put(upload_url, data=img_file, headers=img_headers, timeout=30)
            put_resp.raise_for_status()
        logger.info("Image uploaded successfully. Asset URN: %s", asset_urn)
        return asset_urn
    except (requests.RequestException, OSError) as exc:
        logger.error("Image upload failed: %s", exc)
        return None


def publish_image_post(text: str, image_path: str) -> dict:
    """
    Publish a LinkedIn post with an attached image.

    Falls back to text-only if image upload fails.
    """
    asset_urn = upload_image_for_post(image_path)

    if not asset_urn:
        logger.warning("Image upload failed — falling back to text-only post.")
        return publish_text_post(text)

    url = f"{settings.LINKEDIN_API_BASE}/ugcPosts"
    payload = {
        "author": f"urn:li:person:{settings.LINKEDIN_PERSON_ID}",
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": text},
                "shareMediaCategory": "IMAGE",
                "media": [
                    {
                        "status": "READY",
                        "description": {"text": "AI trend illustration"},
                        "media": asset_urn,
                        "title": {"text": "AI Insight"},
                    }
                ],
            }
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    }

    try:
        resp = requests.post(url, json=payload, headers=_auth_headers(), timeout=15)
    except requests.RequestException as exc:
        raise LinkedInAPIError(f"Request failed: {exc}") from exc

    if resp.status_code in (200, 201):
        post_id = resp.headers.get("X-RestLi-Id", "unknown")
        logger.info("Image post published. ID: %s", post_id)
        return {"success": True, "post_id": post_id, "status_code": resp.status_code}

    raise LinkedInAPIError(f"LinkedIn API {resp.status_code}: {resp.text}")
