"""
tools/vision_tool.py
Groq LLaMA Vision tool — analyzes images from URL or base64.
This is what makes the assistant MULTIMODAL.
"""

from langchain.tools import tool
from groq import Groq
from config.settings import settings
from loguru import logger
import base64
import httpx
from pathlib import Path


def _load_image_as_base64(image_source: str) -> tuple[str, str]:
    """
    Load image from file path or URL and return (base64_data, media_type).
    """
    # ── Local file ──────────────────────────────────────────────────────────────
    if Path(image_source).exists():
        path = Path(image_source)
        suffix = path.suffix.lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_type_map.get(suffix, "image/jpeg")
        with open(path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("utf-8")
        return b64, media_type

    # ── URL ─────────────────────────────────────────────────────────────────────
    if image_source.startswith("http"):
        response = httpx.get(image_source, timeout=15)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "image/jpeg")
        media_type = content_type.split(";")[0].strip()
        b64 = base64.standard_b64encode(response.content).decode("utf-8")
        return b64, media_type

    # ── Already base64 ──────────────────────────────────────────────────────────
    return image_source, "image/jpeg"


@tool
def vision_analyzer(image_source: str, question: str = "Describe this image in detail.") -> str:
    """
    Analyze an image using Groq's LLaMA Vision model.
    Use this when the user provides an image (URL or file path) and asks questions about it.
    This tool makes the assistant multimodal — it can SEE and understand images.

    Args:
        image_source: URL or local file path to the image
        question: What to ask about the image (default: describe the image)

    Returns:
        Detailed analysis/description of the image
    """
    try:
        logger.debug(f"[vision_analyzer] source='{image_source[:50]}...' question='{question}'")

        client = Groq(api_key=settings.groq_api_key)

        # ── Build message content ───────────────────────────────────────────────
        # Try URL first (simpler), fall back to base64
        if image_source.startswith("http"):
            image_content = {
                "type": "image_url",
                "image_url": {"url": image_source}
            }
        else:
            b64_data, media_type = _load_image_as_base64(image_source)
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{b64_data}"
                }
            }

        messages = [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": question}
                ]
            }
        ]

        response = client.chat.completions.create(
            model=settings.model_vision,
            messages=messages,
            max_tokens=1024,
        )

        analysis = response.choices[0].message.content
        logger.debug(f"[vision_analyzer] analysis length={len(analysis)}")
        return analysis

    except Exception as e:
        logger.error(f"[vision_analyzer] error: {e}")
        return f"Vision analysis failed: {str(e)}"
