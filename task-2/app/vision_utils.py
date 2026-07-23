"""
Classical, offline image-analysis utilities.

No internet access is available in this build environment to call a cloud
vision API or download a vision-language model's weights, so this module
extracts real, non-fabricated visual evidence using classical computer
vision instead of hallucinating a caption:
  - OCR text extraction (pytesseract / Tesseract)
  - Color and brightness statistics
  - Basic composition stats (size, aspect ratio, edge density as a proxy
    for visual complexity)

The rest of the app (app.py) treats this as the "vision evidence" that
grounds its responses -- swap this module for a real multimodal LLM call
(e.g. Claude/GPT-4V, or a local LLaVA model via Ollama) without touching
anything else, see README.
"""

import numpy as np
from PIL import Image
import pytesseract


def extract_text(image: Image.Image) -> str:
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception:
        return ""


def color_profile(image: Image.Image, n_colors: int = 5):
    small = image.convert("RGB").resize((100, 100))
    arr = np.array(small).reshape(-1, 3)
    # simple k-means-free binning: quantize to a coarse palette and count
    quantized = (arr // 32 * 32)
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    order = np.argsort(-counts)[:n_colors]
    total = counts.sum()
    return [
        {"rgb": tuple(int(c) for c in colors[i]), "pct": round(100 * counts[i] / total, 1)}
        for i in order
    ]


def brightness_and_contrast(image: Image.Image):
    gray = np.array(image.convert("L"), dtype=np.float32)
    return {
        "mean_brightness": round(float(gray.mean()), 1),
        "contrast_std": round(float(gray.std()), 1),
    }


def edge_density(image: Image.Image):
    """Simple gradient-magnitude based complexity proxy (no extra deps)."""
    gray = np.array(image.convert("L"), dtype=np.float32)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    return round(float((gx + gy) / 2), 2)


def analyze_image(image: Image.Image) -> dict:
    w, h = image.size
    text = extract_text(image)
    colors = color_profile(image)
    bc = brightness_and_contrast(image)
    density = edge_density(image)

    if density > 25:
        complexity = "highly detailed / busy"
    elif density > 10:
        complexity = "moderately detailed"
    else:
        complexity = "simple / low-detail"

    if bc["mean_brightness"] > 170:
        tone = "bright/light"
    elif bc["mean_brightness"] < 80:
        tone = "dark"
    else:
        tone = "medium brightness"

    dominant = colors[0]["rgb"] if colors else None

    return {
        "width": w,
        "height": h,
        "aspect_ratio": round(w / h, 2) if h else None,
        "ocr_text": text,
        "has_text": len(text) > 2,
        "colors": colors,
        "dominant_color_rgb": dominant,
        "tone": tone,
        "brightness": bc["mean_brightness"],
        "contrast": bc["contrast_std"],
        "complexity": complexity,
    }


def describe_evidence(evidence: dict) -> str:
    """Turns raw extracted evidence into a short, honest natural-language
    description -- this is what the response is grounded in, so the model
    only claims things it actually measured."""
    parts = [
        f"Image is {evidence['width']}x{evidence['height']}px, {evidence['tone']}, "
        f"{evidence['complexity']}."
    ]
    if evidence["has_text"]:
        snippet = evidence["ocr_text"][:300]
        parts.append(f"Text detected in the image via OCR: \"{snippet}\"")
    else:
        parts.append("No readable text was detected in the image.")
    if evidence["dominant_color_rgb"]:
        parts.append(f"Dominant color (approx RGB): {evidence['dominant_color_rgb']}.")
    return " ".join(parts)
