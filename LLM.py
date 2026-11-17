"""
Bounding-box detection with Gemma 4 served by llama.cpp (llama-server).

Start the server with the vision projector, e.g.:
    llama-server -m gemma-4-xx.gguf --mmproj mmproj-gemma-4-xx.gguf --port 8080

Requires: numpy, pillow, openai
"""
from __future__ import annotations

import base64
import io
import json
import re
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image

# JSON schema enforced by llama.cpp's grammar-constrained sampling.
_BBOX_SCHEMA = {
    "type": "object",
    "properties": {
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    # [y_min, x_min, y_max, x_max] normalized to 0-1000 (Gemma/Gemini convention)
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["label", "box_2d"],
            },
        }
    },
    "required": ["detections"],
}

_SYSTEM_PROMPT = (
    "You are an object detector. The user will request that you should grab an object. Return ONLY JSON of a bounding box around the requested object"
    '{"detections": [{"label": str, "box_2d": [y_min, x_min, y_max, x_max]}]}. '
    "Coordinates are integers normalized to 0-1000 relative to image height (y) "
    "and width (x). Return an empty list if nothing matches."
)


def _ndarray_to_data_url(image: np.ndarray) -> str:
    """Encode an HxW, HxWx3 (RGB) or HxWx4 numpy array as a base64 PNG data URL."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        # Assume float images are 0-1; otherwise clip into uint8 range.
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _extract_json(text: str) -> Any:
    """Parse JSON, tolerating ```json fences or stray text around it."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(1))


def get_bounding_boxes(
    image: np.ndarray,
    prompt: str,
    server_url: str = "http://localhost:8080",
    model: str = "gemma-4",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: float = 120.0,
    use_schema: bool = True,
    api_key: str = "sk-no-key-required",
    client: OpenAI | None = None,
) -> list[dict]:
    """
    Send an image + prompt to Gemma 4 on llama-server and return bounding boxes.

    Args:
        image:  numpy array (H, W) or (H, W, 3/4), RGB channel order.
                (If from cv2.imread, convert first: image[..., ::-1])
        prompt: what to detect, e.g. "Detect every bolt and washer."
        client: optional pre-built OpenAI client (reuse it when calling in a loop).

    Returns:
        List of dicts:
            {
              "label": str,
              "box_2d": [y_min, x_min, y_max, x_max],   # raw 0-1000 model output
              "bbox_xyxy": [x_min, y_min, x_max, y_max] # pixel coords in `image`
            }
    """
    h, w = image.shape[:2]

    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _ndarray_to_data_url(image)}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    }
    if use_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "detections", "schema": _BBOX_SCHEMA},
        }

    if client is None:
        client = OpenAI(
            base_url=f"{server_url.rstrip('/')}/v1",
            api_key=api_key,  # llama-server ignores it unless started with --api-key
            timeout=timeout,
        )
    resp = client.chat.completions.create(**payload)
    content = resp.choices[0].message.content or ""

    data = _extract_json(content)
    detections = data.get("detections", []) if isinstance(data, dict) else data

    results = []
    for det in detections:
        box = det.get("box_2d") or det.get("bbox")
        if not box or len(box) != 4:
            continue
        y0, x0, y1, x1 = (float(v) for v in box)
        # Normalize ordering and clamp to image bounds.
        x_min, x_max = sorted((x0 / 1000 * w, x1 / 1000 * w))
        y_min, y_max = sorted((y0 / 1000 * h, y1 / 1000 * h))
        results.append(
            {
                "label": det.get("label", ""),
                "box_2d": [y0, x0, y1, x1],
                "bbox_xyxy": [
                    int(round(max(0, min(w, x_min)))),
                    int(round(max(0, min(h, y_min)))),
                    int(round(max(0, min(w, x_max)))),
                    int(round(max(0, min(h, y_max)))),
                ],
            }
        )
    return results


if __name__ == "__main__":
    import sys

    img = np.array(Image.open(sys.argv[1]).convert("RGB"))
    boxes = get_bounding_boxes(img, sys.argv[2] if len(sys.argv) > 2 else "Detect all objects.")
    print(json.dumps(boxes, indent=2))
