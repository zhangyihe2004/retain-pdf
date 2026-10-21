"""Run the remote dots.mocr OpenAI-compatible API through local_command OCR."""

from __future__ import annotations

import base64
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import math
import os
from pathlib import Path
from typing import Any

import fitz
import requests
from json_repair import loads as repair_json_loads

from retainpdf_pipeline.ocr.document_schema.provider_adapters.dots_mocr.projection import (
    project_dots_mocr_block,
)


DEFAULT_DOTS_URL = "http://127.0.0.1:41010/v1/chat/completions"
DOTS_URL_ENV = "RETAIN_DOTS_LOCAL_API_URL"
DOTS_MODEL_ENV = "RETAIN_DOTS_LOCAL_MODEL"
DOTS_TIMEOUT_ENV = "RETAIN_DOTS_LOCAL_API_TIMEOUT_SECONDS"
DOTS_DPI_ENV = "RETAIN_DOTS_RENDER_DPI"
DOTS_API_KEY_ENV = "RETAIN_DOTS_LOCAL_API_KEY"
DOTS_NUM_THREAD_ENV = "RETAIN_DOTS_NUM_THREAD"
DOTS_MAX_COMPLETION_TOKENS_ENV = "RETAIN_DOTS_MAX_COMPLETION_TOKENS"
DEFAULT_MODEL = "model"
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_RENDER_DPI = 200
DEFAULT_NUM_THREAD = 16
DEFAULT_MAX_COMPLETION_TOKENS = 16384
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 1.0
MIN_IMAGE_PIXELS = 3136
MAX_IMAGE_PIXELS = 11289600
IMAGE_FACTOR = 28
MAX_RENDER_DIMENSION = 4500
OUT_OF_PAGE_MARGIN_RATIO = 0.05
VLLM_IMAGE_PREFIX = "<|img|><|imgpad|><|endofimg|>"

# Kept aligned with studio-dots-ai/dots.mocr's prompt_layout_all_en.
LAYOUT_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].
3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.
5. Final Output: The entire output must be a single JSON object.
"""

# Official prompt_layout_only_en and prompt_ocr, used only after truncation.
LAYOUT_ONLY_PROMPT = "Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format."
OCR_PROMPT = "Extract the text content from this image."


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


def _env_positive_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _model_input_dimensions(image_width: int, image_height: int) -> tuple[int, int]:
    """Match dots.mocr's factor-28 smart_resize used for bbox coordinates."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("dots image dimensions must be positive")
    if max(image_width, image_height) / min(image_width, image_height) > 200:
        raise ValueError("dots image aspect ratio must be at most 200")
    height = max(IMAGE_FACTOR, round(image_height / IMAGE_FACTOR) * IMAGE_FACTOR)
    width = max(IMAGE_FACTOR, round(image_width / IMAGE_FACTOR) * IMAGE_FACTOR)
    if height * width > MAX_IMAGE_PIXELS:
        ratio = math.sqrt(image_height * image_width / MAX_IMAGE_PIXELS)
        height = max(IMAGE_FACTOR, math.floor(image_height / ratio / IMAGE_FACTOR) * IMAGE_FACTOR)
        width = max(IMAGE_FACTOR, math.floor(image_width / ratio / IMAGE_FACTOR) * IMAGE_FACTOR)
    elif height * width < MIN_IMAGE_PIXELS:
        ratio = math.sqrt(MIN_IMAGE_PIXELS / (image_height * image_width))
        height = math.ceil(image_height * ratio / IMAGE_FACTOR) * IMAGE_FACTOR
        width = math.ceil(image_width * ratio / IMAGE_FACTOR) * IMAGE_FACTOR
        if height * width > MAX_IMAGE_PIXELS:
            ratio = math.sqrt(height * width / MAX_IMAGE_PIXELS)
            height = max(IMAGE_FACTOR, math.floor(height / ratio / IMAGE_FACTOR) * IMAGE_FACTOR)
            width = max(IMAGE_FACTOR, math.floor(width / ratio / IMAGE_FACTOR) * IMAGE_FACTOR)
    return width, height


def _render_page(page: fitz.Page, dpi: int) -> tuple[fitz.Pixmap, float]:
    """Use the official PDF raster size limits without rendering oversized pages first."""
    page_width, page_height = float(page.rect.width), float(page.rect.height)
    scale = dpi / 72.0
    if page_width * page_height > MAX_IMAGE_PIXELS:
        scale = math.sqrt(MAX_IMAGE_PIXELS / (page_width * page_height))
    if max(page_width * scale, page_height * scale) > MAX_RENDER_DIMENSION:
        scale = 1.0
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
    if max(pixmap.width, pixmap.height) > MAX_RENDER_DIMENSION and scale != 1.0:
        scale = 1.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1, 1), colorspace=fitz.csRGB, alpha=False)
    return pixmap, scale * 72.0


def _source_pdf() -> Path:
    value = str(os.environ.get("RETAIN_OCR_SOURCE_PDF", "") or "").strip()
    if not value:
        raise RuntimeError("RETAIN_OCR_SOURCE_PDF is required")
    path = Path(value).resolve()
    if not path.is_file():
        raise RuntimeError(f"source PDF does not exist: {path}")
    return path


def _output_path() -> Path:
    value = str(os.environ.get("RETAIN_OCR_RAW_PAYLOAD_JSON", "") or "").strip()
    if not value:
        raise RuntimeError("RETAIN_OCR_RAW_PAYLOAD_JSON is required")
    return Path(value).resolve()


def _extract_json(text: str) -> Any:
    candidate = str(text or "").strip()
    if not candidate:
        raise RuntimeError("dots response content is empty")
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    start = next((index for index, char in enumerate(candidate) if char in "{["), None)
    if start is None:
        raise RuntimeError("dots response did not contain a valid JSON object or array")
    candidate = candidate[start:]
    if not candidate.endswith("}" if candidate.startswith("{") else "]"):
        raise RuntimeError("dots response JSON object or array is incomplete")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        if exc.pos >= len(candidate) or exc.msg.startswith("Unterminated string"):
            raise RuntimeError("dots response JSON object or array is incomplete") from exc
    # Repair the whole response, never salvage a nested header as the whole page.
    try:
        return repair_json_loads(candidate, skip_json_loads=True, strict=True)
    except ValueError as exc:
        raise RuntimeError("dots response JSON object or array could not be parsed") from exc


def _message_content(response_payload: dict[str, Any]) -> str:
    try:
        message = response_payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("dots chat completion has no choices[0].message") from exc
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("text")
        )
    raise RuntimeError("dots chat completion message content is not text")


def _layout_items(value: Any) -> list[dict[str, Any]]:
    # The official prompt asks for an object, while model/server versions may
    # return a bare array. Accept both, plus common wrappers around the list.
    if isinstance(value, list):
        items = value
    elif isinstance(value, dict):
        items = None
        for key in ("layout", "elements", "items", "blocks", "result"):
            nested = value.get(key)
            if isinstance(nested, list):
                items = nested
                break
        if items is None and any(key in value for key in ("category", "bbox")):
            items = [value]
        if items is None:
            raise RuntimeError("dots JSON object has no layout element list")
    else:
        raise RuntimeError("dots layout JSON must be an object or array")
    normalized = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise RuntimeError(f"dots layout item {index} is not an object")
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise RuntimeError(f"dots layout item {index} has invalid bbox")
        try:
            bbox_values = [float(value) for value in bbox]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"dots layout item {index} bbox is not numeric") from exc
        if not all(math.isfinite(value) for value in bbox_values):
            raise RuntimeError(f"dots layout item {index} bbox is not finite")
        category = str(item.get("category", item.get("type", "")) or "").strip()
        if not category:
            raise RuntimeError(f"dots layout item {index} has no category")
        normalized.append(
            {
                "category": category,
                "bbox": bbox_values,
                "text": str(item.get("text", "") or ""),
            }
        )
    return normalized


def parse_layout_content(text: str) -> list[dict[str, Any]]:
    return _layout_items(_extract_json(text))


def _to_page_payload(
    *,
    items: list[dict[str, Any]],
    page_width_pt: float,
    page_height_pt: float,
    image_width: int,
    image_height: int,
    model_input_width: int,
    model_input_height: int,
    dpi: float,
) -> dict[str, Any]:
    scale_x = page_width_pt / model_input_width
    scale_y = page_height_pt / model_input_height
    render_scale_x = image_width / model_input_width
    render_scale_y = image_height / model_input_height
    blocks = []
    for item in items:
        raw_bbox = list(item["bbox"])
        effective_bbox = raw_bbox
        limits = (model_input_width, model_input_height, model_input_width, model_input_height)
        clipped = [max(0.0, min(value, limit)) for value, limit in zip(raw_bbox, limits)]
        if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
            # Repair only out-of-page edges. Normal geometry stays untouched;
            # extra blank space is discovered by the shared render path.
            inset = [
                limit * OUT_OF_PAGE_MARGIN_RATIO if value < 0 else
                limit * (1 - OUT_OF_PAGE_MARGIN_RATIO) if value > limit else value
                for value, limit in zip(raw_bbox, limits)
            ]
            effective_bbox = inset if inset[2] > inset[0] and inset[3] > inset[1] else clipped
        if effective_bbox != raw_bbox:
            print(f"dots OCR warning: repaired out-of-page bbox {raw_bbox} -> {[round(value, 3) for value in effective_bbox]}", flush=True)
        bbox = [
            round(effective_bbox[0] * scale_x, 3),
            round(effective_bbox[1] * scale_y, 3),
            round(effective_bbox[2] * scale_x, 3),
            round(effective_bbox[3] * scale_y, 3),
        ]
        render_bbox = [
            round(effective_bbox[0] * render_scale_x, 3),
            round(effective_bbox[1] * render_scale_y, 3),
            round(effective_bbox[2] * render_scale_x, 3),
            round(effective_bbox[3] * render_scale_y, 3),
        ]
        blocks.append(
            {
                "category": item["category"],
                "text": item["text"],
                "bbox": bbox,
                "raw_bbox_px": raw_bbox,
                "render_bbox_px": render_bbox,
            }
        )
    return {
        "width": page_width_pt,
        "height": page_height_pt,
        "unit": "pt",
        "render_dpi": dpi,
        "image_width": image_width,
        "image_height": image_height,
        "model_input_width": model_input_width,
        "model_input_height": model_input_height,
        "blocks": blocks,
    }


def build_request_payload(
    *, image_b64: str, model: str, max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    prompt: str = LAYOUT_PROMPT,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {"type": "text", "text": f"{VLLM_IMAGE_PREFIX}{prompt}"},
                ],
            }
        ],
        "max_completion_tokens": max_completion_tokens,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "stream": False,
    }


def _request_page(
    *,
    page_index: int,
    total_pages: int,
    image_b64: str,
    image_width: int,
    image_height: int,
    page_width_pt: float,
    page_height_pt: float,
    dpi: float,
    url: str,
    model: str,
    headers: dict[str, str],
    timeout: float,
    max_completion_tokens: int,
) -> dict[str, Any]:
    with requests.Session() as session:
        session.trust_env = False

        def request_content(image: str, prompt: str) -> tuple[str, bool]:
            payload = build_request_payload(
                image_b64=image, model=model, max_completion_tokens=max_completion_tokens,
                prompt=prompt,
            )
            response = session.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            response_payload = response.json()
            if not isinstance(response_payload, dict):
                raise RuntimeError(f"dots chat completion on page {page_index + 1} must be a JSON object")
            return (
                _message_content(response_payload),
                response_payload["choices"][0].get("finish_reason") == "length",
            )

        try:
            content, truncated = request_content(image_b64, LAYOUT_PROMPT)
            if truncated:
                print(
                    f"dots OCR: page {page_index + 1}/{total_pages} response truncated; "
                    "switching to layout-only and cropped OCR",
                    flush=True,
                )
                content, layout_truncated = request_content(image_b64, LAYOUT_ONLY_PROMPT)
                if layout_truncated:
                    raise RuntimeError("dots layout-only response was truncated")
            items = parse_layout_content(content)
            model_input_width, model_input_height = _model_input_dimensions(image_width, image_height)
            page_payload = _to_page_payload(
                items=items,
                page_width_pt=page_width_pt,
                page_height_pt=page_height_pt,
                image_width=image_width,
                image_height=image_height,
                model_input_width=model_input_width,
                model_input_height=model_input_height,
                dpi=dpi,
            )
            if truncated:
                if not page_payload["blocks"]:
                    raise RuntimeError("dots layout-only response has no layout elements")
                pixmap = fitz.Pixmap(base64.b64decode(image_b64))
                for index, block in enumerate(page_payload["blocks"]):
                    if block["category"].strip().lower() in {"table", "picture"}:
                        continue
                    clip = fitz.Rect(block["render_bbox_px"]).irect & pixmap.irect
                    if clip.is_empty:
                        raise RuntimeError(f"dots cropped OCR item {index} has an empty crop")
                    crop = fitz.Pixmap(pixmap.colorspace, clip, pixmap.alpha)
                    crop.copy(pixmap, clip)
                    crop_b64 = base64.b64encode(crop.tobytes("png")).decode("ascii")
                    text, crop_truncated = request_content(crop_b64, OCR_PROMPT)
                    if crop_truncated or (
                        not text.strip() and project_dots_mocr_block(block["category"]).translate
                    ):
                        raise RuntimeError(f"dots cropped OCR item {index} is truncated or empty")
                    block["text"] = text.strip()
            return page_payload
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(
                f"dots OCR request failed on page {page_index + 1}/{total_pages}: {exc}"
            ) from exc
        except RuntimeError as exc:
            raise RuntimeError(f"dots OCR response failed on page {page_index + 1}/{total_pages}: {exc}") from exc


def run() -> Path:
    source_pdf = _source_pdf()
    output_path = _output_path()
    url = str(os.environ.get(DOTS_URL_ENV, DEFAULT_DOTS_URL) or DEFAULT_DOTS_URL).strip()
    model = str(os.environ.get(DOTS_MODEL_ENV, DEFAULT_MODEL) or DEFAULT_MODEL).strip()
    timeout = _env_float(DOTS_TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS)
    dpi = _env_positive_int(DOTS_DPI_ENV, DEFAULT_RENDER_DPI)
    num_thread = _env_positive_int(DOTS_NUM_THREAD_ENV, DEFAULT_NUM_THREAD)
    max_completion_tokens = _env_positive_int(
        DOTS_MAX_COMPLETION_TOKENS_ENV, DEFAULT_MAX_COMPLETION_TOKENS
    )
    if not url:
        raise RuntimeError(f"{DOTS_URL_ENV} must not be empty")
    if not model:
        raise RuntimeError(f"{DOTS_MODEL_ENV} must not be empty")

    headers = {"Content-Type": "application/json"}
    api_key = str(os.environ.get(DOTS_API_KEY_ENV, "") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with fitz.open(source_pdf) as pdf:
        total_pages = pdf.page_count
        pages: list[dict[str, Any] | None] = [None] * total_pages
        max_workers = min(num_thread, total_pages) if total_pages else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending: dict[Future[dict[str, Any]], int] = {}

            def collect_completed() -> None:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    page_index = pending.pop(future)
                    page_payload = future.result()
                    pages[page_index] = page_payload
                    print(
                        f"dots OCR: page {page_index + 1}/{total_pages}, "
                        f"blocks={len(page_payload['blocks'])}",
                        flush=True,
                    )

            for page_index, page in enumerate(pdf):
                pixmap, render_dpi = _render_page(page, dpi)
                image_b64 = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
                future = executor.submit(
                    _request_page,
                    page_index=page_index,
                    total_pages=total_pages,
                    image_b64=image_b64,
                    image_width=pixmap.width,
                    image_height=pixmap.height,
                    page_width_pt=float(page.rect.width),
                    page_height_pt=float(page.rect.height),
                    dpi=render_dpi,
                    url=url,
                    model=model,
                    headers=headers,
                    timeout=timeout,
                    max_completion_tokens=max_completion_tokens,
                )
                pending[future] = page_index
                if len(pending) >= max_workers:
                    collect_completed()
            while pending:
                collect_completed()

    payload = {
        "provider": "dots_mocr",
        "model": model,
        "source_pdf": str(source_pdf),
        "page_count": len(pages),
        "pages": pages,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    path = run()
    print(f"dots.mocr payload: {path}", flush=True)


if __name__ == "__main__":
    main()
