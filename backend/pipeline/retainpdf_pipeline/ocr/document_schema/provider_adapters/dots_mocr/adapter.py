from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

from retainpdf_pipeline.ocr.document_schema import default_block_derived
from retainpdf_pipeline.ocr.document_schema.providers import PROVIDER_DOTS_MOCR
from retainpdf_pipeline.ocr.document_schema.provider_adapters.common import (
    build_block_record,
    build_document_record,
    build_line_records,
    build_text_segments,
)
from retainpdf_pipeline.ocr.document_schema.provider_adapters.common.text_geometry import (
    build_explicit_text_lines,
    build_wrapped_text_lines,
    inherit_missing_segment_bboxes,
)
from retainpdf_pipeline.ocr.document_schema.text_flow import (
    classify_text_flow_for_role,
)
from retainpdf_pipeline.ocr.document_schema.provider_adapters.dots_mocr.projection import (
    DOTS_MOCR_CATEGORIES,
    project_dots_mocr_block,
)


def looks_like_dots_mocr(payload: dict) -> bool:
    return (
        isinstance(payload, dict)
        and str(payload.get("provider", "") or "") == PROVIDER_DOTS_MOCR
        and isinstance(payload.get("pages"), list)
    )


def _segments_for_text(text: str, raw_category: str) -> list[dict]:
    return build_text_segments(text, raw_type=raw_category)


def _bbox(value: object) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return [0.0, 0.0, 0.0, 0.0]


_HEADING_MARKER_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+")
_LIST_MARKER_RE = re.compile(r"^\s*(?:\*|-|•|‣|◦)\s+(?P<text>.*)$")


def _normalize_list_item_text(text: str) -> tuple[str, bool]:
    raw_lines = str(text or "").splitlines()
    if not any(_LIST_MARKER_RE.match(line) for line in raw_lines):
        return " ".join(" ".join(raw_lines).split()), False

    paragraphs: list[str] = []
    pending_blank = False
    for raw_line in raw_lines:
        line = " ".join(raw_line.split())
        if not line:
            pending_blank = bool(paragraphs)
            continue
        marker = _LIST_MARKER_RE.match(line)
        if marker:
            if pending_blank and paragraphs and not _LIST_MARKER_RE.match(paragraphs[-1]):
                paragraphs.append("")
            paragraphs.append(f"* {marker.group('text').strip()}")
        elif pending_blank and not raw_line.startswith(("  ", "\t")):
            paragraphs.extend(["", line])
        elif paragraphs:
            paragraphs[-1] = f"{paragraphs[-1]} {line}"
        else:
            paragraphs.append(line)
        pending_blank = False
    return "\n".join(paragraphs), True


def _build_block_spec(
    *,
    page_index: int,
    order: int,
    block: dict,
) -> dict:
    raw_category = str(block.get("category", "") or "")
    projection = project_dots_mocr_block(raw_category)
    bbox = _bbox(block.get("bbox"))
    text = str(block.get("text", "") or "")
    # Dots may label a standalone Markdown heading as Text. Align its
    # canonical role before removing the marker; keep raw provenance intact.
    if (
        raw_category.strip().lower() == "text" and len(text.splitlines()) == 1
        and not text.lstrip(" ").startswith("\t")
        and (heading := _HEADING_MARKER_RE.match(text))
        and text[heading.end():].strip()
    ):
        projection = project_dots_mocr_block("Section-header")
    if projection.layout_role in {"title", "heading"}:
        text = _HEADING_MARKER_RE.sub("", text, count=1)
    is_markdown_list = False
    if raw_category.strip().lower() == "list-item":
        text, is_markdown_list = _normalize_list_item_text(text)
    segments = _segments_for_text(text, raw_category)
    lines: list[dict] = []
    if projection.content_kind == "text":
        lines = build_wrapped_text_lines(
            bbox=bbox, text=text, raw_label=raw_category, segment_builder=_segments_for_text,
        )
        if (
            not lines and projection.layout_role in {"title", "heading"}
            and bbox[2] > bbox[0] and bbox[3] > bbox[1]
        ):
            lines = build_explicit_text_lines(
                bbox=bbox, text=text, raw_label=raw_category, segment_builder=_segments_for_text,
            )
        if (
            not lines and text.strip()
            and bbox[2] > bbox[0] and bbox[3] > bbox[1]
        ):
            # Reuse Paddle's block fallback when line estimation is unavailable.
            lines = build_line_records(bbox, segments)
        inherit_missing_segment_bboxes(bbox=bbox, segments=segments, lines=lines)
    content: dict = {"kind": projection.content_kind, "text": text}
    # Estimated visual rows are capacity evidence, not new semantic line breaks.
    if is_markdown_list or (
        lines
        and classify_text_flow_for_role(
            text=text, lines=[], semantic_role=projection.semantic_role,
            structure_role=projection.structure_role,
        ) == "flow"
    ):
        content["text_flow"] = "flow"
    if projection.content_kind == "table" and text:
        content["table_html"] = text
    if projection.content_kind == "formula" and text:
        content["formula_latex"] = text
    return {
        "block_id": f"p{page_index + 1:03d}-b{order:04d}",
        "page_index": page_index,
        "order": order,
        "reading_order": order,
        "content_kind": projection.content_kind,
        "sub_type": projection.sub_type,
        "bbox": bbox,
        "geometry": {"bbox": list(bbox)},
        "content": content,
        "text": text,
        "lines": lines,
        "segments": segments,
        "tags": list(projection.tags),
        "derived": dict(default_block_derived()),
        "layout_role": projection.layout_role,
        "semantic_role": projection.semantic_role,
        "structure_role": projection.structure_role,
        "policy": {
            "translate": projection.translate,
            "translate_reason": projection.translate_reason,
        },
        "metadata": {
            "raw_category": raw_category,
            "provider_bbox_unit": "pt",
            "provider_bbox_origin": "top_left",
            **(
                {"provider_render_bbox_px": list(block.get("render_bbox_px", block.get("raw_bbox_px", [])))}
                if isinstance(block.get("render_bbox_px", block.get("raw_bbox_px", [])), list)
                else {}
            ),
        },
        "source": {
            "provider": PROVIDER_DOTS_MOCR,
            "raw_page_index": page_index,
            "raw_type": raw_category,
            "raw_sub_type": "",
            "raw_bbox": list(bbox),
            "raw_bbox_px": list(block.get("raw_bbox_px", []) or []),
            "raw_text_excerpt": text[:200],
            "raw_unit": "pt",
            "raw_origin": "top_left",
            "raw_path": f"$.pages[{page_index}].blocks[{order}]",
        },
        "provenance": {
            "provider": PROVIDER_DOTS_MOCR,
            "raw_label": raw_category,
            "raw_sub_type": "",
            "raw_bbox": list(bbox),
            "raw_bbox_px": list(block.get("raw_bbox_px", []) or []),
            "raw_unit": "pt",
            "raw_origin": "top_left",
            "raw_path": f"$.pages[{page_index}].blocks[{order}]",
        },
    }


def build_dots_mocr_document(
    payload: dict,
    document_id: str,
    source_json_path: Path,
    provider_version: str,
) -> dict:
    pages: list[dict] = []
    category_counts: Counter[str] = Counter()
    unknown_categories: set[str] = set()
    for page_index, page in enumerate(payload.get("pages", []) or []):
        if not isinstance(page, dict):
            continue
        raw_blocks = [block for block in page.get("blocks", []) or [] if isinstance(block, dict)]
        blocks = [
            build_block_record(
                _build_block_spec(
                    page_index=page_index,
                    order=order,
                    block=block,
                )
            )
            for order, block in enumerate(raw_blocks)
        ]
        for block in blocks:
            raw_category = str(block.get("provenance", {}).get("raw_label", "") or "")
            category_counts[raw_category or "<missing>"] += 1
            if str(raw_category or "").strip().lower() not in {
                category.lower() for category in DOTS_MOCR_CATEGORIES
            }:
                unknown_categories.add(raw_category or "<missing>")
        pages.append(
            {
                "page_index": page_index,
                "width": float(page.get("width", 0) or 0),
                "height": float(page.get("height", 0) or 0),
                "unit": str(page.get("unit", "pt") or "pt"),
                "blocks": blocks,
                "metadata": {
                    "render_dpi": page.get("render_dpi"),
                    "image_width": page.get("image_width"),
                    "image_height": page.get("image_height"),
                },
            }
        )

    document = build_document_record(
        document_id=document_id,
        provider=PROVIDER_DOTS_MOCR,
        provider_version=provider_version,
        source_json_path=source_json_path,
        pages=pages,
        notes="Adapted from dots.mocr reading-order layout payload.",
    )
    document.setdefault("derived", {})
    document["derived"]["provider_signals"] = {
        "raw_category_counts": dict(sorted(category_counts.items())),
        "unknown_categories": sorted(unknown_categories),
    }
    payload_meta = {
        key: payload.get(key)
        for key in ("provider", "model", "source_pdf", "page_count")
        if key in payload
    }
    document["derived"]["provider_payload_meta"] = payload_meta
    return document


__all__ = [
    "build_dots_mocr_document",
    "looks_like_dots_mocr",
]
