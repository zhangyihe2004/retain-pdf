"""Conservative right-side capacity shared by all text renderers.

Layout geometry stays unchanged. A separate ceiling may be used only by the
final measured fitter; graphics and untranslated blocks remain barriers.
"""
from __future__ import annotations

import json
from math import ceil, floor
from pathlib import Path

import fitz

from retainpdf_pipeline.render.layout.payload.collision_context import flow_height_limit
from retainpdf_pipeline.render.layout.payload.render_item import get_render_inner_bbox
from retainpdf_pipeline.render.layout.payload.block_seed_metrics import collect_page_seed_metrics
from retainpdf_pipeline.render.layout.payload.safe_width import GAP_PT, column_right
from retainpdf_pipeline.render.layout.typography.geometry import cover_bbox, inner_bbox
from retainpdf_pipeline.render.source.background.fill import LocalBackgroundSampler, sample_local_background_fill


PIXEL_SCALE = 2.0
BACKGROUND_TOLERANCE = 18


def source_boxes_by_page(source_pdf_path: Path) -> dict[int, list[dict]]:
    path = Path(source_pdf_path).parent.parent / "ocr" / "normalized" / "document.v1.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(page["page_index"]): [{"bbox": list(block["bbox"]), "kind": block.get("type", "text")} for block in page.get("blocks", []) if len(block.get("bbox", [])) == 4]
        for page in data.get("pages", [])
    }


def _blank_right(
    page: fitz.Page, rect: list[float], right: float, background: tuple[float, float, float],
    *, scan_start: float | None = None,
) -> float:
    if right <= rect[2] + 0.5:
        return rect[2]
    scan_start = max(rect[2], scan_start if scan_start is not None else rect[2])
    if scan_start >= right:
        return right
    clip = fitz.Rect(scan_start, rect[1], right, rect[3]) & page.rect
    pix = page.get_pixmap(matrix=fitz.Matrix(PIXEL_SCALE, PIXEL_SCALE), colorspace=fitz.csRGB, alpha=False, clip=clip)
    samples = pix.samples
    bg = [round(c * 255) for c in background]
    previous_dirty = False
    # Raster clips round outward. Scan only complete pixels outside the source
    # clearing edge and within its vertical band, not its own clipped glyph edge.
    start = max(0, ceil(scan_start * PIXEL_SCALE) - pix.x)
    first_y = max(0, ceil(rect[1] * PIXEL_SCALE) - pix.y)
    last_y = min(pix.height, floor(rect[3] * PIXEL_SCALE) - pix.y)
    if last_y <= first_y:
        return rect[2]
    for x in range(start, pix.width):
        dirty = 0
        for y in range(first_y, last_y):
            offset = (y * pix.width + x) * pix.n
            if max(abs(samples[offset + c] - bg[c]) for c in range(3)) > BACKGROUND_TOLERANCE:
                dirty += 1
        # Ignore isolated JPEG noise, but stop at ink, a rule or a background
        # transition. Never erase a strip merely because OCR omitted it.
        if dirty >= 2 or (dirty and previous_dirty):
            return max(rect[2], (pix.x + max(start, x - 1)) / PIXEL_SCALE - GAP_PT)
        previous_dirty = bool(dirty)
    return right


def apply_safe_text_geometry(
    items: list[dict], *, page: fitz.Page, source_boxes: list[dict],
    inner_boxes: dict[int, list[float]],
) -> None:
    boxes = [block["bbox"] for block in source_boxes] or [list(item["bbox"]) for item in items if len(item.get("bbox", [])) == 4]
    sampler = None
    for index, item in enumerate(items):
        cached = get_render_inner_bbox(item)
        rect = list(cached or inner_boxes.get(index) or item.get("bbox", []))
        source_rect = item["bbox"] if item.get("_use_raw_text_bbox") else inner_bbox(item)
        if cached and len(source_rect) == 4:
            # Old prepared items may contain the former expanded frame.
            rect[2] = min(rect[2], source_rect[2])
        text = str(item.get("render_protected_text") or item.get("protected_translated_text") or item.get("translated_text") or "")
        if len(rect) != 4 or not text or rect[2] <= rect[0] or rect[3] <= rect[1]:
            continue
        height = rect[3] - rect[1]
        in_graphic = any(
            block["kind"] != "text" and fitz.Rect(block["bbox"]).contains(fitz.Rect(item["bbox"]))
            for block in source_boxes
        )
        right = column_right(rect, boxes, float(page.rect.width)) if not in_graphic else rect[2]
        is_title = item.get("block_class") == "title" or item.get("structure_role") in {"title", "heading"}
        background = (1.0, 1.0, 1.0)
        if right > rect[2] + 0.5 or is_title:
            if sampler is None:
                sampler = LocalBackgroundSampler.build(page, [page.rect])
            background = sample_local_background_fill(page, fitz.Rect(rect), sampler=sampler)
            covers = item.get("_render_cover_rects")
            if covers is None:
                covers = [cover_bbox(item)]
            # Only skip an already-cleared strip spanning the full text band;
            # never jump over a source-preservation hole between fragments.
            scan_start = max(
                (
                    cover[2] for cover in covers
                    if cover[0] <= rect[2] <= cover[2] and cover[1] <= rect[1] and cover[3] >= rect[3]
                ),
                default=rect[2],
            )
            right = _blank_right(page, rect, right, background, scan_start=scan_start)
        # Never feed borrowed space to typography, density, line geometry or
        # collision policy. They all continue to see the original frame.
        item.pop("_render_preferred_width_pt", None)
        item["_render_inner_bbox"] = rect
        item["_render_available_width_pt"] = round(max(rect[2], right) - rect[0], 3)
        item["_render_flow_max_height_pt"] = flow_height_limit(item["_render_inner_bbox"], boxes)
        if is_title:
            top = max((r[3] + GAP_PT for r in boxes if r[3] <= rect[1] and min(rect[2], r[2]) > max(rect[0], r[0])), default=0.0)
            shift = min(height * 0.05, max(0.0, rect[1] - top))
            if shift:
                band = fitz.Rect(rect[0], rect[1] - shift, item["_render_inner_bbox"][2], rect[1]) & page.rect
                pix = page.get_pixmap(matrix=fitz.Matrix(PIXEL_SCALE, PIXEL_SCALE), colorspace=fitz.csRGB, alpha=False, clip=band)
                bg = [round(c * 255) for c in background]
                samples = pix.samples
                if any(max(abs(samples[i + c] - bg[c]) for c in range(3)) > BACKGROUND_TOLERANCE for i in range(0, len(samples), pix.n)):
                    shift = 0.0
            item["_render_title_shift_up_limit_pt"] = shift


def apply_safe_geometry_to_pages(pages: dict[int, list[dict]], source_pdf_path: Path | None) -> None:
    if source_pdf_path is None:
        return
    boxes = source_boxes_by_page(source_pdf_path)
    with fitz.open(source_pdf_path) as doc:
        for page_idx, items in pages.items():
            metrics = collect_page_seed_metrics(items, page_width=float(doc[page_idx].rect.width))
            apply_safe_text_geometry(
                items, page=doc[page_idx], source_boxes=boxes.get(page_idx, []),
                inner_boxes=metrics.effective_inner_bboxes,
            )
