"""Conservative style cohorts for repeated short-field rows, not tables/reflow."""
from __future__ import annotations

from statistics import median

from retainpdf_pipeline.foundation.config import layout
from retainpdf_pipeline.render.layout.font_roles import is_title_like_block
from retainpdf_pipeline.render.layout.payload.body_common import SHORT_BODY_INHERIT_MAX_HEIGHT_PT
from retainpdf_pipeline.render.layout.payload.flow_text import is_flow_payload
from retainpdf_pipeline.render.layout.payload.safe_width import ALIGN_PT
from retainpdf_pipeline.render.layout.typography.measurement import LINE_HEIGHT_TO_FONT_SCALE, local_font_metric, source_visual_line_count
from retainpdf_pipeline.render.semantics.item_view import block_kind, is_caption_like_block, is_footnote_like_block


def _short_field(payload: dict) -> bool:
    item = payload["item"]
    rect = payload["bbox"]  # Source geometry, never the expanded/fitted frame.
    source = str(item.get("source_text") or item.get("protected_source_text") or "")
    return (
        0 < rect[3] - rect[1] <= SHORT_BODY_INHERIT_MAX_HEIGHT_PT
        and source_visual_line_count(item) == 1 and "\n" not in source
        and "@" not in source and "://" not in source
    )


def _apply_cohort(rows: list[list[dict]], payloads: list[dict], row_tolerance: float) -> None:
    if len(rows) < 3:
        return
    columns: list[list[dict]] = []
    for payload in (p for row in rows for p in row):
        column = next((c for c in columns if abs(median(p["bbox"][0] for p in c) - payload["bbox"][0]) <= ALIGN_PT), None)
        if column is None:
            columns.append([payload])
        else:
            column.append(payload)
    # At least three stable columns repeated in at least three aligned rows.
    stable = [c for c in columns if len(c) >= 3]
    if len(stable) < 3:
        return
    members = [p for c in stable for p in c]
    member_ids = {id(p) for p in members}
    if sum(sum(id(p) in member_ids for p in row) >= 3 for row in rows) < 3:
        return
    # Do not merge distinct source styles just because their cells align.
    heights = [p["bbox"][3] - p["bbox"][1] for p in members]
    if max(heights) > min(heights) * 1.25:
        return
    starts = [median(p["bbox"][0] for p in c) for c in stable]
    region_bottom = max(p["bbox"][3] for p in members)
    cells = [p for p in payloads if p["bbox"][3] <= region_bottom
             and any(abs(p["bbox"][0] - x) <= ALIGN_PT for x in starts)
             and any(abs(p["bbox"][1] - row[0]["bbox"][1]) <= row_tolerance for row in rows)
             and not any(p["bbox"][0] + ALIGN_PT < x < p["bbox"][2] for x in starts)]
    for weight in {p["font_weight"] for p in members}:
        peers = [p for p in members if p["font_weight"] == weight]
        if len(peers) < 3:
            continue
        known = [local_font_metric(p["item"]) for p in peers]
        known = [font for font in known if font > 0]
        source_font = median(known) if len(known) >= len(peers) / 2 else median(
            p["bbox"][3] - p["bbox"][1] for p in peers
        ) * LINE_HEIGHT_TO_FONT_SCALE
        preferred = round(source_font * layout.BODY_FONT_SIZE_FACTOR, 2)
        # Once the grid is established by short cells, aligned multiline cells
        # share its preference too. Their content is still fitted independently;
        # neither their taller frame nor a tiny width sets the cohort's font.
        for payload in cells:
            if payload["font_weight"] == weight:
                payload["font_size_pt"] = preferred


def apply_field_style_preferences(payloads: list[dict]) -> None:
    """Share source-derived preferences; real capacity remains the fitter's job."""
    fields = [p for p in payloads if is_flow_payload(p) and block_kind(p["item"]) == "text"
              and not is_caption_like_block(p["item"]) and not is_footnote_like_block(p["item"])]
    candidates = sorted((p for p in fields if _short_field(p)), key=lambda p: (p["bbox"][1], p["bbox"][0]))
    if len(candidates) < 9:
        return
    row_tolerance = min(ALIGN_PT, median(p["bbox"][3] - p["bbox"][1] for p in candidates) * 0.16)
    rows: list[list[dict]] = []
    for payload in candidates:
        if rows and payload["bbox"][1] - rows[-1][0]["bbox"][1] <= row_tolerance:
            rows[-1].append(payload)
        else:
            rows.append([payload])
    rows = [sorted(row, key=lambda p: p["bbox"][0]) for row in rows]
    rows = [row for row in rows if all(a["bbox"][2] <= b["bbox"][0] for a, b in zip(row, row[1:]))]
    cohort: list[list[dict]] = []
    for row in rows:
        if cohort:
            previous = cohort[-1]
            y0, y1 = previous[0]["bbox"][1], row[0]["bbox"][1]
            left, right = min(p["bbox"][0] for p in row), max(p["bbox"][2] for p in row)
            separator = any(
                is_title_like_block(p["item"]) and y0 < p["bbox"][1] < y1
                and min(right, p["bbox"][2]) > max(left, p["bbox"][0]) for p in payloads
            )
            if separator or y1 - y0 > 3 * median(p["bbox"][3] - p["bbox"][1] for p in previous + row):
                _apply_cohort(cohort, fields, row_tolerance)
                cohort = []
        cohort.append(row)
    _apply_cohort(cohort, fields, row_tolerance)
