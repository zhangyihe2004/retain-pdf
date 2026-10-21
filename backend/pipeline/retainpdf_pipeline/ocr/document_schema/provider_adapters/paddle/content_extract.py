from __future__ import annotations

import re

from retainpdf_pipeline.ocr.document_schema.provider_adapters.common import (
    build_line_records,
    build_text_segments,
)
from retainpdf_pipeline.ocr.document_schema.provider_adapters.common.text_geometry import (
    TARGET_PSEUDO_LINE_PITCH_PT,
    build_explicit_text_lines,
    build_wrapped_text_lines,
    estimate_wrapped_line_count as _pseudo_line_count,
    inherit_missing_segment_bboxes,
)
from retainpdf_pipeline.ocr.document_schema.protected_formula_tokens import (
    PROTECTED_TOKEN_RE,
    protect_inline_formulas,
)

PSEUDO_TEXT_HEIGHT_SLACK_RATIO = 1.08
BODYLIKE_SUBTYPES = {"body", "heading"}
DOLLAR_MATH_RE = re.compile(
    r"(?<!\\)(?:"
    r"\$\$(?P<display>.+?)(?<!\\)\$\$"
    r"|\$(?!\$)(?P<inline>[^$\n]+?)(?<!\\)\$(?!\$)"
    r")",
    flags=re.DOTALL,
)


def _segment_record(*, text: str, raw_label: str, segment_type: str) -> dict:
    return {
        "type": segment_type,
        "raw_type": raw_label,
        "text": text,
        "bbox": [0, 0, 0, 0],
        "score": None,
    }


def _split_text_with_inline_formulas(text: str, raw_label: str) -> list[dict]:
    dollar_matches = list(DOLLAR_MATH_RE.finditer(text))
    if dollar_matches:
        segments: list[dict] = []
        cursor = 0
        for match in dollar_matches:
            if match.start() > cursor:
                chunk = text[cursor : match.start()]
                if chunk.strip():
                    segments.append(
                        _segment_record(
                            text=chunk.strip(),
                            raw_label=raw_label,
                            segment_type="text",
                        )
                    )
            formula_text = str(match.group("display") or match.group("inline") or "").strip()
            if formula_text:
                segments.append(
                    _segment_record(
                        text=formula_text,
                        raw_label=raw_label,
                        segment_type="inline_formula",
                    )
                )
            cursor = match.end()
        tail = text[cursor:]
        if tail.strip():
            segments.append(
                _segment_record(
                    text=tail.strip(),
                    raw_label=raw_label,
                    segment_type="text",
                )
            )
        if segments:
            return segments

    protected_text, formula_map = protect_inline_formulas(text)
    if not formula_map:
        return build_text_segments(text, raw_type=raw_label, segment_type="text")

    lookup = {entry["placeholder"]: entry["formula_text"] for entry in formula_map}
    segments: list[dict] = []
    cursor = 0
    for match in PROTECTED_TOKEN_RE.finditer(protected_text):
        start, end = match.span()
        if start > cursor:
            chunk = protected_text[cursor:start]
            if chunk.strip():
                segments.append(_segment_record(text=chunk.strip(), raw_label=raw_label, segment_type="text"))
        placeholder = match.group(0)
        formula_text = lookup.get(placeholder, "").strip()
        if formula_text:
            segments.append(
                _segment_record(
                    text=formula_text,
                    raw_label=raw_label,
                    segment_type="inline_formula",
                )
            )
        cursor = end
    tail = protected_text[cursor:]
    if tail.strip():
        segments.append(_segment_record(text=tail.strip(), raw_label=raw_label, segment_type="text"))
    return segments or build_text_segments(text, raw_type=raw_label, segment_type="text")


def build_segments(text: str, raw_label: str) -> list[dict]:
    label = raw_label.strip().lower()
    if label in {"display_formula", "formula"}:
        return build_text_segments(text, raw_type=raw_label, segment_type="formula")
    # Paddle uses semantic labels such as abstract/reference/figure_title for
    # textual blocks. Inline math must be detected for all of them, not only
    # blocks whose raw label happens to be the literal "text".
    return _split_text_with_inline_formulas(text, raw_label)


def tighten_text_bbox(
    *,
    bbox: list[float],
    text: str,
    block_type: str = "",
    sub_type: str = "",
) -> list[float]:
    if block_type != "text" or str(sub_type or "").strip().lower() not in BODYLIKE_SUBTYPES:
        return list(bbox)
    if len(bbox) != 4:
        return list(bbox)
    line_count = _pseudo_line_count(bbox=bbox, text=text)
    if line_count <= 1:
        return list(bbox)
    x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    original_height = max(0.0, y1 - y0)
    target_height = min(
        original_height,
        max(
            TARGET_PSEUDO_LINE_PITCH_PT * 2.0,
            line_count * TARGET_PSEUDO_LINE_PITCH_PT * PSEUDO_TEXT_HEIGHT_SLACK_RATIO,
        ),
    )
    return [x0, y0, x1, round(min(y1, y0 + target_height), 3)]


def build_lines(
    *,
    bbox: list[float],
    segments: list[dict],
    text: str = "",
    raw_label: str = "",
    block_type: str = "",
    sub_type: str = "",
) -> list[dict]:
    explicit_lines = build_explicit_text_lines(
        bbox=bbox, text=text, raw_label=raw_label, segment_builder=build_segments,
    )
    if explicit_lines:
        return explicit_lines
    if block_type == "text" and str(sub_type or "").strip().lower() in BODYLIKE_SUBTYPES:
        pseudo_bbox = tighten_text_bbox(
            bbox=bbox,
            text=text,
            block_type=block_type,
            sub_type=sub_type,
        )
        pseudo_lines = build_wrapped_text_lines(
            bbox=pseudo_bbox, text=text, raw_label=raw_label, segment_builder=build_segments,
        )
        if pseudo_lines:
            return pseudo_lines
    return build_line_records(bbox, segments)


def assign_inline_formula_bboxes(
    *,
    segments: list[dict],
    block_bbox: list[float],
    layout_box_lookup: dict[tuple[float, float, float, float], dict],
) -> dict:
    """Use Paddle layout boxes only when formula-to-box pairing is unambiguous.

    The layout detector does not expose recognized formula text, so a partial
    positional match could attach the wrong box. We therefore assign provider
    geometry only when the number of formula segments exactly equals the
    number of contained ``inline_formula`` boxes; otherwise the normal block
    or line approximation remains explicit.
    """

    formula_segments = [
        segment for segment in segments if segment.get("type") == "inline_formula"
    ]
    candidates: list[tuple[list[float], dict]] = []
    if len(block_bbox) == 4:
        x0, y0, x1, y1 = (float(value) for value in block_bbox)
        for raw_box in layout_box_lookup.values():
            if str(raw_box.get("label", "") or "").strip().lower() != "inline_formula":
                continue
            coordinate = raw_box.get("coordinate")
            if not isinstance(coordinate, list) or len(coordinate) != 4:
                continue
            candidate = [float(value or 0) for value in coordinate]
            if candidate[2] <= candidate[0] or candidate[3] <= candidate[1]:
                continue
            if (
                candidate[0] >= x0
                and candidate[1] >= y0
                and candidate[2] <= x1
                and candidate[3] <= y1
            ):
                candidates.append((candidate, raw_box))
    candidates.sort(key=lambda item: (item[0][1], item[0][0], item[0][3], item[0][2]))

    matched_count = 0
    if formula_segments and len(formula_segments) == len(candidates):
        for segment, (candidate, raw_box) in zip(formula_segments, candidates):
            segment["bbox"] = candidate
            segment["bbox_precision"] = "provider_layout"
            score = raw_box.get("score")
            if isinstance(score, (int, float)):
                segment["score"] = float(score)
            matched_count += 1
    return {
        "provider_inline_formula_segment_count": len(formula_segments),
        "provider_inline_formula_candidate_count": len(candidates),
        "provider_inline_formula_bbox_count": matched_count,
        "provider_inline_formula_bbox_complete": bool(formula_segments)
        and matched_count == len(formula_segments),
    }
