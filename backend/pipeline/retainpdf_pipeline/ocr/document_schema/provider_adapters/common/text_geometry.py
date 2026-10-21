from __future__ import annotations

import math
import re
from typing import Callable

APPROX_TEXT_CHAR_WIDTH_PT = 5.2
MIN_PSEUDO_LINE_PITCH_PT = 11.0
TARGET_PSEUDO_LINE_PITCH_PT = 12.0
MIN_PSEUDO_WRAP_HEIGHT_PT = MIN_PSEUDO_LINE_PITCH_PT * 2.2


def _bbox_width(bbox: list[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) if len(bbox) == 4 else 0.0


def _bbox_height(bbox: list[float]) -> float:
    return max(0.0, float(bbox[3]) - float(bbox[1])) if len(bbox) == 4 else 0.0


def _compact_text_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def _estimated_chars_per_line(width_pt: float) -> int:
    return max(12, int(width_pt / APPROX_TEXT_CHAR_WIDTH_PT))


def _split_words_evenly(text: str, line_count: int, chars_per_line: int) -> list[str]:
    words = (text or "").split()
    if not words:
        compact = " ".join((text or "").split())
        return [compact] if compact else []

    total_compact_len = _compact_text_len(text)
    target_chars_per_line = max(10, math.ceil(total_compact_len / max(1, line_count)))
    break_threshold = min(chars_per_line, max(target_chars_per_line, int(chars_per_line * 0.58)))

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    remaining_words = len(words)
    remaining_lines = max(1, line_count)

    for word in words:
        remaining_words -= 1
        projected = current_len + (1 if current else 0) + len(word)
        force_break = (
            current
            and current_len >= break_threshold
            and remaining_lines > 1
            and remaining_words >= remaining_lines - 1
        )
        if force_break:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
            remaining_lines -= 1
            continue
        current.append(word)
        current_len = projected

    if current:
        chunks.append(" ".join(current))
    return [chunk for chunk in chunks if chunk.strip()]


def estimate_wrapped_line_count(*, bbox: list[float], text: str) -> int:
    """The original Paddle-adapter estimate; not provider-detected line geometry."""
    width_pt = _bbox_width(bbox)
    height_pt = _bbox_height(bbox)
    text_len = _compact_text_len(text)
    if width_pt <= 0 or height_pt <= 0 or text_len < 72:
        return 0
    if height_pt < MIN_PSEUDO_WRAP_HEIGHT_PT:
        return 0

    chars_per_line = _estimated_chars_per_line(width_pt)
    predicted_by_width = max(2, math.ceil(text_len / max(1, chars_per_line)))
    max_lines_by_height = max(1, int(height_pt / MIN_PSEUDO_LINE_PITCH_PT))
    desired_by_height = max(2, round(height_pt / TARGET_PSEUDO_LINE_PITCH_PT))
    return min(max_lines_by_height, max(predicted_by_width, desired_by_height))


def build_explicit_text_lines(
    *,
    bbox: list[float],
    text: str,
    raw_label: str,
    segment_builder: Callable[[str, str], list[dict]],
) -> list[dict]:
    """Reuse Paddle's explicit-newline geometry without interpreting the text."""
    chunks = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if len(chunks) <= 1 or len(bbox) != 4:
        return []
    x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    total_lines = len(chunks)
    line_height = max(1.0, (y1 - y0) / total_lines)
    lines: list[dict] = []
    for index, chunk in enumerate(chunks):
        line_y0 = y0 + line_height * index
        line_y1 = y1 if index == total_lines - 1 else y0 + line_height * (index + 1)
        lines.append(
            {
                "bbox": [x0, round(line_y0, 3), x1, round(line_y1, 3)],
                "bbox_precision": "synthetic_newline",
                "spans": segment_builder(chunk, raw_label),
            }
        )
    return lines


def build_wrapped_text_lines(
    *,
    bbox: list[float],
    text: str,
    raw_label: str,
    segment_builder: Callable[[str, str], list[dict]],
) -> list[dict]:
    """Estimate visual rows without changing the source text or its block bbox.

    Provider adapters retain control of semantic newlines and segment parsing.
    Short/invalid blocks and text that cannot be split keep the existing fallback.
    """
    line_count = estimate_wrapped_line_count(bbox=bbox, text=text)
    if line_count <= 1:
        return []

    width_pt = _bbox_width(bbox)
    height_pt = _bbox_height(bbox)
    chars_per_line = _estimated_chars_per_line(width_pt)
    chunks = _split_words_evenly(text, line_count, chars_per_line)
    if len(chunks) <= 1:
        return []

    x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    total_lines = len(chunks)
    line_height = height_pt / total_lines
    lines: list[dict] = []
    for index, chunk in enumerate(chunks):
        line_y0 = y0 + line_height * index
        line_y1 = y1 if index == total_lines - 1 else y0 + line_height * (index + 1)
        lines.append(
            {
                "bbox": [x0, round(line_y0, 3), x1, round(line_y1, 3)],
                "bbox_precision": "synthetic_wrap",
                "spans": segment_builder(chunk, raw_label),
            }
        )
    return lines


def inherit_missing_segment_bboxes(*, bbox: list[float], segments: list[dict], lines: list[dict]) -> None:
    """Attach the narrowest truthful containing region when glyph boxes are absent.

    `bbox_precision` makes the approximation explicit: top-level segments inherit
    their block, while line spans inherit their generated/observed line region.
    """
    for segment in segments:
        if segment.get("bbox") in (None, [], [0, 0, 0, 0]) or segment.get("bbox_precision") in {
            "block",
            "line",
        }:
            segment["bbox"] = list(bbox)
            segment["bbox_precision"] = "block"
    for line in lines:
        line_bbox = list(line.get("bbox", bbox) or bbox)
        for span in line.get("spans", []) or []:
            if span.get("bbox") in (None, [], [0, 0, 0, 0]) or span.get("bbox_precision") in {
                "block",
                "line",
            }:
                span["bbox"] = line_bbox
                span["bbox_precision"] = "line"
