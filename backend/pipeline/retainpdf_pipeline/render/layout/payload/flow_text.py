"""One routing decision for ordinary text, shared by seed, fit and output."""
from __future__ import annotations

import re

from retainpdf_pipeline.render.layout.font_roles import is_title_like_block
from retainpdf_pipeline.render.layout.inline_content.mode_router import build_item_render_markdown
from retainpdf_pipeline.render.layout.markdown_lists import contains_unordered_markdown_list, is_list_item_block_type
from retainpdf_pipeline.render.layout.payload.toc_structure import render_toc_entries_for_item
from retainpdf_pipeline.render.semantics.legacy_compat import legacy_raw_block_type


BLOCK_MARKDOWN_RE = re.compile(r"(?m)^[ \t]*(?:\d+[.)]\s|#{1,6}\s|>|```|~~~|\||!\[|<table\b)")


def is_flow_text(
    item: dict, text: str, formula_map: list, render_kind: str, *, preserve_lines: bool = False,
    markdown_text: str | None = None, toc_entries: list | None = None,
) -> bool:
    if (
        is_title_like_block(item) or preserve_lines or formula_map
        or is_list_item_block_type(legacy_raw_block_type(item))
        or render_kind not in {"markdown", "plain", "plain_line"}
        or (render_kind == "plain_line" and not item.get("_force_plain_line"))
        or (render_toc_entries_for_item(item, text) if toc_entries is None else toc_entries)
    ):
        return False
    markdown = build_item_render_markdown(item, text, formula_map) if markdown_text is None else markdown_text
    return not contains_unordered_markdown_list(markdown) and not BLOCK_MARKDOWN_RE.search(markdown) and "$" not in markdown


def is_flow_payload(payload: dict, *, markdown_text: str | None = None, toc_entries: list | None = None) -> bool:
    if "fit_flow_text" in payload:
        return bool(payload["fit_flow_text"])
    return is_flow_text(
        payload["item"], payload["translated_text"], payload["formula_map"], payload["render_kind"],
        preserve_lines=bool(payload.get("preserve_line_breaks")),
        markdown_text=markdown_text, toc_entries=toc_entries,
    )
