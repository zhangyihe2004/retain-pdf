from __future__ import annotations

from retainpdf_pipeline.render.layout.markdown_lists import contains_unordered_markdown_list
from retainpdf_pipeline.render.layout.markdown_lists import is_list_item_block_type
from retainpdf_pipeline.render.layout.model.models import RenderBlock
from retainpdf_pipeline.render.layout.inline_content.core.markdown import build_plain_text_from_text
from retainpdf_pipeline.render.layout.inline_content.mode_router import build_item_render_markdown
from retainpdf_pipeline.render.layout.payload.formula_safety import formula_safe_inner_bbox
from retainpdf_pipeline.render.layout.payload.flow_text import is_flow_payload
from retainpdf_pipeline.render.layout.payload.metrics import resolve_typst_binary_fit
from retainpdf_pipeline.render.layout.payload.metrics import estimated_required_lines
from retainpdf_pipeline.render.layout.payload.fit_common import FLOW_TEXT_MIN_FONT_SIZE_PT
from retainpdf_pipeline.render.layout.payload.fit_typst_bounds import resolve_min_leading_em
from retainpdf_pipeline.render.layout.payload.fit_typst_context import build_typst_fit_context
from retainpdf_pipeline.render.layout.payload.line_structure import preserved_line_boxes_for_item
from retainpdf_pipeline.render.layout.payload.toc_structure import render_toc_entries_for_item
from retainpdf_pipeline.render.policy import item_uses_white_overlay_fill
from retainpdf_pipeline.render.semantics.legacy_compat import legacy_raw_block_type


def payload_to_render_block(payload: dict) -> RenderBlock:
    safe_inner_bbox, _formula_insets = formula_safe_inner_bbox(
        payload["inner_bbox"],
        payload["translated_text"],
        payload["formula_map"],
        font_size_pt=float(payload["font_size_pt"] or 0.0),
    )
    title_fit = payload.get("title_fit")
    preserve_line_breaks = bool(payload.get("preserve_line_breaks"))
    preserved_line_boxes = preserved_line_boxes_for_item(payload["item"], payload["translated_text"])
    toc_entries = payload.get("toc_entries")
    if toc_entries is None:
        toc_entries = render_toc_entries_for_item(payload["item"], payload["translated_text"])
    markdown_text = payload.get("markdown_text")
    if markdown_text is None:
        markdown_text = build_item_render_markdown(payload["item"], payload["translated_text"], payload["formula_map"])
    is_list_item = is_list_item_block_type(legacy_raw_block_type(payload["item"]))
    has_markdown_list = contains_unordered_markdown_list(markdown_text)
    fit_flow_text = title_fit is None and is_flow_payload(payload, markdown_text=markdown_text, toc_entries=toc_entries)
    # Specialized renderers keep their existing source-line exemption and fit
    # bounds. Ordinary flow never uses it, regardless of font inheritance.
    allow_legacy_overflow = bool(not fit_flow_text and payload.get("_allow_short_text_bbox_overflow"))
    restrict_legacy_fit = False
    if allow_legacy_overflow and not is_list_item:
        source_line_count = sum(
            1 for line in payload["item"].get("lines", [])
            if len(box := line.get("bbox", [])) == 4 and box[2] > box[0] and box[3] > box[1]
        )
        allow_legacy_overflow = (
            not payload["adjacent_collision_risk"] and source_line_count > 0
            and safe_inner_bbox[3] - safe_inner_bbox[1] >= payload["font_size_pt"]
            and estimated_required_lines(
                safe_inner_bbox, payload["translated_text"], payload["formula_map"], payload["font_size_pt"],
            ) <= source_line_count
        )
        restrict_legacy_fit = not allow_legacy_overflow
    if title_fit is not None:
        fit_to_box = title_fit.fit_to_box
        fit_single_line = title_fit.fit_single_line
        fit_min_font_size_pt = title_fit.fit_min_font_size_pt
        fit_max_font_size_pt = title_fit.fit_max_font_size_pt
        fit_min_leading_em = title_fit.fit_min_leading_em
        fit_max_height_pt = title_fit.fit_max_height_pt
        fit_target_width_pt = title_fit.fit_target_width_pt
        fit_target_height_pt = title_fit.fit_target_height_pt
    else:
        fit_item = {
            **payload["item"],
            "_render_inner_bbox": safe_inner_bbox,
            "_is_body_text_candidate": payload["is_body"],
            "_dense_small_box": payload["dense_small_box"],
            "_heavy_dense_small_box": payload["heavy_dense_small_box"],
            "_short_body_inherited_font_floor_pt": 0.0 if fit_flow_text or restrict_legacy_fit else payload.get("_short_body_inherited_font_floor_pt", 0.0),
            "_relaxed_fit_height_pt": 0.0 if fit_flow_text or restrict_legacy_fit else payload.get("_relaxed_fit_height_pt", 0.0),
        }
        if fit_flow_text:
            context = build_typst_fit_context(
                fit_item, payload["translated_text"], payload["formula_map"],
                payload["font_size_pt"], payload["leading_em"],
                adjacent_collision_risk=False, adjacent_available_height_pt=None,
            )
            fit_min_leading_em = round(resolve_min_leading_em(context, leading_em=payload["leading_em"]), 2) if context is not None else 0.0
            # Typography supplies preferences, not extra capacity. Preserve
            # smaller known limits without subtracting safety padding again.
            fit_to_box = True
            fit_min_font_size_pt = 0.0  # The shared flow fitter owns its font floor.
            fit_max_height_pt = safe_inner_bbox[3] - safe_inner_bbox[1]
            if payload["adjacent_available_height_pt"] is not None:
                fit_max_height_pt = min(fit_max_height_pt, payload["adjacent_available_height_pt"])
        else:
            fit_to_box, fit_min_font_size_pt, fit_min_leading_em, fit_max_height_pt = resolve_typst_binary_fit(
                fit_item, payload["translated_text"], payload["formula_map"],
                payload["font_size_pt"], payload["leading_em"],
                page_body_font_size_pt=payload["page_body_font_size_pt"],
                prefer_typst_fit=payload["prefer_typst_fit"],
                adjacent_collision_risk=payload["adjacent_collision_risk"],
                adjacent_available_height_pt=payload["adjacent_available_height_pt"],
            )
            if restrict_legacy_fit and fit_to_box:
                fit_min_font_size_pt = min(fit_min_font_size_pt, FLOW_TEXT_MIN_FONT_SIZE_PT)
            elif allow_legacy_overflow:
                fit_to_box = False
                fit_min_font_size_pt = 0.0
                fit_min_leading_em = 0.0
                fit_max_height_pt = 0.0
        fit_single_line = False
        fit_max_font_size_pt = 0.0
        fit_target_width_pt = 0.0
        fit_target_height_pt = 0.0
    flowing_list_item = (
        is_list_item
        and payload["render_kind"] == "markdown"
        and not preserve_line_breaks
        and has_markdown_list
    )
    if flowing_list_item:
        fit_to_box = True
        fit_min_font_size_pt = max(1.0, float(payload["font_size_pt"] or 0.0) * 0.72)
        fit_min_leading_em = max(0.1, min(float(payload["leading_em"] or 0.54), 0.54))
    if (
        restrict_legacy_fit and fit_to_box and not preserve_line_breaks
        and not payload["formula_map"] and "$" not in payload["translated_text"]
        and "\n" not in payload["translated_text"] and not has_markdown_list
        and 0 < safe_inner_bbox[3] - safe_inner_bbox[1] <= payload["font_size_pt"]
    ):
        fit_max_font_size_pt = safe_inner_bbox[3] - safe_inner_bbox[1]
    # Transfer capacity only after the original typography/policy stages.
    # Reuse the existing fitter target instead of changing the shared bbox.
    preferred_width_pt = max(0.0, payload["inner_bbox"][2] - payload["inner_bbox"][0])
    available_width_pt = float(payload["item"].get("_render_available_width_pt") or 0.0)
    if available_width_pt > preferred_width_pt:
        fit_target_width_pt = max(fit_target_width_pt, available_width_pt)
    return RenderBlock(
        block_id=f"item-{payload['index']}",
        bbox=payload["bbox"],
        cover_bbox=payload["cover_bbox"],
        cover_rects=payload["item"].get("_render_cover_rects"),
        inner_bbox=payload["inner_bbox"],
        markdown_text=markdown_text,
        plain_text=build_plain_text_from_text(payload["translated_text"]),
        render_kind=payload["render_kind"],
        font_size_pt=payload["font_size_pt"],
        leading_em=payload["leading_em"],
        first_line_indent_pt=payload.get("first_line_indent_pt", 0.0),
        justify_text=bool(payload.get("is_body") and payload["render_kind"] == "markdown"),
        font_weight=payload.get("font_weight", "regular"),
        fit_to_box=bool(fit_to_box and (fit_flow_text or payload["render_kind"] == "markdown") and not preserve_line_breaks),
        fit_flow_text=bool(fit_flow_text),
        fit_single_line=fit_single_line,
        fit_min_font_size_pt=fit_min_font_size_pt,
        fit_max_font_size_pt=fit_max_font_size_pt,
        fit_min_leading_em=fit_min_leading_em,
        fit_max_height_pt=fit_max_height_pt,
        fit_target_width_pt=fit_target_width_pt,
        fit_target_height_pt=fit_target_height_pt,
        fit_shift_up_pt=max(0.0, float(payload["item"].get("_render_title_shift_up_limit_pt", 0.0))),
        text_color=tuple(payload.get("text_color", (0, 0, 0))),
        cover_fill=tuple(payload.get("cover_fill", (1, 1, 1))),
        use_cover_fill=item_uses_white_overlay_fill(payload["item"]),
        math_map=list(payload["formula_map"]),
        skip_reason="adjacent_collision_risk" if payload.get("adjacent_collision_risk") else "",
        source_item_id=str(payload["item"].get("item_id") or ""),
        preserve_line_breaks=preserve_line_breaks,
        preserved_line_boxes=preserved_line_boxes or None,
        toc_entries=toc_entries or None,
    )


def emit_render_blocks(block_payloads: list[dict]) -> list[RenderBlock]:
    return [payload_to_render_block(payload) for payload in sorted(block_payloads, key=lambda payload: payload["index"])]
