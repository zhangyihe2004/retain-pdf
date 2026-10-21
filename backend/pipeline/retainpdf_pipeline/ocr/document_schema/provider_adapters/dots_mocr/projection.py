from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DotsMocrProjection:
    content_kind: str
    sub_type: str
    layout_role: str = "unknown"
    semantic_role: str = "unknown"
    structure_role: str = ""
    translate: bool = False
    translate_reason: str = "provider_non_body:unknown"
    tags: tuple[str, ...] = ()


_BODY = DotsMocrProjection(
    content_kind="text",
    sub_type="body",
    layout_role="paragraph",
    semantic_role="body",
    structure_role="body",
    translate=True,
    translate_reason="provider_body_whitelist:text",
)
_HEADING = DotsMocrProjection(
    content_kind="text",
    sub_type="heading",
    layout_role="heading",
    semantic_role="unknown",
    structure_role="heading",
    translate=True,
    translate_reason="provider_heading_candidate",
)
_TITLE = DotsMocrProjection(
    content_kind="text",
    sub_type="title",
    layout_role="title",
    semantic_role="unknown",
    structure_role="document_title",
    translate=True,
    translate_reason="provider_title_candidate",
)
_LIST = DotsMocrProjection(
    content_kind="text",
    sub_type="list_item",
    layout_role="list_item",
    semantic_role="body",
    structure_role="body",
    translate=True,
    translate_reason="provider_body_whitelist:list",
    tags=("list",),
)
_CAPTION = DotsMocrProjection(
    content_kind="text",
    sub_type="caption",
    layout_role="caption",
    semantic_role="unknown",
    structure_role="caption",
    translate=True,
    translate_reason="provider_caption_whitelist:caption",
    tags=("caption",),
)
_PAGE_HEADER = DotsMocrProjection(
    content_kind="text",
    sub_type="metadata",
    layout_role="header",
    semantic_role="metadata",
    structure_role="metadata",
    translate=False,
    translate_reason="provider_non_body:page_header",
    tags=("metadata", "skip_translation"),
)
_PAGE_FOOTER = DotsMocrProjection(
    content_kind="text",
    sub_type="metadata",
    layout_role="footer",
    semantic_role="metadata",
    structure_role="metadata",
    translate=False,
    translate_reason="provider_non_body:page_footer",
    tags=("metadata", "skip_translation"),
)
_FOOTNOTE = DotsMocrProjection(
    content_kind="text",
    sub_type="footnote",
    layout_role="footnote",
    semantic_role="metadata",
    structure_role="footnote",
    translate=False,
    translate_reason="provider_non_body:footnote",
    tags=("footnote", "skip_translation"),
)
_FORMULA = DotsMocrProjection(
    content_kind="formula",
    sub_type="display_formula",
    translate=False,
    translate_reason="provider_non_text:formula",
    tags=("formula",),
)
_TABLE = DotsMocrProjection(
    content_kind="table",
    sub_type="table_body",
    translate=False,
    translate_reason="provider_non_text:table",
    tags=("table", "skip_translation"),
)
_IMAGE = DotsMocrProjection(
    content_kind="image",
    sub_type="figure",
    translate=False,
    translate_reason="provider_non_text:image",
    tags=("image", "skip_translation"),
)
def project_dots_mocr_block(category: str) -> DotsMocrProjection:
    label = str(category or "").strip().lower()
    if label == "text":
        return _BODY
    if label == "section-header":
        return _HEADING
    if label == "title":
        return _TITLE
    if label == "list-item":
        return _LIST
    if label == "caption":
        return _CAPTION
    if label == "page-header":
        return _PAGE_HEADER
    if label == "page-footer":
        return _PAGE_FOOTER
    if label == "footnote":
        return _FOOTNOTE
    if label == "formula":
        return _FORMULA
    if label == "table":
        return _TABLE
    if label == "picture":
        return _IMAGE
    return DotsMocrProjection(
        content_kind="unknown",
        sub_type=label or "unknown",
        translate=False,
        translate_reason=f"provider_unknown:{label or 'missing'}",
        tags=("unknown",),
    )


DOTS_MOCR_CATEGORIES = (
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
)


__all__ = [
    "DOTS_MOCR_CATEGORIES",
    "DotsMocrProjection",
    "project_dots_mocr_block",
]
