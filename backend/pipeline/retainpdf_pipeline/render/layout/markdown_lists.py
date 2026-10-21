from __future__ import annotations

import re


# These are lexical checks, not a parser or permission to reflow a block.
MARKDOWN_LIST_MARKER_RE = re.compile(r"(?m)^[ \t]*(?:\*|\+|-)(?=[ \t]+\S)")
_LIST_ITEM_BLOCK_TYPES = frozenset({"list-item", "list_item"})


def contains_unordered_markdown_list(text: str) -> bool:
    return MARKDOWN_LIST_MARKER_RE.search(text or "") is not None


def is_list_item_block_type(raw_type: object) -> bool:
    return str(raw_type or "").strip().lower() in _LIST_ITEM_BLOCK_TYPES
