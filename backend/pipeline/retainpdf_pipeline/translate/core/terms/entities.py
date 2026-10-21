"""Title-only entity selection and occurrence-scoped terminology self-mappings."""

from __future__ import annotations

import re
from collections.abc import Iterable

from retainpdf_pipeline.translate.core.canonical_semantics import from_normalized_block, uses_title_style
from retainpdf_pipeline.translate.core.ocr.normalized_reader import (
    block_children, block_kind, block_text, get_pages, iter_page_blocks,
)

from .glossary import GlossaryEntry, TERM_WORD_CHARS, context_matches, term_pattern


def document_entity_texts(data: dict) -> list[str]:
    """Collect canonical titles once, without widening the NER input scope."""
    texts: dict[str, None] = {}

    def visit(block: dict) -> None:
        if block_kind(block) == "text":
            text = block_text(block).strip()
            if text and uses_title_style(from_normalized_block(block)):
                texts.setdefault(text, None)
        for child in block_children(data, block):
            visit(child)

    for page in get_pages(data):
        for block in iter_page_blocks(data, page):
            visit(block)
    return list(texts)


def matched_entity_entries(
    names: tuple[str, ...],
    texts: Iterable[str],
    manual_entries: list[GlossaryEntry],
) -> list[GlossaryEntry]:
    if not names:
        return []
    patterns = [
        re.compile(
            rf"(?<![{TERM_WORD_CHARS}])"
            + r"\s+".join(re.escape(word) for word in name.split())
            + rf"(?![{TERM_WORD_CHARS}])",
            re.IGNORECASE,
        )
        for name in names
    ]
    selected: dict[str, GlossaryEntry] = {}
    for text in texts:
        # An explicit manual mapping wins, including regex/contextual terms.
        occupied = [
            match.span()
            for entry in manual_entries
            for match in term_pattern(entry).finditer(text)
            if context_matches(text, entry, start=match.start(), end=match.end())
        ]
        matches = [match for pattern in patterns for match in pattern.finditer(text)]
        for match in sorted(matches, key=lambda value: (-(value.end() - value.start()), value.start())):
            start, end = match.span()
            if any(start < other_end and other_start < end for other_start, other_end in occupied):
                continue
            occupied.append((start, end))
            # Use the spelling/case in this segment, not another page's spelling.
            name = " ".join(match.group().split())
            selected.setdefault(name, GlossaryEntry(source=name, target=name, level="preferred"))
    return sorted(selected.values(), key=lambda entry: (-len(entry.source), entry.source))
