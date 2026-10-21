from __future__ import annotations

import json
from typing import Any

from retainpdf_pipeline.services.pipeline_shared.direct_typst_math import find_mitex_rewrites
from retainpdf_pipeline.services.pipeline_shared.direct_typst_math import has_balanced_unescaped_dollars
from retainpdf_pipeline.translate.core.context import TranslationItemContext
from retainpdf_pipeline.translate.prompt_loader import load_prompt

from retainpdf_pipeline.translate.llm.shared.prompt_protocols import (
    DEFAULT_TARGET_LANGUAGE_NAME,
    SOURCE_TERMINAL_RE,
    _output_protocol,
    _target_language_name,
    direct_math_guidance,
)


HY_MT_RULES_HEADING = "# 严格约束"
HY_MT_BACKGROUND_HEADING = "# 背景信息"

CONTINUATION_GROUP_NOTE = "这是跨栏或跨页续接正文的一部分，请结合上下文理解后直接输出这一整段的译文。"
MATH_DELIMITER_DAMAGE_HINT = (
    "本段原文的数学定界符 `$` 数量为奇数，说明 OCR 丢失了配对的 `$`。"
    "请按语义判断公式的真实边界，在译文中修复补全，确保每个公式的 `$...$` 成对闭合。"
)


def hy_mt_input_heading(target_language_name: str = DEFAULT_TARGET_LANGUAGE_NAME) -> str:
    """Input heading based on the official background-context instruction."""
    return (
        f"# 请结合背景信息将以下文本翻译为{_target_language_name(target_language_name)}，"
        f"注意只需要输出翻译后的结果，不要额外解释："
    )


def _source_looks_incomplete(text: str) -> bool:
    source = str(text or "").strip()
    return bool(source) and SOURCE_TERMINAL_RE.search(source) is None


def _numbered_rule(number: int, title: str, body: str) -> str:
    lines = str(body or "").strip().splitlines()
    if not lines:
        return f"{number}. {title}："
    return "\n".join([
        f"{number}. {title}：{lines[0].strip()}",
        *(f"   {line}" for line in lines[1:]),
    ])


def system_prompt(
    kind: str,
    *,
    domain_guidance: str,
    mode: str,
    response_style: str,
    target_language_name: str,
    direct_typst: bool,
    structured_decision: bool = False,
) -> str:
    """Build the complete stable Hy-MT system prompt from official-style blocks."""
    protocol = _output_protocol(kind, response_style, sci_decision=mode == "sci" and structured_decision)
    # Hy-MT is a small MT-specialized model. Keep its stable prompt free of
    # per-document rule profiles.
    del domain_guidance
    rules = [
        "1. 结构锁定：绝对保持原有文本数据的数据结构、缩进和层级完全不变。",
        "2. 选择性翻译：仅翻译面向用户展示的可见文本内容。",
        "3. 禁止修改：严禁翻译或更改任何代码标签、键名 (Key)、变量占位符（如 @@P1@@、<f1-730/>、{{var}}、${var}、%s 等）或代码属性。",
        (
            "4. 缩写保留：英文缩写、专业简称和单位缩写按原拼写保留，包括大写、混合大小写、小写及含数字、空格或符号的形式。"
            "缩写不翻译、不展开；普通英文词和缩写后的英文释义正常翻译，不能仅因单词大写就保留它。\n"
            "以下是翻译格式示例，不要输出示例：\n"
            "原文：EPS rose 6%, while VNB and NB CSM were stable.\n"
            "译文：EPS增长6%，而VNB和NB CSM保持稳定。\n"
            "原文：MSe was 8%; capex fell 3% and NIM improved 4bps.\n"
            "译文：MSe为8%；capex下降3%，NIM改善4bps。\n"
            "原文：D&A: Depreciation and Amortization\n"
            "译文：D&A：折旧与摊销。\n"
            "原文：Revenue: Total sales\n"
            "译文：收入：总销售额。\n"
            "原文：SALES AND PROFIT improved.\n"
            "译文：销售额和利润有所改善。\n"
            "原文：IT supports our plans; it may help us.\n"
            "译文：IT支持我们的计划，它可能帮助我们。\n"
            "原文：ABC grew 12% (CER vs. XYZ at 7%).\n"
            "译文：ABC增长12%（CER口径，XYZ预期为7%）。\n"
            "注意数字与原文对象的对应关系，不把对比数值误译为两者差额。\n"
            "原文：NII grew 2%; NIM improved 3bps, while CET1 was stable.\n"
            "译文：NII增长2%；NIM改善3bps，而CET1保持稳定。\n"
            "原文：The API supports JSON, while eSIM uses QR codes.\n"
            "译文：API支持JSON，而eSIM使用QR码。"
        ),
    ]
    next_rule = 5
    if protocol.instruction:
        rules.append(_numbered_rule(next_rule, "输出协议", protocol.instruction))
        next_rule += 1
    if kind == "group":
        rules.append(
            _numbered_rule(
                next_rule,
                "Member 边界",
                "必须严格按照 group 数据中的 member_ids 和每个 member 的 source_text 分别输出译文；"
                "只返回每个 member 的译文，不要重复整组，不要借用相邻上下文。",
            )
        )
        next_rule += 1
    if direct_typst and protocol.include_math_guidance:
        rules.append(_numbered_rule(next_rule, "数学排版规则", direct_math_guidance(target_language_name=target_language_name)))
        next_rule += 1
    if protocol.include_sci_decision:
        rules.append(_numbered_rule(next_rule, "决策协议", load_prompt("translation_sci_decision.txt")))

    # 任务说明由 user 消息的翻译指令行承担,system 不再重复任务目标。
    lines = [
        HY_MT_RULES_HEADING,
        *rules,
    ]
    return "\n".join(lines).strip()


def _terms_guidance(item: TranslationItemContext) -> str:
    guidance = item.scoped_terms_guidance.strip()
    if not guidance:
        return ""
    if not guidance.startswith("参考下面的翻译："):
        guidance = f"参考下面的翻译：\n{guidance}"
    return guidance


def _structure_guidance(item: TranslationItemContext) -> str:
    if item.prompt_is_reference:
        return "参考文献规则：只翻译作品标题；作者、期刊、年份、卷期、页码和引文顺序保持原样。"
    if item.prompt_is_title:
        return "标题规则：译成简洁正式的标题，符合研报行文，不扩写为解释性句子。"
    structure_role = str((item.metadata or {}).get("structure_role", "") or "").strip().lower()
    if item.toc_entries or structure_role == "table_of_contents" or str(item.semantic_role or "").strip().lower() == "table_of_contents":
        return (
            "当前原文是目录/图表清单。必须逐行翻译，每个原文行输出一个译文行；"
            "翻译行首标签和标题，保留行尾页码且不要改动页码；不要合并行，不要输出解释。"
        )
    if item.preserve_line_structure_for_prompt and item.line_texts:
        return "当前原文是多行结构块；译文应尽量保持相同换行数量和行序，不要合并成普通段落。"
    return ""


def _context_guidance(item: TranslationItemContext) -> list[str]:
    guidance: list[str] = []
    context_before = item.context_before_for_prompt()
    context_after = item.context_after_for_prompt()
    if context_before:
        guidance.append(f"前文上下文（仅供理解，禁止翻译进输出）：{context_before}")
    if context_after:
        guidance.append(f"后文上下文（仅供理解，禁止翻译进输出）：{context_after}")
    if item.continuation_group:
        guidance.append(CONTINUATION_GROUP_NOTE)
    return guidance


def _math_repair_guidance(item: TranslationItemContext) -> list[str]:
    if not item.prompt_raw_direct_typst:
        return []
    guidance: list[str] = []
    if not has_balanced_unescaped_dollars(item.source_for_prompt()):
        guidance.append(MATH_DELIMITER_DAMAGE_HINT)
    rewrites = find_mitex_rewrites(item.source_for_prompt())
    if rewrites:
        pairs = "；".join(f"`{command}` 改用 `{preferred}`" for command, preferred in rewrites)
        guidance.append(f"渲染器不支持本段公式中的部分 LaTeX 写法，请在译文公式中替换：{pairs}。")
    return guidance


def background_prompt(
    item: TranslationItemContext,
    *,
    include_math_hints: bool = False,
) -> str:
    """Build item-specific user-side background; empty when nothing applies."""
    entries: list[str] = []
    next_number = 1
    incomplete = _source_looks_incomplete(item.source_for_prompt())

    def add(title: str, body: str) -> None:
        nonlocal next_number
        if str(body or "").strip():
            entries.append(_numbered_rule(next_number, title, body))
            next_number += 1

    if terms := _terms_guidance(item):
        add("术语", terms)
    if incomplete:
        add("完整性", "原文是未写完的半句，译文保持同等不完整，严禁从前后文借词补全。")
    for context in _context_guidance(item):
        add("上下文", context)
    if structure := _structure_guidance(item):
        add("结构", structure)
    if include_math_hints:
        for math_hint in _math_repair_guidance(item):
            add("数学修复", math_hint)
    if not entries:
        return ""
    return "\n".join([HY_MT_BACKGROUND_HEADING, *entries])


def plain_text_single_user_prompt(
    item: TranslationItemContext,
    *,
    target_language_name: str = DEFAULT_TARGET_LANGUAGE_NAME,
    include_math_hints: bool = False,
) -> str:
    background = background_prompt(item, include_math_hints=include_math_hints)
    parts = [hy_mt_input_heading(target_language_name), item.source_for_prompt()]
    return "\n\n".join([background, *parts]).strip() if background else "\n".join(parts).strip()


def _items_payload(batch: list[TranslationItemContext]) -> list[dict[str, Any]]:
    return [{"item_id": item.item_id, "source_text": item.source_for_prompt()} for item in batch]


def batch_json_user_prompt(
    batch: list[TranslationItemContext],
    *,
    target_language_name: str = DEFAULT_TARGET_LANGUAGE_NAME,
) -> str:
    backgrounds: list[str] = []
    for item in batch:
        if background := background_prompt(
            item,
            include_math_hints=bool(item.prompt_raw_direct_typst),
        ):
            backgrounds.append(f"{item.item_id}：\n{background}")
    parts: list[str] = []
    if backgrounds:
        parts.extend([HY_MT_BACKGROUND_HEADING, *backgrounds, ""])
    parts.extend([hy_mt_input_heading(target_language_name), json.dumps({"items": _items_payload(batch)}, ensure_ascii=False)])
    return "\n".join(parts).strip()


def single_json_user_prompt(
    item: TranslationItemContext,
    *,
    target_language_name: str = DEFAULT_TARGET_LANGUAGE_NAME,
    structured_decision: bool = False,
) -> str:
    del structured_decision  # The response schema is owned by the system rules.
    background = background_prompt(
        item,
        include_math_hints=bool(item.prompt_raw_direct_typst),
    )
    payload = {"item": {"item_id": item.item_id, "source_text": item.source_for_prompt()}}
    parts: list[str] = []
    if background:
        parts.extend([background, ""])
    parts.extend([hy_mt_input_heading(target_language_name), json.dumps(payload, ensure_ascii=False)])
    return "\n".join(parts).strip()


def group_member_json_user_prompt(
    item: TranslationItemContext,
    *,
    target_language_name: str = DEFAULT_TARGET_LANGUAGE_NAME,
) -> str:
    if item.prompt_member_ids_error is not None:
        raise TypeError(item.prompt_member_ids_error)
    if item.prompt_member_sources_error is not None:
        raise TypeError(item.prompt_member_sources_error)
    member_ids = list(item.prompt_member_ids) or [item.item_id]
    member_source_by_id = dict(item.prompt_member_sources)

    background_entries = background_prompt(
        item,
        include_math_hints=bool(item.prompt_raw_direct_typst),
    ).splitlines()
    # background_prompt uses one numbered list; the first line is its heading.
    background_body = "\n".join(background_entries[1:]).strip()

    group_payload: dict[str, Any] = {
        "item_id": item.item_id,
        "continuation_group": item.continuation_group,
        "member_ids": member_ids,
        "members": [
            {"item_id": member_id, "source_text": member_source_by_id.get(member_id, "")}
            for member_id in member_ids
        ],
    }
    if any(not member_source_by_id.get(member_id) for member_id in member_ids):
        group_payload["combined_source_text"] = item.source_for_prompt()

    payload: dict[str, Any] = {"group": group_payload}
    parts: list[str] = []
    if background_body:
        parts.extend([HY_MT_BACKGROUND_HEADING, background_body, ""])
    parts.extend([hy_mt_input_heading(target_language_name), json.dumps(payload, ensure_ascii=False)])
    return "\n".join(parts).strip()


__all__ = [
    "HY_MT_BACKGROUND_HEADING",
    "HY_MT_RULES_HEADING",
    "batch_json_user_prompt",
    "group_member_json_user_prompt",
    "hy_mt_input_heading",
    "plain_text_single_user_prompt",
    "single_json_user_prompt",
    "system_prompt",
]
