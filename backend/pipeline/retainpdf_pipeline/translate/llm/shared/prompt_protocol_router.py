from __future__ import annotations

from retainpdf_pipeline.translate.core.context import TranslationItemContext

from retainpdf_pipeline.translate.llm.shared import hy_mt_prompt_protocol as hy_mt
from retainpdf_pipeline.translate.llm.shared import prompt_protocols as generic


def _is_hy_mt2_model(model: str) -> bool:
    # Lazy import keeps prompt_building -> router -> provider.client from
    # becoming a hard import cycle.
    from retainpdf_pipeline.translate.llm.providers.deepseek.client import is_hy_mt2_model
    return is_hy_mt2_model(model)


def protocol_system_prompt(
    kind: str,
    *,
    domain_guidance: str,
    mode: str,
    response_style: str,
    target_language_name: str,
    direct_typst: bool,
    structured_decision: bool = False,
    model: str = "",
) -> str:
    if _is_hy_mt2_model(model):
        return hy_mt.system_prompt(
            kind,
            domain_guidance=domain_guidance,
            mode=mode,
            response_style=response_style,
            target_language_name=target_language_name,
            direct_typst=direct_typst,
            structured_decision=structured_decision,
        )
    return generic.protocol_system_prompt(
        kind,
        domain_guidance=domain_guidance,
        mode=mode,
        response_style=response_style,
        target_language_name=target_language_name,
        direct_typst=direct_typst,
        structured_decision=structured_decision,
    )


def single_user_prompt(
    item: TranslationItemContext,
    *,
    mode: str,
    response_style: str,
    target_language_name: str,
    structured_decision: bool = False,
    model: str = "",
) -> str:
    if not _is_hy_mt2_model(model):
        return generic.single_user_prompt(
            item,
            mode=mode,
            response_style=response_style,
            target_language_name=target_language_name,
            structured_decision=structured_decision,
        )
    if item.math_mode == "direct_typst":
        return hy_mt.plain_text_single_user_prompt(item, target_language_name=target_language_name, include_math_hints=True)
    if (mode == "sci" and structured_decision) or response_style == "json":
        return hy_mt.single_json_user_prompt(
            item,
            target_language_name=target_language_name,
            structured_decision=structured_decision,
        )
    return hy_mt.plain_text_single_user_prompt(item, target_language_name=target_language_name)


def direct_typst_batch_user_prompt(
    batch: list[TranslationItemContext],
    *,
    mode: str,
    target_language_name: str = generic.DEFAULT_TARGET_LANGUAGE_NAME,
    response_style: str = "tagged",
    model: str = "",
) -> str:
    if not _is_hy_mt2_model(model):
        return generic.direct_typst_batch_user_prompt(
            batch,
            mode=mode,
            target_language_name=target_language_name,
            response_style=response_style,
        )
    return hy_mt.batch_json_user_prompt(batch, target_language_name=target_language_name)


def batch_json_user_prompt(
    batch: list[TranslationItemContext],
    *,
    target_language_name: str = generic.DEFAULT_TARGET_LANGUAGE_NAME,
    model: str = "",
) -> str:
    if _is_hy_mt2_model(model):
        return hy_mt.batch_json_user_prompt(batch, target_language_name=target_language_name)
    return generic.batch_json_user_prompt(batch, target_language_name=target_language_name)


def group_member_json_user_prompt(
    item: TranslationItemContext,
    *,
    target_language_name: str = generic.DEFAULT_TARGET_LANGUAGE_NAME,
    model: str = "",
) -> str:
    if _is_hy_mt2_model(model):
        return hy_mt.group_member_json_user_prompt(item, target_language_name=target_language_name)
    return generic.group_member_json_user_prompt(item, target_language_name=target_language_name)


__all__ = [
    "batch_json_user_prompt",
    "direct_typst_batch_user_prompt",
    "group_member_json_user_prompt",
    "protocol_system_prompt",
    "single_user_prompt",
]
