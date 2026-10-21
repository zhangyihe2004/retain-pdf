from __future__ import annotations

import re


MODEL_REQUEST_PROMPT_RE = re.compile(
    r"^(?:"
    r"请\s*(?:提供|输入|给出|粘贴|发送)\s*(?:待翻译的?)?\s*(?:原文|文本|内容|source)(?:[。.!！?？\s]*)|"
    r"(?:please\s+)?(?:provide|send|enter|paste)\s+(?:the\s+)?(?:source\s+)?(?:text|content)(?:\s+to\s+translate)?(?:[。.!！?？\s]*)"
    r")$",
    re.I,
)
MODEL_REQUEST_PROMPT_MAX_CHARS = 48

# Hy-MT user 模板回写：模型把任务说明原样复制进译文。
HY_MT_TASK_ECHO_MARKERS = (
    "# 任务目标",
    "# 严格约束",
    "# 背景信息",
    "# 数据输入",
    # The current background instruction is also a protocol echo, not a translation.
    "# 请结合背景信息将以下文本翻译为",
    "将数据输入中的",
    "将下方",
    "结构锁定",
    "选择性翻译",
    "禁止修改",
    "参考下面的翻译",
    # 现行通用 JSON 协议指令句前缀，模型整句回写时按协议壳拦截。
    "按 system 指定的 JSON 协议",
)


def looks_like_protocol_shell_output(translated_text: str) -> bool:
    text = str(translated_text or "").strip()
    if not text:
        return False
    if len(text) <= MODEL_REQUEST_PROMPT_MAX_CHARS and MODEL_REQUEST_PROMPT_RE.fullmatch(text):
        return True
    for marker in HY_MT_TASK_ECHO_MARKERS:
        if text.startswith(marker):
            return True
    if not text.startswith("{"):
        return False
    return (
        '"translated_text"' in text
        or '"translations"' in text
        or "“translated_text”" in text
        or "“translations”" in text
    )


__all__ = [
    "MODEL_REQUEST_PROMPT_MAX_CHARS",
    "MODEL_REQUEST_PROMPT_RE",
    "looks_like_protocol_shell_output",
]
