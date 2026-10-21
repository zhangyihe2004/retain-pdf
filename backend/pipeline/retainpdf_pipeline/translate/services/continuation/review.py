from __future__ import annotations

import json
import time
from typing import Callable

import requests

from retainpdf_pipeline.foundation.shared.local_env import get_secret
from retainpdf_pipeline.translate.llm.providers.deepseek.client import is_hy_mt2_model
from retainpdf_pipeline.translate.llm.shared.executor_context import scoped_request, execution_enabled

from retainpdf_pipeline.translate.prompt_loader import load_prompt
from retainpdf_pipeline.translate.llm.shared.structured_models import CONTINUATION_REVIEW_RESPONSE_SCHEMA
from retainpdf_pipeline.translate.llm.shared.structured_parsers import parse_continuation_review_response
from retainpdf_pipeline.translate.llm.shared import upstream_resilience as _resilience


def _build_messages(pairs: list[dict]) -> list[dict[str, str]]:
    payload = {
        "task": load_prompt("continuation_review_task.txt"),
        "pairs": pairs,
    }
    return [
        {"role": "system", "content": load_prompt("continuation_review_system.txt")},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _request_deepseek_review(messages: list[dict[str, str]]) -> str:
    key = get_secret(env_var="DEEPSEEK_API_KEY", env_file_name="deepseek.env")
    if not key:
        raise RuntimeError("Continuation review requires DEEPSEEK_API_KEY or pipeline/.env/deepseek.env")
    # Independent review must not inherit the Rust/Hy-MT translation worker.
    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "deepseek-flash",
            "messages": messages,
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "max_tokens": 2048,
            "stream": False,
        },
        timeout=(10, 120),
    )
    response.raise_for_status()
    choice = response.json()["choices"][0]
    if choice.get("finish_reason") != "stop":
        # Incomplete decisions use the existing advisory parse fallback.
        return ""
    return choice["message"]["content"]


def review_candidate_pairs(
    pairs: list[dict],
    *,
    api_key: str,
    model: str,
    base_url: str,
    request_label: str = "",
    request_chat_content_fn: Callable[..., str],
    max_retries: int = _resilience.REVIEW_MAX_RETRIES,
    sleep_fn: Callable[[float], None] | None = None,
) -> dict[str, str]:
    if not pairs:
        return {}
    use_deepseek = is_hy_mt2_model(model)
    max_retries = 0 if execution_enabled() and not use_deepseek else max(0, int(max_retries))
    sleep = sleep_fn or time.sleep
    last_error: Exception | None = None
    # Transient upstream errors (Timeout/429/5xx/connection) get a limited
    # retry with exponential backoff; 400/401/402/403 fail fast with a
    # balance/Key hint and are never retried.
    for attempt in range(max_retries + 1):
        try:
            if use_deepseek:
                content = _request_deepseek_review(_build_messages(pairs))
            else:
                content = scoped_request("continuation", sorted((str(pair.get("prev_item_id", "")),str(pair.get("next_item_id", ""))) for pair in pairs), request_chat_content_fn,
                    _build_messages(pairs),
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                    temperature=0.0,
                    response_format=CONTINUATION_REVIEW_RESPONSE_SCHEMA,
                    timeout=120,
                    request_label=request_label,
                )
            try:
                return parse_continuation_review_response(content)
            except ValueError as parse_error:
                # Advisory review must never kill the job: rule-based groups
                # stand as-is when the model returns unparseable output
                # (weak models, relays ignoring response_format).
                preview = " ".join((content or "").split())[:200]
                print(
                    f"{request_label or 'continuation-review'}: review response "
                    f"unparseable, keeping rule decisions: {parse_error}; "
                    f"preview={preview!r}",
                    flush=True,
                )
                return {}
        except Exception as exc:  # noqa: BLE001 - classification decides retry vs fail-fast
            last_error = exc
            if _resilience.is_non_retryable_client_error(exc):
                raise type(exc)(_resilience.describe_upstream_error(exc)) from exc
            if attempt >= max_retries or not _resilience.is_transient_upstream_error(exc):
                raise
            delay = _resilience.retry_after_seconds(exc)
            if delay is None:
                delay = _resilience.review_retry_delay(attempt)
            if request_label:
                print(
                    f"{request_label}: continuation review transient failure "
                    f"attempt {attempt + 1}/{max_retries + 1}, retrying in {delay:.2f}s: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Continuation review failed without an exception.")
