"""Run the local PaddleOCR-VL HTTP service as a RetainPDF local OCR command.

The RetainPDF ``local_command`` driver owns job-directory handling and invokes
this file with the source PDF and output paths supplied through environment
variables.  Paddle's serving API wraps the layout payload in ``result``;
RetainPDF's Paddle adapter expects that payload at the top level, so this
wrapper deliberately unwraps it before writing the raw artifact.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import requests


DEFAULT_PADDLE_URL = "http://127.0.0.1:41013/layout-parsing"
PADDLE_URL_ENV = "RETAIN_PADDLE_LOCAL_API_URL"
PADDLE_TIMEOUT_ENV = "RETAIN_PADDLE_LOCAL_API_TIMEOUT_SECONDS"
DEFAULT_TIMEOUT_SECONDS = 3600.0


def _timeout_seconds() -> float:
    raw = str(os.environ.get(PADDLE_TIMEOUT_ENV, "") or "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_TIMEOUT_SECONDS


def _source_pdf() -> Path:
    value = str(os.environ.get("RETAIN_OCR_SOURCE_PDF", "") or "").strip()
    if not value:
        raise RuntimeError("RETAIN_OCR_SOURCE_PDF is required")
    path = Path(value).resolve()
    if not path.is_file():
        raise RuntimeError(f"source PDF does not exist: {path}")
    return path


def _output_path() -> Path:
    value = str(os.environ.get("RETAIN_OCR_RAW_PAYLOAD_JSON", "") or "").strip()
    if not value:
        raise RuntimeError("RETAIN_OCR_RAW_PAYLOAD_JSON is required")
    return Path(value).resolve()


def build_request_payload(source_pdf_path: Path) -> dict[str, Any]:
    encoded = base64.b64encode(source_pdf_path.read_bytes()).decode("ascii")
    return {
        "file": encoded,
        "fileType": 0,
        "visualize": False,
        "returnMarkdownImages": False,
    }


def unwrap_paddle_response(response_payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(response_payload, dict):
        raise RuntimeError("Paddle response must be a JSON object")

    error_code = response_payload.get("errorCode", 0)
    if str(error_code) not in {"0", ""} and error_code is not None:
        message = str(response_payload.get("errorMsg", "unknown Paddle error") or "unknown Paddle error")
        raise RuntimeError(f"Paddle layout-parsing failed ({error_code}): {message}")

    result = response_payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("Paddle response does not contain result")
    if not isinstance(result.get("layoutParsingResults"), list):
        raise RuntimeError("Paddle result does not contain layoutParsingResults")

    # Keep the provider payload shape expected by
    # ocr.document_schema.provider_adapters.paddle.adapter.
    payload = dict(result)
    payload["_meta"] = {
        "source": "local_paddle_vl_http",
        "provider": "paddle",
        "logId": str(response_payload.get("logId", "") or ""),
        "errorMsg": str(response_payload.get("errorMsg", "") or ""),
    }
    return payload


def run() -> Path:
    source_pdf_path = _source_pdf()
    output_path = _output_path()
    url = str(os.environ.get(PADDLE_URL_ENV, DEFAULT_PADDLE_URL) or DEFAULT_PADDLE_URL).strip()
    if not url:
        raise RuntimeError(f"{PADDLE_URL_ENV} must not be empty")

    session = requests.Session()
    session.trust_env = False
    try:
        response = session.post(
            url,
            json=build_request_payload(source_pdf_path),
            timeout=_timeout_seconds(),
        )
        response.raise_for_status()
        payload = unwrap_paddle_response(response.json())
    finally:
        session.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    path = run()
    print(f"PaddleOCR-VL payload: {path}", flush=True)


if __name__ == "__main__":
    main()
