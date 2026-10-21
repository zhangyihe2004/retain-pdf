"""Thin CPU HTTP adapter around the official GLiNER2.5 extraction API."""

from __future__ import annotations

import argparse
from threading import Lock

from fastapi import FastAPI
from pydantic import BaseModel


ENTITY_TYPES = {
    "company": {"dtype": "list", "threshold": 0.6},
}


class EntityRequest(BaseModel):
    texts: list[str]


def create_app(extractor) -> FastAPI:
    app = FastAPI()
    lock = Lock()  # The official processor has mutable request state.

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "entity_types": list(ENTITY_TYPES)}

    @app.post("/entities")
    def entities(request: EntityRequest) -> dict:
        texts = [text for text in request.texts if text.strip()]
        if not texts:
            return {"entities": []}
        with lock:
            results = extractor.batch_extract_entities_long(
                texts,
                ENTITY_TYPES,
                batch_size=8,
                chunk_size=384,
                chunk_overlap=64,
                include_spans=True,
                overlap_policy="allow",
            )
        names: list[str] = []
        for text, result in zip(texts, results, strict=True):
            for entity in result.get("entities", {}).get("company", []):
                start, end = entity["start"], entity["end"]
                # Short inputs may include the processor's appended period.
                appended_period = (
                    end == len(text) + 1
                    and not text.endswith((".", "!", "?"))
                    and entity["text"] == text[start:] + "."
                )
                if appended_period:
                    end = len(text)
                if not (0 <= start < end <= len(text)):
                    continue
                if not appended_period and entity["text"] != text[start:end]:
                    continue
                # Return original mentions; the pipeline consolidates the pool.
                name = text[start:end]
                if name.strip():
                    names.append(name)
        return {"entities": names}

    return app


def main() -> None:
    import torch
    import uvicorn
    from gliner2 import AutoExtractor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local GLiNER2.5-base-v1 model directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41012)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    model = AutoExtractor.from_pretrained(args.model, map_location="cpu", local_files_only=True)
    model.eval()
    uvicorn.run(create_app(model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
