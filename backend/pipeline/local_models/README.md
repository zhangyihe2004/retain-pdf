# Local OCR adapter

`paddle_vl_wrapper.py` is the runtime adapter used by the existing
`local_command` provider. It sends a PDF to Paddle's `/layout-parsing`,
unwraps `result`, and writes the raw payload for normal Paddle normalization.
Configure its URL and timeout with `RETAIN_PADDLE_LOCAL_API_URL` and
`RETAIN_PADDLE_LOCAL_API_TIMEOUT_SECONDS`.

The Dots runtime is `retainpdf_pipeline.ocr.ocr_provider.dots_mocr_runner`.
See the [Linux API deployment guide](../../../ops/deployment/docker/delivery/README.md#linux-api本地-paddle--dots-二选一--hy-mt)
for provider selection, service addresses, and Hy-MT configuration.

Formal regression tests live in `../devtools/tests/`. Personal end-to-end
helpers are maintained outside the repository and are not part of the runtime.

## Model service ports

All model services bind `127.0.0.1`. Dots uses `41010`, Hy-MT uses `41011`,
GLiNER uses `41012`; `41013` is reserved for Paddle (not started).
RetainPDF's own API and internal service ports remain unchanged.
On the current server, `DOTS_PORT`, `HYMT_PORT`, `GLINER_PORT` and the reserved
`PADDLE_PORT` live in `~/ocr/model-services/model-services.env`; the model launch
commands and systemd readiness checks use these settings.

For local Windows runs, open the model tunnel:

```powershell
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 41010:127.0.0.1:41010 -L 41011:127.0.0.1:41011 -L 41012:127.0.0.1:41012 zyh@10.249.42.254
```

## Optional GLiNER2 entity terminology

`gliner2_server.py` uses the official `AutoExtractor` and
`batch_extract_entities_long` APIs. It detects only `company`
on CPU, from title inputs. Long titles use the official
overlapping-window scan rather than truncation. Blocks remain separate to
avoid cross-block fake names.

On the current Linux model host, reuse the Hy-MT interpreter without upgrading
its Torch/vLLM environment. Install the small inference package into a separate
target directory (the interpreter already has Torch, Transformers, FastAPI and
Uvicorn):

```bash
~/.local/bin/uv pip install --python ~/ocr/.venv_hymt/bin/python \
  --target ~/ocr/gliner2-runtime --no-deps gliner2==2.1.0
PYTHONPATH=$HOME/ocr/gliner2-runtime CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
  ~/ocr/.venv_hymt/bin/python /path/to/retain-pdf/backend/pipeline/local_models/gliner2_server.py \
  --model ~/ocr/gliner2.5-base-v1 --port 41012 --threads 4
```

The model must already exist locally. On a fresh host, provide an independent
inference environment with GLiNER2's dependencies; the no-deps installation
above is specific to the existing Hy-MT runtime. Do not install ML dependencies
into RetainPDF's backend or Docker image.

Enable the preflight in the process launching RetainPDF:

```powershell
# Forward localhost:41012 to the model host, just like the existing model ports.
$env:RETAIN_GLINER2_URL = 'http://127.0.0.1:41012'
# Flash review uses DEEPSEEK_API_KEY, or pipeline/.env/deepseek.env locally.
# Then launch the pipeline through the normal API or a local helper.
```

For Docker, set `RETAIN_GLINER2_URL` in `docker/app.env` to the container-reachable
service base URL. The adapter binds localhost by default; only change `--host`
to a private/container-reachable address when the network is appropriately
restricted. Do not expose this unauthenticated adapter to the public internet.

The pipeline collects only canonical titles across all pages, including
disclosure sections, makes one GLiNER request, deduplicates the names, and
reviews that pool once with the official `deepseek-flash` API. Review sends
each candidate and only the titles where it occurs, never body text or the PDF.
It uses non-thinking JSON output and accepts only input candidate IDs; names
cannot be added, renamed or translated. Only the reviewed names become
occurrence-matched self-mappings in each translation segment's
existing terminology prompt. Longer overlapping mentions win, but a separate
short mention survives. Manual terminology takes priority. Automatic entities
are always segment-scoped, even with manual `glossary_mode=all`.
Captions, body paragraphs, short fields, leading bold phrases and table cells
are not NER inputs. There is no length/case fallback or disclosure cutoff.
The service verifies original spans and returns company mentions; whitespace
normalization and case-insensitive deduplication happen once in the pipeline.
Names found in eligible titles can still guide matching body segments.
The document pool participates in checkpoint identity; segment terms already
participate in the existing translation cache identity.
Review is independent of the Hy-MT/Rust translation transport. It uses the
existing `DEEPSEEK_API_KEY` credential, or `pipeline/.env/deepseek.env` for local
development; do not put the key in the GLiNER service or an image. Empty pools
make no Flash request. No new cache, uppercase rule or financial-word blacklist
is introduced. Review can still miss a company abbreviation or retain an
ambiguous name; it does not recover GLiNER omissions.

Unset `RETAIN_GLINER2_URL` (or set `glossary_mode=off`) for the original behavior.
An explicitly configured but unavailable/malformed service, missing review key,
or invalid/incomplete review response raises an error in
preflight rather than silently translating without the requested terminology.
NER can still miss or misclassify entities, and terminology hints are not hard
placeholders; this improves preservation but does not guarantee every name.

### Current server: systemd user service

On `/home/zyh/ocr/model-services`, `start-model.sh gliner` reuses `.venv_hymt`
with the separate `gliner2-runtime` package directory, no CUDA device, and
offline weights. `gliner.service` uses the same user service manager and journal
as Dots/Hy-MT; `wait-ready.py` checks its HTTP health without requiring a vLLM
model endpoint. The existing installer now includes GLiNER.

```bash
systemctl --user status gliner
systemctl --user restart gliner
systemctl --user stop gliner
journalctl --user -u gliner -f
curl -fsS http://127.0.0.1:41012/health
```

The installed service is enabled and the current user has `Linger=yes`, so it
does not depend on the SSH session and starts at boot. It binds localhost by
default. `GLINER_HOST`, `GLINER_PORT` and `GLINER_THREADS` in the existing
`model-services.env` can override the defaults (127.0.0.1, 41012, 4); restart only
GLiNER after changing them and update the client URL/tunnel if the port changes.
For same-host Docker, use a restricted host/bridge address reachable from the
container, not a public unauthenticated listener. Dots/Hy-MT are not restarted
when installing or restarting GLiNER.

### Trial findings

- Document-wide title pooling helps when one title detects `Boss Zhipin` but another
  detects only `Zhipin`. Do not remove nested short names from the document pool.
- The short-input processor can append a period (`Kanzhun Ltd.`). Accept only
  the verifiable single appended period, slice back to the original text, and
  discard other inconsistent spans.
- CPU extraction avoids changing the two GPU model services. Keep one loaded
  extractor and serialize calls rather than spawning a model per PDF/segment.

Official references: [entity extraction](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/2-ner.md),
[long-context extraction](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/12-long_context.md).
