# Explicit local AI model setup

Manual editing, color-region separation, Telea, and PNG/PSD export do not require a model.

```bash
uv sync --locked --extra ai
uv run --locked --extra ai python -m live2d_semi_auto.models download-lama --output /path/to/models/big-lama.pt
export LAMA_MODEL_PATH=/path/to/models/big-lama.pt
uv run --locked --extra ai live2d-semi-auto
```

On Windows, set `LAMA_MODEL_PATH` through PowerShell's `$env:LAMA_MODEL_PATH` or select the `.pt` file in the GUI. Linux/Windows uv installations use PyTorch's CPU wheel index; macOS uses the default PyPI source. Only Linux CPU inference was validated in this environment. Models are optional and never fetched by `uv sync` or by opening the app.

The downloader validates HTTPS and the pinned SHA-256, and publishes a complete verified file exclusively without overwriting existing files. Existing files are checked on repeated invocations. Model acquisition needs `github.com` and its release asset host `release-assets.githubusercontent.com`; Python dependencies use PyPI and `download.pytorch.org` (including `download-r2.pytorch.org` redirects). No API token is needed for these public assets.

Model provenance:

- Original LaMa project: https://github.com/advimman/lama (Apache-2.0 source).
- Checkpoint source linked by IOPaint: https://github.com/Sanster/IOPaint/blob/main/iopaint/model/lama.py
- Artifact: https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt
- SHA-256: `344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9`.

The model is stored outside the repository and is not a sample artifact. Sample results and their model provenance are committed under `samples/results`; the roughly 196 MiB checkpoint is not committed to Git. No model weights are necessary to inspect those results.

To explicitly run the real neural inference test:

```bash
LAMA_MODEL_PATH=/path/to/models/big-lama.pt QT_QPA_PLATFORM=offscreen uv run --locked --extra ai pytest -q
```

Without a configured model, the real inference test is marked skipped. The rest of the suite runs without downloads and uses synthetic images to verify preservation, provenance, undo, archive compatibility, and PSD structure.

## Optional GPT vision part proposals

Set `GPT_API_KEY` securely in cloud environment settings, or your local process environment. Do not paste API keys into chat or repository files. `GPT_MODEL` defaults to `gpt-4.1` and can be changed to a model supporting images and structured JSON outputs. A local `OPENAI_API_KEY` is also accepted, but the cloud secret binding uses `GPT_API_KEY` because the platform reserves `OPENAI_` names.

The GUI identifies `api.openai.com` and asks before sending the original artwork (flattened onto white and resized to at most 1024 px). It uses Chat Completions with a strict JSON schema for unique names, kinds, and normalized bounding boxes. The image is not sent during import/startup or by the local-only `scripts/create_samples.py`. GPT sample runners require an explicit `--upload` flag. API charges and OpenAI's applicable data handling policy apply to this explicit request. Cancelling after a request has begun can discard its result but cannot undo the transmission or charges.

Returned boxes are mapped into canvas coordinates, refined with local GrabCut, and fall back to rectangles if refinement fails. They are editable proposals, with classification, boxes, model, resize transform and refinement method preserved in operation history. GPT localization may be inaccurate or overlapping; it does not produce certified pixel masks, guarantee complete coverage, or infer genuinely hidden anatomy.

Schema handling, coordinate conversion, mask refinement, rejection of malformed output, and the GUI upload-confirmation gate are tested with controlled responses. On 2026-10-04 the first explicitly supplied key authenticated successfully and could read the gpt-4.1 model. Vision and minimal text completion requests were rejected with HTTP 429, `insufficient_quota` / `credit_balance_exhausted`: that API account had no credits remaining. This historical result is retained in [samples/gpt_validation.json](../samples/gpt_validation.json). A subsequently supplied service-account key authenticated, exposed `gpt-5.6-luna`, and completed both a minimal text request and the real dual-image atlas-alignment request documented below. The current key is usable; the earlier balance failure is not the current blocker. A live request does not certify segmentation or matching quality. No credential value was saved.

With credits and an available image/structured-output model, explicitly generate GPT part-proposal samples with:

```bash
uv run --locked python scripts/create_gpt_sample.py --upload --output samples/gpt_results
```

The script uses an injected environment binding or prompts for a hidden key; it never takes the key as a command-line argument or saves it. It writes the response, masks, editable project, PNG layers, PSD, and coverage report after a successful request. Preserve current data by choosing a new output directory. A saved `response.json` can be replayed offline using `--response /path/to/response.json` instead of `--upload`. Reports include uncovered and overlapping pixels; correct reconstruction alone does not establish useful semantic separation. Independent overlapping masks may duplicate opaque artwork and increase the alpha of translucent artwork.

## Optional GPT atlas alignment

The same key binding and Chat Completions transport are used by `GPTAlignmentBackend`. Choose a model available to the supplied API project; the `gpt-4.1` default does not grant access to that model. The GUI allows the model name to be entered for atlas alignment, and the sample runner accepts `--model`. Image input and strict structured JSON support are required.

Alignment requests allow 8192 completion tokens by default and request low reasoning effort for GPT-5 models. An earlier live attempt exhausted a 4096-token budget on reasoning without returning candidate JSON; the backend reports truncated responses as failures rather than accepting incomplete candidates. Invalid coordinates are also rejected. These API failures leave existing edits intact.

The user first imports a completed reference, then selects a separate atlas via 「GPTでシートを位置合わせ（外部送信）」. The GUI and CLI require identical ICC profile bytes or two untagged images before uploading; a mismatch is rejected with instructions to keep the originals and export new copies using the same sRGB profile. Before the request the GUI identifies `api.openai.com`, the selected model, both images, and API charges. Both images are flattened onto white and reduced independently to at most 1024 px for the request; full-resolution original assets stay local and are preserved in the project. Neither import nor manual placement adjustment sends artwork. An already-sent request cannot be recalled by cancelling its result.

The strict response includes one isolated-part bounding box in atlas-normalized coordinates, 2–8 paired reference/atlas anchors, a name, kind, confidence, notes, and bottom-to-top order for each of up to 24 proposals. Local validation rejects malformed, duplicate, degenerate or unusable candidates. Local fitting computes uniform scale/rotation/translation without reflection. It records API resize transforms, original source/destination sizes, crop, anchors, matrix, residuals, model, confidence, and notes. Low confidence, high anchor residuals and transformed crops outside the canvas produce warnings; a good residual does not prove that anatomical landmarks were correctly matched.

Extraction uses source alpha or explicit border-connected near-white removal. White removal is a heuristic and may remove white clothing connected to the crop edge or retain enclosed white background. Warping uses premultiplied alpha. Alignment adds imported artwork rather than generating new pixels or overwriting the completed reference. Preview/adoption are separate from upload, existing manual parts remain, and accepted placements can be manually corrected with Undo/Redo before hidden-region repair. Corrections use a stable pivot derived from the original extracted asset, re-render its original pixels and a separately retained canonical edited mask, and avoid cumulative image/mask resampling. Review layer order, visibility, duplicate coverage, clipping, and genuinely missing regions before export.

For an explicit live request:

```bash
uv run --locked python scripts/create_alignment_sample.py \
  --atlas samples/parts_atlas.png \
  --background alpha \
  --model gpt-5.6-luna \
  --upload \
  --output /path/to/new_alignment_result
```

`--source` defaults to `samples/original_character.png`; specify another completed reference when using a different character. The model above succeeded in this environment; use a model available to your own API project. `--background white` opts into the heuristic. The runner uses an injected key or hidden prompt, never a key argument, and preserves the original atlas, API response, masks, `.l2split`, PNG layers, PSD, recomposition/difference/coverage images, and verification report in a new output directory. Use `--response /path/to/response.json` instead of `--upload` for an offline replay. Optional `--response-log /path/to/new_response.json` saves a successful HTTP response before candidate validation, including candidates subsequently rejected by the validator; the log path must be new and contains no key or request headers.

## Recorded live alignment result

On 2026-10-04 a real `gpt-5.6-luna` request matched the completed original character against the newly generated transparent [parts atlas](../samples/parts_atlas.png) and returned 9 candidates: costume/body, back hair, face/neck, front hair, headband, visible eye, neck ribbon, hair clip, and mouth. The response consumed 4020 total tokens, including 1178 completion tokens with 232 reasoning tokens. The [response](../samples/results/alignment/response.json), [verification report](../samples/results/alignment/report.json), [reference/recomposition comparison](../samples/results/alignment/comparison.png), [isolated layers](../samples/results/alignment/isolated_layers.png), editable project, PNG package and PSD are committed under [samples/results/alignment](../samples/results/alignment).

Project save/load preserved the recomposition, stable IDs and original atlas pixels. The PSD was reopened and each independent layer's RGB and alpha matched its exported artwork. These checks passed, but the material still needs human correction: larger parts have anchor RMS residuals of approximately 20.7–43.3 px on the 1254×1254 canvas, the generated shapes differ from the reference, the atlas costume is cropped and lacks separate arms, and front-hair/headband crop bounds extend outside the canvas. Inspect and adjust these candidates before use. Genuine hidden-region completeness and Cubism desktop import remain unverified.
