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

The GUI identifies `api.openai.com` and asks before sending the original artwork (flattened onto white and resized to at most 1024 px). It uses Chat Completions with a strict JSON schema for unique names, kinds, and normalized bounding boxes. The image is not sent during import/startup or by the sample-generation script. API charges and OpenAI's applicable data handling policy apply to this explicit request. Cancelling after a request has begun can discard its result but cannot undo the transmission or charges.

Returned boxes are mapped into canvas coordinates, refined with local GrabCut, and fall back to rectangles if refinement fails. They are editable proposals, with classification, boxes, model, resize transform and refinement method preserved in operation history. GPT localization may be inaccurate or overlapping; it does not produce certified pixel masks, guarantee complete coverage, or infer genuinely hidden anatomy.

No GPT API key was available during implementation. Schema handling, coordinate conversion, mask refinement, rejection of malformed output, and the GUI upload-confirmation gate are tested with controlled responses. Live GPT classification and quality remain unverified; the committed sample outputs are local color clustering and LaMa, not fabricated GPT results.
