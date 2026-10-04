# Character sample

`original_character.png` was generated with OpenAI image generation on 2026-10-04 for this repository, following the user's request for a pink-haired, red-eyed, maid-like character for testing.

The prompt requested an original adult anime character with pastel pink bob hair, a silver star hair clip, a black-and-white maid-inspired outfit, and a mint ribbon. It explicitly excluded recreating an existing franchise character, insignia, pose, or composition. No existing character artwork was uploaded or copied as a fixture. This generated illustration is included as a manual-editing sample, not as an AI segmentation accuracy benchmark or training dataset.

The automated sample test splits the canvas into two geometric regions to verify non-destructive import, project save/load, recomposition, and PNG export. These are synthetic test masks, not inferred hair/face masks. Deterministic unit tests use small synthetic pixel arrays independently of this image.

## Committed artifacts

- `results/automatic/`: 8 deterministic color-region masks, color-coded overlay, editable project, PNG package, and layered PSD. Regions are color clusters, not semantic hair/face/eye annotations. Background is retained, and the PNG/PSD layer composites reconstruct the source exactly.
- `results/inpainting/`: a synthetic missing hair patch, before/after images and projects, target/generated masks, neural-repaired PNG package, and PSD. This intentionally withholds a visible 100×100 patch, predicts it with CPU LaMa, and preserves all other visible pixels. It demonstrates the inference path; it does not establish recovery of genuinely unseen anatomy or exact recovery of the withheld patch.
- `results/report.json`: recipe, model checksum, recorded preprocessing transform, artifact SHA-256 values, and observed checks. Cubism validation is recorded as unrun.
- `gpt_validation.json`: the historical first-key credit failure and the subsequent key's successful live alignment. No credential values are included.
- `parts_atlas.png` and `parts_atlas_generation.json`: a new transparent nine-part atlas generated using the original character as a reference, with the exact prompt and provenance. These are newly generated shapes, not pixels separated from the original. Shape differences and small alpha-edge speckles remain visible and need manual cleanup.
- `results/alignment/`: **live GPT** matching and initial placement of nine atlas parts using `gpt-5.6-luna`, including the actual response, original inputs, masks, original crops, transforms, editable project, transparent PNG layers, PSD, comparison/difference/coverage images, isolated-layer contact sheet, and report. Placement is a proposal: large parts have anchor residuals of approximately 20–43px, some crops extend outside the reference canvas, arms are absent from the generated atlas, and generated shapes differ from the reference. This is a functional demonstration, not finished Live2D material or a quality benchmark.
- `results/alignment_attempts/`: the recorded empty response from an earlier output-budget exhaustion and its diagnosis. No project or masks were created from invalid responses.
- `alignment_ui.png`: the current GUI displaying the nine-part alignment project.
- `manual_ui.png`: screenshot of the earlier manual-editing slice.
- `current_ui.png`: screenshot of the current app with the automatic sample loaded.
- `results/web/`: real Chromium editor checks, downloaded project/PNG/PSD/Web packages, screenshots, a recorded animation, and an independently served character bundle. The report distinguishes passed checks from a local-file check blocked by browser policy. The editable test project includes intentional mask edits and color proposals; `character-web.zip` contains the nine-part animated sample before those edits.
- `results/web_lama/`: real CPU LaMa through the Web API, including the input project, two proposals, rejection and adoption states, original/visible/hidden/generated pixels, saved and reloaded projects, and a report with artifact hashes. All ten checks passed; the synthetic 10,000-pixel withheld region was filled while source and visible pixels remained unchanged.

All example artwork and derived images/data are stored in this repository. The optional approximately 196 MiB model is a dependency stored outside the checkout; it is obtained explicitly through [docs/MODELS.md](../docs/MODELS.md).

Reproduce results into a **new** directory (the script refuses to overwrite):

```bash
uv run --locked --extra ai python scripts/create_samples.py --model /path/to/models/big-lama.pt --output /tmp/live2d-sample-results
```

The fixed occlusion coordinates are designed for the included 1254×1254 original fixture. Reload any `.l2split` file in the app to inspect editable masks and generation provenance. PSDs are verified with psd-tools, not with a running Cubism instance.

Replay the recorded successful AI correspondences without a key or network request:

```bash
uv run --locked python scripts/create_alignment_sample.py \
  --atlas samples/parts_atlas.png --model gpt-5.6-luna \
  --response samples/results/alignment/response.json \
  --output /tmp/alignment-replay
```

The offline replay was verified to reproduce identical composited pixels and part IDs. For a new live request, replace `--response ...` with `--upload`; the API key is read from `GPT_API_KEY` / `OPENAI_API_KEY` or a hidden prompt. `--response-log /new/path.json` can retain a response for diagnosis even when its proposal is rejected. The GUI requires separate upload and adoption confirmations. Keys are never included in these artifacts.

For Web playback, extract [character-web.zip](results/web/character-web.zip) and serve its directory with any static HTTP server. The [recorded motion](results/web/character_motion.webm) shows the exported model playing independently of the editor. Embedding instructions and browser-check reproduction are in [docs/WEB.md](../docs/WEB.md). The sample still needs artwork cleanup and has no separate arm parts; playback does not supply missing anatomy.

With the Web server running with `LAMA_MODEL_PATH` and both `web` and `ai` extras, reproduce the Web repair check into a new directory:

```bash
uv run --locked --extra web --extra ai python scripts/check_web_lama.py \
  --base-url http://127.0.0.1:8080 --output /tmp/new-web-lama-check
```
