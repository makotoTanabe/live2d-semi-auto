# Character sample

`original_character.png` was generated with OpenAI image generation on 2026-10-04 for this repository, following the user's request for a pink-haired, red-eyed, maid-like character for testing.

The prompt requested an original adult anime character with pastel pink bob hair, a silver star hair clip, a black-and-white maid-inspired outfit, and a mint ribbon. It explicitly excluded recreating an existing franchise character, insignia, pose, or composition. No existing character artwork was uploaded or copied as a fixture. This generated illustration is included as a manual-editing sample, not as an AI segmentation accuracy benchmark or training dataset.

The automated sample test splits the canvas into two geometric regions to verify non-destructive import, project save/load, recomposition, and PNG export. These are synthetic test masks, not inferred hair/face masks. Deterministic unit tests use small synthetic pixel arrays independently of this image.

## Committed artifacts

- `results/automatic/`: 8 deterministic color-region masks, color-coded overlay, editable project, PNG package, and layered PSD. Regions are color clusters, not semantic hair/face/eye annotations. Background is retained, and the PNG/PSD layer composites reconstruct the source exactly.
- `results/inpainting/`: a synthetic missing hair patch, before/after images and projects, target/generated masks, neural-repaired PNG package, and PSD. This intentionally withholds a visible 100×100 patch, predicts it with CPU LaMa, and preserves all other visible pixels. It demonstrates the inference path; it does not establish recovery of genuinely unseen anatomy or exact recovery of the withheld patch.
- `results/report.json`: recipe, model checksum, recorded preprocessing transform, artifact SHA-256 values, and observed checks. Cubism validation is recorded as unrun.
- `gpt_validation.json`: the live GPT attempt's result. Authentication/model access succeeded; generation was blocked by an exhausted API credit balance. There are no fabricated GPT masks or images and no credential values in this report.
- `manual_ui.png`: screenshot of the earlier manual-editing slice.
- `current_ui.png`: screenshot of the current app with the automatic sample loaded.

All example artwork and derived images/data are stored in this repository. The optional approximately 196 MiB model is a dependency stored outside the checkout; it is obtained explicitly through [docs/MODELS.md](../docs/MODELS.md).

Reproduce results into a **new** directory (the script refuses to overwrite):

```bash
uv run --locked --extra ai python scripts/create_samples.py --model /path/to/models/big-lama.pt --output /tmp/live2d-sample-results
```

The fixed occlusion coordinates are designed for the included 1254×1254 original fixture. Reload any `.l2split` file in the app to inspect editable masks and generation provenance. PSDs are verified with psd-tools, not with a running Cubism instance.
