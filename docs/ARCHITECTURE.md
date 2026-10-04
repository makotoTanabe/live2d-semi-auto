# Editable parts, local inference, and export

This implements the manual path plus color-region proposals, local neural repair, and PSD export. Semantic character parsing and Cubism desktop validation remain outside the verified scope. The decisions are recorded in [ADR 0001](adr/0001-local-inference-and-psd.md).

## Boundaries

- `core.py`: project/part data, validation, mask bounds, RGBA compositing. No UI or model imports.
- `application.py`: editing transactions, undo/redo, accepting proposals.
- `infrastructure.py`: Pillow import, project archives, PNG export.
- `inference.py`: backend protocol and an explicit nontransparent-pixel baseline.
- `inpainting.py`: local LaMa and non-AI Telea adapters with typed repair proposals.
- `models.py`: explicit, checksum-verified acquisition of the optional checkpoint.
- `gpt_parts.py`: opt-in remote GPT classification/boxes and local GrabCut refinement; no core model dependency.
- `exporters.py`: exporter protocol and an sRGB PSD adapter.
- `ui.py`: PySide6 desktop widgets, dialogs, coordinate conversion, rendering.

Python 3.12+, NumPy, Pillow, headless OpenCV, PySide6, and psd-tools are runtime dependencies. pytest is a development dependency. OpenCV provides brush lines, polygon filling, color-space conversion and Telea repair. PyTorch is optional in the `ai` extra and is not imported by the domain. Dependencies and supported platform wheels are locked in `uv.lock`.

## Coordinates and pixels

The canonical canvas is the imported image after EXIF orientation normalization. Pixel origin is top-left, x points right, y points down. There is no resize during import, editing, save, or export. Qt view transforms affect display only; pointer coordinates are mapped back into canvas coordinates before painting. Brush radius is in original pixels, independent of zoom. Lasso is a freehand polygon closed on mouse release.

Source pixels are an immutable `uint8[height,width,4]` RGBA array. Each mask is a separate `uint8[height,width]` array with values 0–255. Parts reference stable UUIDs; names and types are editable. The list order is bottom to top, serialized as contiguous `z_order` integers. The alpha of an exported layer is source alpha multiplied by mask/255. Compositing uses source-over and preserves straight RGB for nontransparent pixels; RGB values under fully transparent pixels have no visual meaning.

Each part also has a requested hidden-region mask, immutable accepted generated RGBA pixels, and an immutable generated-region mask. Inference targets must not overlap that part's visible mask. Original pixels always take priority during composition; generated pixels only supply the currently non-visible portion. Metadata and assets are preserved separately to make repairs inspectable and reversible.

Original ICC metadata is retained for PNG. The PSD adapter converts profile-tagged pixels to sRGB and embeds the destination profile; untagged RGB is assumed sRGB. Color-managed CMYK import and Cubism desktop validation are future work.

## Project archive, schema version 2

`.l2split` is a ZIP archive containing:

```text
manifest.json
source.png
masks/0.png
masks/1.png
hidden_mask/0.png       # optional requested repair area
generated/0.png         # optional accepted pixels
generated_mask/0.png    # optional accepted provenance
...
```

The manifest records schema/application versions, canvas dimensions, original filename and file SHA-256, source asset path, part IDs, names, types, visibility, order, mask paths, optional generation assets, and generation history. History includes backend, model checksum and inference crop/resize/padding transforms. Schema 1 archives remain readable. The original hash identifies the imported file, including its original metadata; it is not the hash of the normalized embedded PNG. The project is self-contained, with original ICC profile embedded in `source.png`. Loading does not extract ZIP paths onto the filesystem. Unknown schemas, missing assets, malformed order, duplicate identities, and canvas/mask mismatches are rejected. Empty masks are permitted in saved drafts.

Saving writes a temporary archive alongside the destination, flushes it, and replaces the project atomically. A failed replacement leaves the previous archive intact. The imported original path is held only in memory for overwrite protection, not embedded as a machine-specific absolute path.

PNG export validates names and masks, stages files, then reserves a new destination directory. Existing destinations are rejected. A failed transfer removes only the new destination created by that export. Files use a sanitized name plus numeric order to avoid path traversal and filename collisions. The manifest preserves original names/IDs, operation history, generated-mask paths and visible-mask bounds as exclusive `(left, top, right, bottom)`. Every part is exported even if hidden; visibility controls the preview and is recorded for consumers. A part can be exported with only generated coverage. PSD export writes to a verified complete temporary file and publishes it without replacing existing destinations.

## History and current limitations

Undo/redo holds up to 50 snapshots of editable masks and part metadata, sharing the immutable source and accepted generation arrays. One brush stroke is one transaction. Undo history is session-only; save/load retains the resulting editable state and generation provenance, not the undo timeline.

Local color separation does not classify hair/face. Optional GPT proposals supply semantic names/boxes, refined with GrabCut, but have not been validated with live credentials. Anatomical hidden-region detection, grouping, side metadata, mask merge/split, and rigging remain unimplemented. All inference proposals must be inspected and explicitly accepted; rejection leaves manual edits unchanged. GPT additionally requires explicit artwork-upload confirmation. GUI inference and PNG/PSD export run in worker threads. Import and project save/load remain synchronous. Cancelled inference discards the result when the model returns, without interrupting a running kernel or undoing an already-sent API request. Large-scale memory optimization remains future work.

Tests use deterministic synthetic images for domain logic, real Qt mouse events for the edit path, and the generated original character for a representative file roundtrip. Headless GUI tests verify widgets and editing behavior; interactive desktop usability and Windows/macOS packaging need separate validation.
