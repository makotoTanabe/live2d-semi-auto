# Initial manual editing slice

This implements the manual path through the MVP; it does not claim completion of every feature in the product specification.

## Boundaries

- `core.py`: project/part data, validation, mask bounds, RGBA compositing. No UI or model imports.
- `application.py`: editing transactions, undo/redo, accepting proposals.
- `infrastructure.py`: Pillow import, project archives, PNG export.
- `inference.py`: backend protocol and an explicit nontransparent-pixel baseline.
- `ui.py`: PySide6 desktop widgets, dialogs, coordinate conversion, rendering.

Python 3.12+, NumPy, Pillow, headless OpenCV, and PySide6 are runtime dependencies. pytest is a development dependency. OpenCV provides brush lines and polygon filling; PyTorch is not required. Dependencies and supported platform wheels are locked in `uv.lock`. Review the dependencies' licenses (including Qt/PySide6 distribution obligations) before distributing a packaged application.

## Coordinates and pixels

The canonical canvas is the imported image after EXIF orientation normalization. Pixel origin is top-left, x points right, y points down. There is no resize during import, editing, save, or export. Qt view transforms affect display only; pointer coordinates are mapped back into canvas coordinates before painting. Brush radius is in original pixels, independent of zoom. Lasso is a freehand polygon closed on mouse release.

Source pixels are an immutable `uint8[height,width,4]` RGBA array. Each mask is a separate `uint8[height,width]` array with values 0–255. Parts reference stable UUIDs; names and types are editable. The list order is bottom to top, serialized as contiguous `z_order` integers. The alpha of an exported layer is source alpha multiplied by mask/255. Compositing uses source-over and preserves straight RGB for nontransparent pixels; RGB values under fully transparent pixels have no visual meaning.

There is no generation of new artwork in this slice. Original ICC metadata is retained, not converted to sRGB. Color-managed CMYK import and a validated sRGB/Cubism PSD pipeline are future work.

## Project archive, schema version 1

`.l2split` is a ZIP archive containing:

```text
manifest.json
source.png
masks/0.png
masks/1.png
...
```

The manifest records schema/application versions, canvas dimensions, original filename and file SHA-256, source asset path, part IDs, names, types, visibility, order, and mask paths. The original hash identifies the imported file, including its original metadata; it is not the hash of the normalized embedded PNG. The project is self-contained, with original ICC profile embedded in `source.png`. Loading does not extract ZIP paths onto the filesystem. Unknown schemas, missing assets, malformed order, duplicate identities, and canvas/mask mismatches are rejected. Empty masks are permitted in saved drafts.

Saving writes a temporary archive alongside the destination, flushes it, and replaces the project atomically. A failed replacement leaves the previous archive intact. The imported original path is held only in memory for overwrite protection, not embedded as a machine-specific absolute path.

PNG export validates names and masks, stages files, then reserves a new destination directory. Existing destinations are rejected. A failed transfer removes only the new destination created by that export. Files use a sanitized name plus numeric order to avoid path traversal and filename collisions. The manifest preserves original names/IDs and records bounds as exclusive `(left, top, right, bottom)`. Every part is exported even if hidden; visibility controls the preview and is recorded for consumers.

## History and current limitations

Undo/redo holds up to 50 snapshots of editable masks and part metadata, sharing the immutable source. One brush stroke is one transaction. History is session-only; save/load retains the resulting editable state, not the undo timeline.

No semantic segmentation, hidden-region inpainting, generated-pixel provenance, grouping, side metadata, mask merge/split, PSD, or rigging is implemented. The alpha proposal must be inspected and explicitly accepted; rejection leaves manual edits unchanged. Import/save/export currently run synchronously, so large images or many layers may pause the UI. Large-scale memory optimization and cancellable background operations remain future work.

Tests use deterministic synthetic images for domain logic, real Qt mouse events for the edit path, and the generated original character for a representative file roundtrip. Headless GUI tests verify widgets and editing behavior; interactive desktop usability and Windows/macOS packaging need separate validation.
