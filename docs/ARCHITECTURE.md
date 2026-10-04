# Editable Web characters, inference, and export

The current goal is an animated character that can be embedded in a Web application. A local browser editor reuses manual editing, color-region proposals, optional GPT proposals and atlas matching, and local neural repair. Its primary export is a standalone Web model with transparent textures and a Canvas2D runtime. Desktop editing, PNG material packages and PSD remain optional adapters. AI output requires review; numerical and playback checks do not certify material quality. Decisions are recorded in [ADR 0001](adr/0001-local-inference-and-psd.md), [ADR 0002](adr/0002-atlas-alignment.md), and [ADR 0003](adr/0003-embeddable-web-character.md).

## Boundaries

- `core.py`: project/part data, validation, mask bounds, RGBA compositing. No UI or model imports.
- `application.py`: editing transactions, undo/redo, accepting proposals.
- `infrastructure.py`: Pillow import, project archives, PNG export.
- `inference.py`: backend protocol and an explicit nontransparent-pixel baseline.
- `inpainting.py`: local LaMa and non-AI Telea adapters with typed repair proposals.
- `models.py`: explicit, checksum-verified acquisition of the optional checkpoint.
- `gpt_parts.py`: opt-in remote GPT classification/boxes and local GrabCut refinement; no core model dependency.
- `alignment.py`: opt-in GPT atlas correspondences, validation, local extraction and similarity transforms, and reversible manual placement corrections.
- `exporters.py`: exporter protocol and an sRGB PSD adapter.
- `ui.py`: PySide6 desktop widgets, dialogs, coordinate conversion, rendering.
- `web.py`: local FastAPI editing/session/job adapter and model/sprite endpoints, without domain dependence on HTTP.
- `web_export.py`: stable-ID role bindings, runtime model descriptor, cropped sRGB sprites and atomic standalone ZIP export.
- `web_static/`: HTML/CSS/JavaScript editor and dependency-free Canvas2D character runtime.

Python 3.12+, NumPy, Pillow, headless OpenCV, PySide6, and psd-tools are runtime dependencies. The optional `web` extra supplies FastAPI, Uvicorn and python-multipart; `ai` supplies PyTorch. Neither web dependencies nor PyTorch are imported by the domain. pytest/httpx and Playwright are development tools. The browser runtime needs no Node build, external SDK or CDN. OpenCV provides brush lines, polygon filling, color-space conversion and Telea repair. Python dependencies and supported platform wheels are locked in `uv.lock`; browser automation is pinned separately in `package-lock.json`.

## Coordinates and pixels

The canonical canvas is the imported image after EXIF orientation normalization. Pixel origin is top-left, x points right, y points down. There is no resize during import, editing, save, or export. Qt view transforms affect display only; pointer coordinates are mapped back into canvas coordinates before painting. Brush radius is in original pixels, independent of zoom. Lasso is a freehand polygon closed on mouse release.

Source pixels are an immutable `uint8[height,width,4]` RGBA array. Each mask is a separate `uint8[height,width]` array with values 0–255. Parts reference stable UUIDs; names and types are editable. The list order is bottom to top, serialized as contiguous `z_order` integers. A normal source part uses the project source pixels; an explicitly imported atlas part uses its own immutable canvas-sized `artwork`. The alpha of an exported layer is that part's artwork alpha multiplied by mask/255. Compositing uses source-over and preserves straight RGB for nontransparent pixels; RGB values under fully transparent pixels have no visual meaning.

Each part also has a requested hidden-region mask, immutable accepted generated RGBA pixels, and an immutable generated-region mask. Inference targets must not overlap that part's visible mask. Its visible artwork always takes priority during composition; generated pixels only supply the currently non-visible portion. Source-image pixels, explicitly imported atlas artwork, and accepted repair pixels have distinct provenance. Adding atlas artwork is an explicit, reversible operation and never rewrites the project source. Metadata and assets are preserved separately to make repairs inspectable and reversible.

An atlas part additionally retains the unmodified cropped RGBA `asset`, independent extraction `asset_mask`, and `alignment` metadata. `Project.assets` holds immutable full atlases keyed by SHA-256. The canonical output canvas remains the completed reference, even when the atlas has a different size. Metadata records full atlas and destination dimensions, source crop, normalized GPT coordinates, paired full-image pixel anchors, API preprocessing sizes/scales, cropped-asset-to-canvas 2×3 matrix, scale, rotation, offset, confidence, notes, warnings, anchor residuals, and transformed crop bounds. Crop offsets are subtracted before fitting; resized API coordinates do not become project coordinates. A transformed crop extending beyond the reference canvas is reported as a clipping warning.

GPT sends only correspondences. Local least-squares fitting produces a similarity transform with uniform scale, rotation and translation, without reflection or shape deformation. Extraction uses existing alpha by default. Explicit `white` mode removes near-white pixels connected to the crop border; this heuristic can remove white artwork or leave enclosed background. Pixels are warped with premultiplied alpha and then restored to straight RGBA to avoid dark interpolation fringes. All original arrays are retained.

Manual placement corrections use a stable pivot: the bounds center of the original extracted asset is mapped by its current matrix into the canonical canvas. They compose another similarity transform around that pivot and render once from the original crop, avoiding cumulative image resampling or drift from changing raster bounds. Edited masks retain an immutable canvas-sized `alignment_edit_mask` and its edit-time `alignment.edit_mask_matrix`; subsequent corrections warp that canonical mask rather than repeatedly resampling the previous mask. A new brush/lasso edit becomes a new canonical mask when the next placement correction is applied. A part with repair assets or a nonempty requested hidden mask cannot be moved this way; placement must be completed before repair. Manual corrections and mask edits remain undoable. Similarity transforms cannot correct mismatched anatomy, different poses, or perspective differences, and alignment does not generate pixels.

The GUI and alignment CLI check color profiles before any upload. Reference and atlas must have identical ICC profile bytes, or both be untagged. A mismatch is rejected with instructions to retain the originals and export both images using the same sRGB profile. This conservative check avoids interpreting atlas RGB under a different reference profile without silently altering imported pixels.

Original ICC metadata is retained for PNG material packages. PSD and Web sprite adapters convert profile-tagged pixels to sRGB; untagged RGB is assumed sRGB. Color-managed CMYK import remains future work. Compatibility of optional PSD exports with a specific desktop application is checked separately from the Web character workflow.

## Web model and rendering

`web_export.py` derives a version-1 `live2d-semi-auto-web-character` model independently of project schema 3. Each nonempty layer has a stable ID, editable role, visibility/order, relative image path and exclusive canvas-space bounds. Textures contain only nontransparent layer bounds, retaining placement in the original canonical canvas. Visible artwork and accepted repair pixels use the same composition rules as existing exports. Original image and mask arrays are unchanged.

Bindings override name-derived roles by stable part ID. Optional per-part pivots, head/body pivots, grid resolution and motion strength are normalized and validated. This rig configuration is saved in project history with `operation: web-runtime-settings`, so existing project save/load and Undo/Redo preserve it without a new archive schema. Live slider values, selected expressions/motions and playback toggles are preview state rather than a complete serialized playback session. Deleted or missing IDs cannot silently retain valid bindings. Empty layers are listed as omitted in the runtime model; an export must contain at least one visible, nonempty layer.

`web_static/runtime.js` exposes `window.Live2DWeb.Character`. It loads cropped images, validates bounds and image dimensions, and renders regular-grid textured triangles on Canvas2D. Head/body group transforms combine face angles with local eye/iris/brow/mouth movement, breathing and secondary hair motion. Expressions are parameter presets with procedural blush/tear overlays; idle, nod, shake and greeting motions are time-driven presets. Public setters control expression, motion, parameters, auto-blink, idle movement and pointer tracking. Start/stop/destroy manage animation frames and event listeners. Missing material is not synthesized by the runtime.

Unchanged and affine-deformed layers are drawn once with a direct image transform; static or identity playback does not accumulate triangle seams. Nonlinear deformations assemble each layer in a temporary canvas with additive premultiplied-alpha coverage at complementary triangle clip edges, then composite that layer once onto the stage. This preserves translucent-edge coverage without overlapping every triangle or using source-over at internal mesh boundaries.

Standalone ZIPs contain `model.json`, `runtime.js`, `index.html`, `README.md` and transparent `parts/*.png`. Stable IDs determine safe texture filenames. The example embeds HTML-safe JSON and uses relative image/runtime paths, avoiding a JSON fetch requirement for browsers that permit local-file playback. Any static HTTP server can serve the example independently of the editor. Hosting the same model allows direct JavaScript integration through Canvas plus a model object; `baseUrl` or per-ID `assetUrls` resolve texture locations. Export stages a complete ZIP and publishes to a new filename atomically. API keys, server sessions, editor endpoints and original absolute paths do not belong to the playback bundle.

## Local Web editing and jobs

The server listens on `127.0.0.1:8080` by default and serves same-origin editor assets. Artwork first enters the local editor through an explicit upload. Sessions use random identities, per-session locks, bounded counts and a TTL; project edits live in memory until downloaded. This is a local/private editor adapter, not a public hosting or authentication service. A restart discards unsaved sessions, while exported models can run on an arbitrary static host.

Remote GPT jobs require consent before sending the selected artwork to `api.openai.com`. A password input supplies an ephemeral key or the server uses an injected environment key. Credentials do not enter saved state, runtime descriptors, bundles or logs. API/model errors are sanitized. Thread-pool jobs retain revision identity and reviewable candidates; acceptance checks the originating revision and adds parts or repairs transactionally. Cancellation discards the candidate and may cancel queued work, but cannot recall an API request or interrupt an already-running neural kernel. Other sessions remain isolated.

The browser editor calls existing application services for masks, order, names, visibility, undo/redo, proposals and repair. It fetches original/recomposition/difference/layer images for material editing and the runtime descriptor/sprites for animation. Roles and rig configuration remain editable before exporting. Downloads produce project, PNG material, PSD or Web ZIP outputs. Resource limits are 32 MiB per upload, 4096 px per image dimension, 128 parts and a stored-image pixel budget of `64 × 1024²`. The pixel budget counts source, full atlases and all part image/mask arrays, including shared immutable arrays as persisted assets. The same budget applies to loading and candidate/edit operations so downloadable work remains reloadable; excess operations fail before replacing current state. Errors advise smaller copies or fewer candidates while retaining manual work.

## Project archive, schema version 3

`.l2split` is a ZIP archive containing:

```text
manifest.json
source.png
masks/0.png
masks/1.png
hidden_mask/0.png       # optional requested repair area
generated/0.png         # optional accepted pixels
generated_mask/0.png    # optional accepted provenance
artwork/0.png           # optional atlas artwork in canonical canvas
asset/0.png             # optional unmodified atlas crop
asset_mask/0.png        # optional extraction mask
alignment_edit_mask/0.png # optional canonical manually edited canvas mask
assets/<sha256>.png     # original full atlas, deduplicated by identity
...
```

The manifest records schema/application versions, canvas dimensions, original filename and file SHA-256, source asset path, part IDs, names, types, visibility, order, mask paths, optional generation and alignment assets, canonical edited-mask paths and origin matrices, atlas references, alignment metadata, and operation history. History includes backend, model identity/checksum and inference crop/resize/padding transforms. Schema 1 and 2 archives remain readable. The original hash identifies the imported file, including its original metadata; it is not the hash of the normalized embedded PNG. Atlas identifiers similarly refer to imported files; the archive keeps their normalized full-resolution RGBA pixels. The project is self-contained, with original ICC profile embedded in `source.png`. Loading does not extract ZIP paths onto the filesystem. Unknown schemas, missing assets, malformed order, duplicate identities, and canvas/mask mismatches are rejected. Empty masks are permitted in saved drafts.

Saving writes a temporary archive alongside the destination, flushes it, and replaces the project atomically. A failed replacement leaves the previous archive intact. The imported original path is held only in memory for overwrite protection, not embedded as a machine-specific absolute path.

PNG export validates names and masks, stages files, then reserves a new destination directory. Existing destinations are rejected. A failed transfer removes only the new destination created by that export. Files use a sanitized name plus numeric order to avoid path traversal and filename collisions. The manifest preserves original names/IDs, operation history, generated-mask paths and visible-mask bounds as exclusive `(left, top, right, bottom)`. Every part is exported even if hidden; visibility controls the preview and is recorded for consumers. A part can be exported with only generated coverage. PSD export writes to a verified complete temporary file and publishes it without replacing existing destinations.

## History and current limitations

Undo/redo holds up to 50 snapshots of editable masks and part metadata, sharing the immutable source, atlases, crops, aligned artwork, canonical edited masks and accepted generation arrays. One brush stroke is one transaction. Undo history is session-only; save/load retains the resulting editable state and generation/alignment provenance, not the undo timeline.

Local color separation does not classify hair/face. Optional GPT proposals supply semantic names/boxes, refined with GrabCut, and atlas correspondences for similarity alignment. API access and numerical correctness are separate from model quality: names, masks, anchors, order, and genuinely hidden anatomy still require human inspection. Anatomical hidden-region detection, general-purpose grouping, side metadata and mask merge/split remain unimplemented. All inference proposals must be inspected and explicitly accepted; rejection leaves manual edits unchanged. GPT additionally requires explicit artwork-upload confirmation. Atlas alignment uploads both the completed reference and atlas, provides comparison previews, and appends accepted parts instead of replacing existing manual parts. Desktop GUI inference and PNG/PSD export run in worker threads; desktop import and project save/load remain synchronous. Web inference uses bounded background jobs. Large-scale memory optimization remains future work.

Tests use deterministic synthetic images for domain logic, real Qt mouse events for the desktop edit path, and the generated original character for representative file roundtrips. HTTP tests exercise sessions, consent, project edits, candidate safety and export contents. Real-browser automation exercises editing, role persistence, expressions/motions, parameter-driven frame changes and standalone exported playback separately from the editor server. Reports and screenshots describe observed behavior; playback does not repair inconsistent atlas geometry or missing body parts. Interactive desktop usability and Windows/macOS packaging need separate validation.

Browser evidence is stored under [samples/results/web](../samples/results/web), including the report, downloaded model, independent example, screenshots and animation recording. Independent HTTP playback passed. This cloud Chromium's managed policy rejects `file://` with `ERR_BLOCKED_BY_ADMINISTRATOR`; the optional local-file check is recorded as skipped without bypassing that policy. Unit checks and browser artifacts are evaluated separately from the generated material's visual quality.
