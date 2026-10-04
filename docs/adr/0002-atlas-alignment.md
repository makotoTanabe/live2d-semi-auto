# AI correspondences and reversible atlas alignment

Status: accepted for the user's requested completed-reference / parts-sheet workflow.

## Context

Generating isolated parts for a new character can reduce separation and hidden-region painting. A presentation sheet still has mixed positions, scales and sometimes poses, so cropping it alone does not create consistent Live2D layers. The application must preserve the reference, the atlas and human corrections, remain useful without remote inference, and avoid coupling project state to one GPT response format.

## Decision

An optional remote `GPTAlignmentBackend` proposes atlas crops, semantic names, bottom-to-top order and 2–8 paired anatomical anchors against a completed reference. All coordinates are normalized independently to each full image. The existing GPT transport is reused, including sanitized API failures and a user-supplied model name. The GUI requires explicit confirmation before sending both images, followed by a separate preview and adoption decision. A rejected or failed proposal does not invalidate existing manual parts.

Pixel extraction and geometry run locally. Uniform scale, rotation and translation are fitted by least squares, without reflection. Crop offsets and API preprocessing sizes/scales are recorded so inference coordinates never leak into the canonical reference canvas. Alpha is preferred; border-connected near-white removal is an explicit heuristic for opaque sheets. Premultiplied-alpha warping avoids dark transparent edges. Transformed crops outside the canonical canvas produce clipping warnings. No image synthesis, perspective warp, anatomical deformation or rigging is introduced by alignment.

Before upload, GUI and CLI reject reference/atlas ICC mismatches. Exact profile bytes must match, or both images must be untagged. The actionable error asks for new copies exported under the same sRGB profile while keeping the originals. This conservative rule avoids silently interpreting imported RGB under another asset's profile.

Each accepted atlas part stores immutable canvas-sized `artwork`, an unmodified cropped `asset`, an independent `asset_mask`, and `alignment` metadata. The project stores immutable full atlases keyed by SHA-256. Metadata includes original source/destination sizes, crop, anchors, matrix, scale, rotation, offsets, model, confidence, notes, warnings and residuals. Existing masks, names, types, stable IDs, order and visibility remain editable. Imported artwork is explicitly distinct from source-image pixels and generated repair pixels; visible artwork takes precedence over repair pixels.

Schema 3 archives retain these assets and remain self-contained; schema 1 and 2 loading remains supported. Original input files are never overwritten. Undo/redo shares immutable image arrays and snapshots editable masks and metadata. Manual placement corrections use a stable pivot derived from original extracted-asset bounds and mapped by the current matrix. They compose a transform and render once from the original crop rather than repeatedly resampling the previous output. A manually edited mask is retained as immutable canvas-sized `alignment_edit_mask` with its edit-time `alignment.edit_mask_matrix`; adjustments warp this canonical mask, and a subsequent brush/lasso edit establishes a new canonical mask on the next adjustment. These assets are persisted as optional schema 3 fields. Placement corrections are disallowed when a nonempty requested hidden mask or accepted repair assets exist, because moving only the imported artwork would detach repairs from their target; adjust placement before repairing.

PNG and PSD adapters continue to export independent canonical-canvas layers through the existing interfaces. Cubism desktop compatibility still requires an actual import check.

## Consequences

The user can inspect and correct every accepted placement, keep existing work, and replay saved GPT responses without a network request. Similarity transforms are deliberately limited: a different pose, perspective or inconsistent generated shape cannot be corrected by location and scale alone. Confidence and small anchor residuals are useful diagnostics, not quality guarantees. White removal may remove white artwork, crop proposals may contain neighbouring parts, and adding overlapping layers may change translucent alpha. Recomposition, difference images, mask coverage, per-layer inspection and hide/move checks remain necessary.

A separate CLI sample runner saves the atlas, response, masks, editable project, PNG/PSD layers and numerical verification artifacts after explicit live upload or offline replay. Keys are supplied through environment bindings or hidden input and never stored in project or sample data. Actual API/model quality checks remain separate from deterministic serialization, compositing and transform tests.
