# AGENTS.md

## Project

This repository contains a semi-automatic Live2D illustration parts-separation tool.

Read `docs/SPEC.md` before making architectural or product decisions.

The primary goal is to reduce manual Live2D material-separation work while keeping the workflow editable by a human.

This is NOT a project for fully automatic Live2D model generation.

---

# Core Principle

Prefer:

> AI-assisted + human-editable

over:

> fully automatic + difficult to correct

Every automatic result must remain inspectable and replaceable wherever reasonably possible.

---

# Source of Truth

Use these files in this order:

1. `docs/SPEC.md`
2. `docs/ARCHITECTURE.md`
3. ADRs under `docs/adr/`
4. existing code and tests

Do not treat external chat conversations as the authoritative specification once the requirement has been recorded in the repository.

If documentation and implementation disagree, explicitly identify the discrepancy before making a large architectural change.

---

# Development Strategy

Implement the smallest working vertical slice first.

Do NOT begin by integrating a large AI model.

Preferred order:

1. project structure
2. core data model
3. image import
4. parts model
5. mask representation
6. manual mask editing
7. compositing
8. project save/load
9. PNG export
10. inference interfaces
11. simple inference backend
12. advanced AI models
13. PSD export

The product must remain useful even when AI inference is unavailable.

---

# Architecture Rules

Keep these areas separated:

```text
UI
↓
Application services
↓
Core/domain
↓
Inference / Export / Infrastructure adapters
```

Core code must not import UI modules.

Core data models must not depend directly on a specific AI model.

Do not let a segmentation model define the project data format.

Use backend abstractions for AI functionality.

Example:

```python
class SegmentationBackend(Protocol):
    def segment(self, image, request):
        ...
```

Example:

```python
class InpaintingBackend(Protocol):
    def inpaint(self, image, mask, context):
        ...
```

Model-specific code belongs in adapters.

---

# Preferred Stack

Unless an ADR changes this decision:

- Python 3.12+
- PySide6
- NumPy
- Pillow
- OpenCV
- Pydantic
- pytest

PyTorch may be used by inference adapters.

Do not make PyTorch a dependency of the core domain layer.

---

# Dependency Policy

Before adding a dependency, consider:

- license
- maintenance status
- binary size
- Windows support
- macOS support
- Python 3.12+ support
- GPU requirements
- commercial-use restrictions

Do not add a large dependency for a trivial utility function.

Do not download model weights automatically during import, installation, or application startup.

Model downloads must be explicit.

---

# File Safety

Never overwrite the original illustration.

Generated data must go into project-managed or export directories.

Use atomic writes where practical for project metadata.

A failed save must not destroy the previous valid project.

---

# Image Coordinate System

Establish one canonical coordinate system and document it.

Masks, parts, exports, previews, and AI results must be convertible to that same coordinate system.

Avoid silent resizing.

Any transformation must record:

- source size
- destination size
- scale
- offset

Do not allow inference preprocessing to leak resized coordinates into project state.

---

# Parts

Every part must have a stable internal ID.

Display names are editable and must not be used as primary keys.

Example:

```json
{
  "id": "stable-generated-id",
  "name": "eye_white_L",
  "type": "eye_white",
  "side": "left"
}
```

Renaming a layer must not break project references.

---

# Masks

Masks must be editable independently of the source image.

Prefer a representation that supports:

- deterministic serialization
- undo/redo
- partial replacement
- bounding boxes
- efficient preview generation

Do not destructively bake a mask into the only copy of image pixels.

---

# Visible vs Generated Pixels

Distinguish original visible pixels from generated hidden-region pixels when feasible.

Never silently replace visible source artwork with AI-generated pixels.

AI generation should primarily be used to fill areas that do not exist in the original image because of occlusion.

If visible pixels must be modified, the operation must be explicit and reversible.

---

# AI Behavior

Treat model output as a proposal.

Do not assume segmentation is correct.

Do not assume semantic labels are correct.

Do not assume z-order is correct.

Do not assume inpainting is correct.

Expose confidence or uncertainty when the backend provides it.

Keep manual override paths.

---

# Privacy

User artwork is private project data.

Do not:

- upload images without explicit user action
- retain images for training
- add telemetry that sends artwork
- send masks or crops to third-party APIs silently

Any remote inference integration must clearly identify itself as remote.

---

# PSD / Live2D Constraints

The target is compatibility with Live2D Cubism.

PSD export should account for:

- RGB
- 8 bit/channel
- sRGB
- unique layer names
- independent part layers
- stable hierarchy
- compatible blend behavior
- avoidance of unsupported layer-mask assumptions

Keep PSD export behind an exporter abstraction.

Do not couple project state to one PSD-writing library.

Validate produced PSDs against actual Cubism behavior before declaring PSD export complete.

---

# UI Policy

Prioritize editing speed over visual decoration.

Important interactions:

- zoom
- pan
- brush add
- brush erase
- undo
- redo
- hide/show
- solo
- layer reorder
- source/reconstruction toggle

Keyboard shortcuts should eventually exist for frequent operations.

Long-running inference must not freeze the UI thread.

---

# Long-running Work

Inference and export tasks should expose:

- progress where possible
- cancellation where possible
- failure state
- actionable error information

A failed inference task must not invalidate existing manual work.

---

# Performance

Do not prematurely optimize.

However, avoid repeatedly copying full-resolution images during interactive editing when a cheaper representation is practical.

Use lower-resolution previews when appropriate, while preserving full-resolution source/export data.

---

# Testing

Add tests for core logic.

At minimum, cover:

- project serialization
- part IDs
- layer ordering
- mask bounds
- compositing
- validation
- exporter metadata

Regression tests should be added when fixing bugs.

AI model quality tests should be separated from deterministic unit tests.

Tests must not require downloading multi-gigabyte model weights.

---

# Fixtures

Small synthetic images should be used for most automated tests.

Do not commit copyrighted character artwork merely as a convenient test fixture.

Any real illustration fixture must have clear permission for repository use.

---

# Changes

Keep changes focused.

Do not combine:

- unrelated refactors
- dependency upgrades
- UI redesign
- model replacement

into a single task unless necessary.

For large work, split changes into independently testable steps.

---

# Refactoring

Do not refactor working modules solely for stylistic preference while implementing an unrelated feature.

Refactor when it:

- enables the requested feature
- fixes a demonstrated design issue
- reduces clear technical risk

Explain significant architectural changes.

---

# Documentation

Update relevant documentation when introducing:

- new project-file schema
- new inference backend
- new exporter
- new required dependency
- architectural change
- user-visible workflow change

Architecture decisions with long-term impact should be recorded under:

```text
docs/adr/
```

---

# Error Handling

Errors shown to users should explain:

1. what failed
2. what data was affected
3. whether their project is safe
4. what they can do next

Do not expose only raw Python tracebacks as user-facing errors.

Tracebacks may be recorded in logs.

---

# Logging

Never log full image data.

Avoid logging sensitive absolute paths unless required for local diagnostics.

Logs should focus on:

- operation
- backend
- duration
- dimensions
- result status
- exception information

---

# MVP Discipline

Do not implement post-MVP functionality merely because it is technically interesting.

Before implementing something substantial, verify that it contributes directly to the current milestone.

In particular, avoid prematurely implementing:

- automatic Live2D rigging
- motion generation
- facial tracking
- VTube Studio integration
- automatic ArtMesh generation
- cloud accounts
- collaboration systems
- training pipelines

unless the specification is explicitly updated.

---

# When Requirements Are Ambiguous

Choose the smallest reversible design.

Do not invent large product requirements.

Record important assumptions.

If a decision is expensive to reverse, create an ADR before committing to it.

---

# Definition of Done

A task is complete only when:

- implementation is finished
- relevant tests pass
- error handling is reasonable
- documentation is updated when needed
- no unrelated files were modified
- user data remains non-destructive
- manual workflow still works without AI where applicable