# Editable, embeddable Web character runtime

Status: accepted for the user's Web character goal on 2026-10-05.

## Context

The goal is a character that animates inside a Web application. The earlier material-separation and PSD milestone supplies useful editing and image adapters, but Cubism compatibility is not the completion criterion. The user prioritizes finished-character expression and motion playback in a browser.

## Decision

Reuse the existing domain and editing services from a local FastAPI adapter. Serve a dependency-free HTML/JavaScript editor with explicit inference previews, editable masks and part roles. Keep desktop functionality available. FastAPI, Uvicorn and python-multipart are small, maintained permissively licensed optional Web dependencies with CPU-only, Python 3.12 and major desktop-platform support. Browser automation is development tooling; application rendering needs no Node build or external CDN.

A separate Canvas2D runtime loads cropped transparent sprites and a model descriptor with stable layer IDs, source-canvas bounds, roles and group pivots. Regular-grid textured triangle deformation, head/body transforms, eye/mouth controls, breathing and secondary hair motion provide editable 2D animation. Expressions and procedural motion presets are driven by public parameters. Source artwork is unchanged. Missing body parts and occluded artwork remain material-editing concerns rather than invisible synthesis by the renderer.

Web bundle export includes the runtime, model descriptor, sprites and a standalone example. It runs independently of the Python editor server and exposes a public JavaScript embedding interface. Existing PNG, project and PSD outputs remain optional additional formats. No vendor SDK, proprietary compiled model or account service is required.

Editor sessions and jobs are bounded and isolated. Artwork uploads target the local editor by default. Remote inference requires explicit consent identifying the destination, image contents and cost. Credentials remain ephemeral or injected into the server environment; they do not enter the model, project, exported bundle or logs. Long jobs produce reviewable candidates and reject stale results after state changes. Cancellation discards results without pretending to recall an API request.

## Validation

Exercise actual browser import, editing, undo/redo, candidate adoption/rejection, project roundtrip and downloads. Verify that parameter changes alter rendered frames and that expression/motion playback works. Extract the Web bundle and run its standalone example separately from the editor. Report actual animation and export checks, without treating them as certification for an unrelated runtime.
