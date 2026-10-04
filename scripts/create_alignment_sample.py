"""Explicit dual-image GPT alignment, with offline response reproduction."""

import argparse
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from PIL import Image, ImageDraw, ImageCms
from psd_tools import PSDImage

from live2d_semi_auto.alignment import GPTAlignmentBackend
from live2d_semi_auto.application import Editor
from live2d_semi_auto.core import composite, layer_pixels, mask_bounds
from live2d_semi_auto.exporters import PsdExporter
from live2d_semi_auto.infrastructure import export_png, import_image, load_project, save_project, require_matching_profiles


def main():
    parser = argparse.ArgumentParser(description="GPT atlas alignment; explicit upload or offline replay")
    parser.add_argument("--source", type=Path, default=Path("samples/original_character.png"))
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--background", choices=["alpha", "white"], default="alpha")
    parser.add_argument("--upload", action="store_true", help="Authorize sending BOTH images to OpenAI")
    parser.add_argument("--response", type=Path, help="Replay a response offline; no network or key needed")
    parser.add_argument("--response-log", type=Path, help="Save the raw successful API response, including rejected proposals, to a NEW file")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory; existing samples are never overwritten")
    if args.response_log and args.response_log.exists():
        parser.error("Use a new response-log file to preserve earlier responses")
    if not args.response and not args.upload:
        parser.error("Use --upload for explicit artwork upload, or --response for offline reproduction")
    project, atlas = import_image(args.source), import_image(args.atlas)
    require_matching_profiles(project, atlas)
    received = []
    key = None
    if args.response:
        received.append(json.loads(args.response.read_text(encoding="utf-8")))
        backend = GPTAlignmentBackend(model=args.model, transport=lambda _: received[0])
    else:
        key = os.environ.get("GPT_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            key = getpass.getpass("GPT API key (hidden; not saved): ")
        backend = GPTAlignmentBackend(model=args.model, api_key=key)

        def transport(body):
            response = backend._request(body)
            received.append(response)
            if args.response_log:
                with args.response_log.open("x", encoding="utf-8") as stream:
                    json.dump(response, stream, ensure_ascii=False, indent=2)
            return response

        backend.transport = transport
    try:
        proposal = backend.propose_alignment(project.source, atlas.source, atlas_name=atlas.source_name,
                                             atlas_hash=atlas.source_hash, background_mode=args.background)
    finally:
        backend.api_key = None
        key = None
    response = received[0]
    for index, part in enumerate(proposal.parts):
        part.id = uuid5(NAMESPACE_URL, f"{project.source_hash}:{atlas.source_hash}:{response.get('id')}:{index}").hex
        proposal.metadata["parts"][index]["id"] = part.id
    proposal.metadata.update({"response_model": response.get("model", args.model),
                              "response_id": response.get("id"), "usage": response.get("usage")})
    editor = Editor(project)
    editor.accept_parts(proposal)
    project = editor.project
    # Nothing is written until a complete, validated proposal has been received.
    args.output.mkdir(parents=True)
    shutil.copyfile(args.source, args.output / ("reference_original" + args.source.suffix.lower()))
    shutil.copyfile(args.atlas, args.output / ("atlas_original" + args.atlas.suffix.lower()))
    (args.output / "response.json").write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    save_project(project, args.output / "project.l2split")
    export_png(project, args.output / "png")
    PsdExporter().export(project, args.output / "parts.psd")
    reconstructed = composite(project)
    Image.fromarray(reconstructed).save(args.output / "reconstructed.png")
    difference = np.abs(project.source.astype(np.int16) - reconstructed.astype(np.int16)).astype(np.uint8)
    difference[..., 3] = 255
    Image.fromarray(difference).save(args.output / "difference.png")
    comparison = Image.new("RGBA", (project.size[0] * 2, project.size[1]), (100, 100, 100, 255))
    comparison.alpha_composite(Image.fromarray(project.source), (0, 0))
    comparison.alpha_composite(Image.fromarray(reconstructed), (project.size[0], 0))
    comparison.save(args.output / "comparison.png")
    coverage = np.sum([part.mask > 0 for part in project.parts], axis=0)
    Image.fromarray(np.clip(coverage * 40, 0, 255).astype(np.uint8)).save(args.output / "coverage.png")
    part_checks = []
    for index, part in enumerate(project.parts):
        Image.fromarray(part.mask).save(args.output / f"mask_{index + 1:02d}.png")
        part_checks.append({"id": part.id, "name": part.name, "kind": part.kind,
                            "bounds": mask_bounds(part.mask), "pixels": int(np.count_nonzero(part.mask)),
                            "confidence": part.alignment["confidence"],
                            "anchor_residual_px": part.alignment["anchor_residual_px"],
                            "warnings": part.alignment["warnings"], "notes": part.alignment["notes"]})
    # A contact sheet exposes actual isolated layers, including background leakage.
    tile = 256
    columns = min(4, len(project.parts))
    rows = (len(project.parts) + columns - 1) // columns
    contact = Image.new("RGB", (columns * tile, rows * (tile + 28)), "#777777")
    draw = ImageDraw.Draw(contact)
    for index, part in enumerate(project.parts):
        image = Image.fromarray(layer_pixels(project, part))
        bounds = image.getbbox()
        if bounds:
            image = image.crop(bounds)
        image.thumbnail((tile, tile))
        x, y = (index % columns) * tile, (index // columns) * (tile + 28)
        contact.paste(image, (x + (tile - image.width) // 2, y), image)
        draw.text((x + 4, y + tile + 4), part.name[:32], fill="white")
    contact.save(args.output / "isolated_layers.png")
    loaded = load_project(args.output / "project.l2split")
    assert np.array_equal(composite(loaded), reconstructed)
    assert np.array_equal(loaded.assets[atlas.source_hash], atlas.source)
    assert [part.id for part in loaded.parts] == [part.id for part in project.parts]
    psd = PSDImage.open(args.output / "parts.psd")
    assert psd.size == project.size and psd.depth == 8
    assert [layer.name for layer in psd] == [part.name for part in project.parts]
    assert all(not layer.has_mask() for layer in psd)
    for layer, part in zip(psd, project.parts):
        actual, expected = np.array(layer.topil()), layer_pixels(project, part)
        if project.icc_profile:
            source_profile = ImageCms.ImageCmsProfile(io.BytesIO(project.icc_profile))
            srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
            expected = np.array(ImageCms.profileToProfile(Image.fromarray(expected), source_profile, srgb, outputMode="RGBA"))
        assert np.array_equal(actual[..., 3], expected[..., 3])
        assert np.array_equal(actual[..., :3][expected[..., 3] > 0], expected[..., :3][expected[..., 3] > 0])
    active = reconstructed[..., 3] > 0
    rgb_error = np.abs(project.source[..., :3].astype(np.int16) - reconstructed[..., :3].astype(np.int16))
    report = {"source_sha256": project.source_hash, "atlas_sha256": atlas.source_hash,
              "metadata": proposal.metadata, "parts": part_checks,
              "coverage": {"reference_opaque_pixels_without_aligned_coverage": int(np.count_nonzero((coverage == 0) & (project.source[..., 3] > 0))),
                           "overlapping_pixels": int(np.count_nonzero(coverage > 1)),
                           "mean_rgb_error_on_aligned_coverage": float(rgb_error[active].mean()),
                           "note": "Reference backgrounds count as uncovered. Differences and overlap are diagnostic, not semantic quality scores."},
              "checks": {"gpt_request": "replayed" if args.response else "live_success",
                         "project_roundtrip": True, "original_atlas_preserved": True,
                         "psd_independent_layer_pixels_and_alpha": True,
                         "alignment_quality": "requires_human_review", "cubism_import": "not_run"}}
    report["artifacts_sha256"] = {str(path.relative_to(args.output)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted(args.output.rglob("*")) if path.is_file()}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model": proposal.metadata["response_model"], "parts": part_checks,
                      "usage": response.get("usage"), "checks": report["checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
