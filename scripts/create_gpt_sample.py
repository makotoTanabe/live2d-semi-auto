"""Explicit GPT artwork upload, or offline reproduction from a saved response."""

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from PIL import Image
from psd_tools import PSDImage

from live2d_semi_auto.application import Editor
from live2d_semi_auto.core import composite, mask_bounds
from live2d_semi_auto.exporters import PsdExporter
from live2d_semi_auto.gpt_parts import GPTPartsBackend
from live2d_semi_auto.infrastructure import export_png, import_image, load_project, save_project


def main():
    parser = argparse.ArgumentParser(description="Explicit GPT sample upload; key is never stored")
    parser.add_argument("--source", type=Path, default=Path("samples/original_character.png"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--response", type=Path, help="Replay saved response offline, without an API call")
    parser.add_argument("--upload", action="store_true", help="Explicitly authorize sending artwork to OpenAI")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory to preserve existing artifacts")
    if not args.response and not args.upload:
        parser.error("Use --upload for an explicit API request, or --response to replay offline")
    project = import_image(args.source)
    received = []
    if args.response:
        received.append(json.loads(args.response.read_text()))
        backend = GPTPartsBackend(model=args.model, transport=lambda _: received[0])
    else:
        key = os.environ.get("GPT_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            key = getpass.getpass("GPT API key (hidden; used only for this request): ")
        backend = GPTPartsBackend(model=args.model, api_key=key)
        def transport(body):
            body["max_completion_tokens"] = 4096
            response = backend._request(body)
            received.append(response)
            return response
        backend.transport = transport
    proposal = backend.propose_parts(project.source)
    # No key is included in response, metadata, project, or sample output.
    backend.api_key = None
    response = received[0]
    proposal.metadata["response_model"] = response.get("model", args.model)
    proposal.metadata["response_id"] = response.get("id")
    proposal.metadata["usage"] = response.get("usage")
    editor = Editor(project)
    editor.accept_parts(proposal)
    project = editor.project
    for index, part in enumerate(project.parts):
        part.id = uuid5(NAMESPACE_URL, f"{project.source_hash}:gpt:{response.get('id')}:{index}").hex
    args.output.mkdir(parents=True)
    (args.output / "response.json").write_text(json.dumps(response, ensure_ascii=False, indent=2))
    save_project(project, args.output / "project.l2split")
    export_png(project, args.output / "png")
    PsdExporter().export(project, args.output / "parts.psd")
    overlay = project.source.copy()
    part_checks = []
    for index, part in enumerate(project.parts):
        Image.fromarray(part.mask).save(args.output / f"mask_{index + 1:02d}.png")
        color = np.array([(index * 79 + 50) % 256, (index * 131 + 90) % 256, (index * 193 + 130) % 256])
        overlay[part.mask > 0, :3] = np.rint(overlay[part.mask > 0, :3] * .35 + color * .65).astype(np.uint8)
        part_checks.append({"name": part.name, "kind": part.kind, "bounds": mask_bounds(part.mask),
                            "pixels": int(np.count_nonzero(part.mask))})
    Image.fromarray(overlay).save(args.output / "overlay.png")
    coverage = np.sum([part.mask > 0 for part in project.parts], axis=0)
    Image.fromarray(np.clip(coverage * 40, 0, 255).astype(np.uint8)).save(args.output / "coverage.png")
    loaded = load_project(args.output / "project.l2split")
    assert np.array_equal(composite(loaded), composite(project))
    psd = PSDImage.open(args.output / "parts.psd")
    assert [part.name for part in psd] == [part.name for part in project.parts]
    assert all(not part.has_mask() for part in psd)
    report = {"source_sha256": project.source_hash, "metadata": proposal.metadata,
              "parts": part_checks, "coverage": {
                  "uncovered_pixels": int(np.count_nonzero((coverage == 0) & (project.source[..., 3] > 0))),
                  "overlapping_pixels": int(np.count_nonzero(coverage > 1)),
              }, "checks": {"gpt_request": "replayed" if args.response else "live_success",
                            "project_roundtrip": True, "psd_layer_names_and_transparency": True,
                            "semantic_mask_quality": "requires_human_review", "cubism_import": "not_run"}}
    report["artifacts_sha256"] = {str(path.relative_to(args.output)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted(args.output.rglob("*")) if path.is_file()}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"model": proposal.metadata["response_model"], "parts": [p.name for p in project.parts],
                      "usage": response.get("usage"), "coverage": report["coverage"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
