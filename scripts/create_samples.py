"""Explicitly reproduce sample data; requires the pinned local LaMa model."""

import argparse
import hashlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from PIL import Image
from psd_tools import PSDImage

from live2d_semi_auto.application import Editor
from live2d_semi_auto.core import composite, layer_pixels
from live2d_semi_auto.exporters import PsdExporter
from live2d_semi_auto.inference import ColorPartsBackend
from live2d_semi_auto.inpainting import LaMaBackend
from live2d_semi_auto.infrastructure import export_png, import_image, load_project, save_project


def write_image(path, pixels):
    Image.fromarray(pixels).save(path)


def create_samples(source_path: Path, output: Path, model: Path):
    if output.exists():
        raise ValueError("既存の成果物を守るため、新しい出力先を指定してください。")
    output.mkdir(parents=True)
    original = source_path.read_bytes()
    project = import_image(source_path)
    editor = Editor(project)
    proposal = ColorPartsBackend(8).propose_parts(project.source)
    editor.accept_parts(proposal)
    project = editor.project
    for index, part in enumerate(project.parts):
        part.id = uuid5(NAMESPACE_URL, f"{project.source_hash}:color:{index}").hex
    split = output / "automatic"
    split.mkdir()
    save_project(project, split / "project.l2split")
    export_png(project, split / "png")
    PsdExporter().export(project, split / "parts.psd")
    palette = np.array([[50, 180, 230], [230, 70, 90], [90, 210, 80], [210, 90, 220],
                        [230, 180, 30], [90, 90, 230], [30, 200, 180], [180, 120, 70]])
    overlay = project.source.copy()
    for index, part in enumerate(project.parts):
        overlay[part.mask > 0, :3] = np.rint(
            overlay[part.mask > 0, :3] * 0.35 + palette[index] * 0.65).astype(np.uint8)
        write_image(split / f"mask_{index + 1:02d}.png", part.mask)
    write_image(split / "segmentation_overlay.png", overlay)
    restored = load_project(split / "project.l2split")
    assert np.array_equal(composite(restored), project.source)
    psd = PSDImage.open(split / "parts.psd")
    assert np.array_equal(np.asarray(psd.composite().convert("RGBA")), project.source)

    # Synthetic occlusion: withhold a known hair patch, then predict its pixels.
    repair = output / "inpainting"
    repair.mkdir()
    project = import_image(source_path)
    editor = Editor(project)
    part = editor.add_part()
    part.name, part.kind = "synthetic_occlusion_demo", "demo"
    part.id = uuid5(NAMESPACE_URL, f"{project.source_hash}:repair").hex
    part.mask[:] = 255
    part.hidden_mask = np.zeros_like(part.mask)
    part.hidden_mask[180:280, 500:600] = 255
    part.mask[part.hidden_mask > 0] = 0
    write_image(repair / "target_mask.png", part.hidden_mask)
    write_image(repair / "before.png", layer_pixels(project, part))
    save_project(project, repair / "before.l2split")
    result = LaMaBackend(model).propose(project.source, editor.repair_mask(0))
    editor.accept_repair(0, result)
    write_image(repair / "after.png", layer_pixels(project, part))
    write_image(repair / "generated_mask.png", part.generated_mask)
    save_project(project, repair / "project.l2split")
    export_png(project, repair / "png")
    PsdExporter().export(project, repair / "parts.psd")
    loaded = load_project(repair / "project.l2split")
    assert np.array_equal(layer_pixels(loaded, loaded.parts[0]), layer_pixels(project, part))
    assert np.array_equal(layer_pixels(project, part)[part.mask > 0], project.source[part.mask > 0])
    assert source_path.read_bytes() == original
    report = {
        "source": source_path.name, "source_sha256": hashlib.sha256(original).hexdigest(),
        "automatic": proposal.metadata, "inpainting": result.metadata,
        "inpainting_purpose": "Synthetic occlusion of a visible hair patch; not ground-truth hidden anatomy",
        "checks": {"split_reconstruction_exact": True, "psd_reconstruction_exact": True,
                   "repair_visible_source_preserved": True, "project_roundtrips": True,
                   "cubism_validation": "not_run"},
    }
    report["artifacts_sha256"] = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report["checks"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("samples/original_character.png"))
    parser.add_argument("--output", type=Path, default=Path("samples/results"))
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    create_samples(args.source, args.output, args.model)
