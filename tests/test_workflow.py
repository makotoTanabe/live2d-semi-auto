import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from live2d_semi_auto.application import Editor
from live2d_semi_auto.core import composite, layer_pixels, mask_bounds, validate
from live2d_semi_auto.inference import AlphaBackend
from live2d_semi_auto.infrastructure import export_png, import_image, load_project, save_project


def populate(project):
    editor = Editor(project)
    left = editor.add_part()
    left.mask[:, :8] = 255
    right = editor.add_part()
    right.mask[:, 8:] = 255
    return editor


def test_import_preserves_original(project):
    original = Path(project.original_path).read_bytes()
    assert project.size == (16, 12)
    assert project.source_hash == hashlib.sha256(original).hexdigest()
    with pytest.raises(ValueError):
        project.source[:] = 0
    assert Path(project.original_path).read_bytes() == original


@pytest.mark.parametrize("format,suffix", [("PNG", ".png"), ("JPEG", ".jpg"), ("WEBP", ".webp")])
def test_supported_imports(tmp_path, format, suffix):
    path = tmp_path / ("image" + suffix)
    Image.new("RGB", (7, 9), "pink").save(path, format=format)
    project = import_image(path)
    assert project.size == (7, 9)
    assert np.all(project.source[..., 3] == 255)


def test_exif_orientation(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (7, 9), "pink").save(path, exif=exif)
    assert import_image(path).size == (9, 7)


def test_brush_stroke_undo_redo_and_bounds(project):
    editor = Editor(project)
    part = editor.add_part()
    identity = part.id
    editor.checkpoint()
    editor.paint(0, (-2, 3), (10, 3), 1)
    editor.paint(0, (10, 3), (12, 3), 1)
    painted = part.mask.copy()
    assert mask_bounds(painted) == (0, 1, 15, 6)
    assert painted[3, 0] == 255
    editor.undo()
    assert not np.any(editor.project.parts[0].mask)
    assert editor.project.parts[0].id == identity
    editor.redo()
    assert np.array_equal(editor.project.parts[0].mask, painted)
    editor.checkpoint()
    editor.paint(0, (4, 3), (4, 3), 1, erase=True)
    assert editor.project.parts[0].mask[3, 4] == 0
    editor.undo()
    assert np.array_equal(editor.project.parts[0].mask, painted)


def test_polygon_and_undo(project):
    editor = Editor(project)
    editor.add_part()
    editor.polygon(0, [(2, 2), (8, 2), (8, 8), (2, 8)])
    assert mask_bounds(editor.project.parts[0].mask) == (2, 2, 9, 9)
    editor.polygon(0, [(3, 3), (7, 3), (7, 7)], erase=True)
    assert editor.project.parts[0].mask[4, 5] == 0
    editor.undo()
    assert editor.project.parts[0].mask[4, 5] == 255


def test_identity_order_and_properties(project):
    editor = populate(project)
    ids = [p.id for p in project.parts]
    editor.update_part(0, "face", "face", False)
    assert project.parts[0].id == ids[0]
    assert not project.parts[0].visible
    assert editor.move(0, 1) == 1
    assert [p.id for p in project.parts] == ids[::-1]
    editor.undo()
    assert [p.id for p in editor.project.parts] == ids
    with pytest.raises(ValueError, match="同じ名前"):
        editor.update_part(1, "FACE", "face", True)
    editor.delete(0)
    editor.undo()
    assert len(editor.project.parts) == 2


def test_reconstruction_and_visibility(project):
    populate(project)
    assert np.array_equal(composite(project), project.source)
    project.parts[0].visible = False
    result = composite(project)
    assert np.all(result[:, :8, 3] == 0)
    assert np.array_equal(result[:, 8:], project.source[:, 8:])


def test_soft_alpha_compositing(project):
    editor = Editor(project)
    a, b = editor.add_part(), editor.add_part()
    a.mask[:] = 128
    b.mask[:] = 128
    pixels = composite(project)
    assert abs(int(pixels[2, 2, 3]) - 192) <= 1
    assert np.array_equal(pixels[2, 2, :3], project.source[2, 2, :3])
    assert layer_pixels(project, a)[4, 6, 3] == 64


def test_proposal_is_explicit_and_reversible(project):
    editor = Editor(project)
    editor.add_part()
    proposed = editor.proposal(AlphaBackend())
    assert not np.any(editor.project.parts[0].mask)
    editor.accept_mask(0, proposed)
    assert editor.project.parts[0].mask[4, 6] == 255
    assert np.array_equal(composite(editor.project), project.source)
    proposed[:] = 0
    assert editor.project.parts[0].mask[4, 6] == 255
    editor.undo()
    assert not np.any(editor.project.parts[0].mask)


def test_bad_proposal_preserves_edits(project):
    class Broken:
        def propose(self, source):
            return np.ones((3, 3), dtype=np.uint8)
    editor = populate(project)
    before = project.parts[0].mask.copy()
    with pytest.raises(ValueError):
        editor.proposal(Broken())
    assert np.array_equal(before, project.parts[0].mask)


def test_save_load_self_contained(project, tmp_path):
    editor = populate(project)
    editor.update_part(0, "顔", "face", False)
    target = tmp_path / "project.l2split"
    save_project(project, target)
    Path(project.original_path).unlink()
    loaded = load_project(target)
    assert loaded.size == project.size
    assert loaded.source_hash == project.source_hash
    assert loaded.source_name == project.source_name
    assert np.array_equal(loaded.source, project.source)
    assert not loaded.source.flags.writeable
    for before, after in zip(project.parts, loaded.parts, strict=True):
        assert (before.id, before.name, before.kind, before.visible) == (
            after.id, after.name, after.kind, after.visible)
        assert np.array_equal(before.mask, after.mask)
    save_project(loaded, target)
    assert np.array_equal(composite(load_project(target)), composite(project))


def test_failed_save_preserves_existing_archive(project, tmp_path, monkeypatch):
    populate(project)
    target = tmp_path / "project.l2split"
    save_project(project, target)
    original = target.read_bytes()
    def fail(*args):
        raise OSError("disk error")
    monkeypatch.setattr("live2d_semi_auto.infrastructure.os.replace", fail)
    with pytest.raises(OSError):
        save_project(project, target)
    assert target.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_no_source_overwrite(project):
    before = Path(project.original_path).read_bytes()
    with pytest.raises(ValueError):
        save_project(project, project.original_path)
    assert Path(project.original_path).read_bytes() == before


def test_save_empty_masks_allowed_export_rejected(project, tmp_path):
    Editor(project).add_part()
    save_project(project, tmp_path / "draft.l2split")
    with pytest.raises(ValueError, match="マスクが空"):
        export_png(project, tmp_path / "export")
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("problem", ["missing_mask", "order", "canvas", "version", "duplicate", "mask_size"])
def test_invalid_archives_rejected(project, tmp_path, problem):
    populate(project)
    path = tmp_path / "project.l2split"
    save_project(project, path)
    with zipfile.ZipFile(path) as archive:
        content = {name: archive.read(name) for name in archive.namelist()}
    data = json.loads(content["manifest.json"])
    if problem == "missing_mask":
        del content[data["parts"][0]["mask"]]
    elif problem == "order":
        data["parts"][0]["z_order"] = 7
    elif problem == "canvas":
        data["canvas"] = [99, 99]
    elif problem == "version":
        data["schema_version"] = 99
    elif problem == "duplicate":
        data["parts"][1]["id"] = data["parts"][0]["id"]
    elif problem == "mask_size":
        stream = io.BytesIO()
        Image.new("L", (2, 2)).save(stream, "PNG")
        content[data["parts"][0]["mask"]] = stream.getvalue()
    content["manifest.json"] = json.dumps(data).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in content.items():
            archive.writestr(name, value)
    with pytest.raises(ValueError):
        load_project(path)


def test_export_package_matches_source(project, tmp_path):
    populate(project)
    project.parts[0].name = "../../顔"
    project.parts[1].name = "..\\顔"
    original = Path(project.original_path).read_bytes()
    target = tmp_path / "export"
    export_png(project, target)
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["canvas"] == [16, 12]
    assert [item["id"] for item in manifest["parts"]] == [p.id for p in project.parts]
    image = Image.new("RGBA", project.size)
    for order, item in enumerate(manifest["parts"]):
        assert item["z_order"] == order
        file = target / item["file"]
        assert file.resolve().is_relative_to(target.resolve())
        with Image.open(file) as part:
            assert part.mode == "RGBA" and part.size == project.size
            image = Image.alpha_composite(image, part)
    assert np.array_equal(np.array(image), project.source)
    with Image.open(target / "preview.png") as preview:
        assert np.array_equal(np.array(preview), project.source)
    with pytest.raises(ValueError):
        export_png(project, target)
    assert Path(project.original_path).read_bytes() == original


def test_validation_reports_duplicate_and_canvas(project):
    populate(project)
    project.parts[1].name = project.parts[0].name.upper()
    project.parts[1].mask = np.zeros((3, 3), dtype=np.uint8)
    assert len(validate(project)) == 2
