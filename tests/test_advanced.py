import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from PIL import ImageCms
from psd_tools import PSDImage
from psd_tools.constants import ColorMode, Resource
import pytest

from live2d_semi_auto.application import Editor
from live2d_semi_auto.core import layer_pixels, composite
from live2d_semi_auto.exporters import PsdExporter
from live2d_semi_auto.inference import ColorPartsBackend
from live2d_semi_auto.inpainting import LaMaBackend, TeleaBackend, RepairProposal
from live2d_semi_auto.infrastructure import import_image, save_project, load_project, export_png
from live2d_semi_auto.models import verify_lama, download_lama


def setup_repair(project):
    editor = Editor(project)
    part = editor.add_part()
    part.mask[2:7, 2:8] = 255
    part.hidden_mask = np.zeros_like(part.mask)
    part.hidden_mask[7:9, 2:8] = 255
    return editor


def test_auto_parts_partition_and_repeatability(project):
    before = project.source.copy()
    a = ColorPartsBackend(4).propose_parts(project.source)
    b = ColorPartsBackend(4).propose_parts(project.source)
    assert 1 < len(a.parts) <= 4
    coverage = np.sum([p.mask > 0 for p in a.parts], axis=0)
    assert np.array_equal(coverage, project.source[..., 3] > 0)
    for left, right in zip(a.parts, b.parts, strict=True):
        assert np.array_equal(left.mask, right.mask)
    editor = Editor(project)
    editor.accept_parts(a)
    assert np.array_equal(composite(editor.project), before)
    assert editor.project.history[-1]["semantic_labels"] is False
    editor.undo()
    assert not editor.project.parts and not editor.project.history


def test_auto_parts_keeps_manual_edits(project):
    editor = Editor(project)
    manual = editor.add_part()
    manual.name = "color_region_01"
    manual.mask[3, 3] = 255
    candidate = ColorPartsBackend(3).propose_parts(project.source)
    editor.accept_parts(candidate)
    assert editor.project.parts[0].id == manual.id
    assert editor.project.parts[0].mask[3, 3] == 255
    assert len({p.name for p in editor.project.parts}) == len(editor.project.parts)
    editor.undo()
    assert len(editor.project.parts) == 1


def test_uniform_and_transparent_auto_parts():
    source = np.full((16, 16, 4), 255, dtype=np.uint8)
    assert len(ColorPartsBackend(8).propose_parts(source).parts) == 1
    source[..., 3] = 0
    with pytest.raises(ValueError):
        ColorPartsBackend().propose_parts(source)


def test_repair_preserves_visible_pixels_and_undo(project):
    editor = setup_repair(project)
    source = project.source.copy()
    part = project.parts[0]
    before = layer_pixels(project, part)
    target = editor.repair_mask(0)
    proposal = TeleaBackend().propose(source, target)
    assert np.array_equal(proposal.pixels[target == 0], source[target == 0])
    editor.accept_repair(0, proposal)
    result = layer_pixels(editor.project, editor.project.parts[0])
    assert np.array_equal(result[part.mask > 0], before[part.mask > 0])
    assert np.all(result[target > 0, 3] == 255)
    assert np.array_equal(editor.project.source, source)
    assert editor.project.history[-1]["ai"] is False
    assert not editor.project.parts[0].generated.flags.writeable
    editor.undo()
    assert editor.project.parts[0].generated is None
    editor.redo()
    assert np.array_equal(layer_pixels(editor.project, editor.project.parts[0]), result)


def test_hidden_brush_is_independent(project):
    editor = Editor(project)
    editor.add_part()
    editor.checkpoint()
    editor.paint(0, (3, 3), (5, 3), 1, hidden=True)
    assert not np.any(editor.project.parts[0].mask)
    assert np.any(editor.project.parts[0].hidden_mask)
    editor.undo()
    assert editor.project.parts[0].hidden_mask is None


def test_visible_overlap_and_invalid_repairs_rejected(project):
    editor = setup_repair(project)
    editor.project.parts[0].hidden_mask[3, 3] = 255
    with pytest.raises(ValueError, match="重なって"):
        editor.repair_mask(0)
    editor.project.parts[0].hidden_mask[3, 3] = 0
    with pytest.raises(ValueError):
        editor.accept_repair(0, RepairProposal(project.source.copy(), np.zeros((2, 2), np.uint8), {}))
    assert editor.project.parts[0].generated is None


def test_regeneration_retains_other_repairs(project):
    editor = setup_repair(project)
    first_mask = editor.repair_mask(0)
    editor.accept_repair(0, TeleaBackend().propose(project.source, first_mask))
    part = editor.project.parts[0]
    previous = part.generated.copy()
    part.hidden_mask[:] = 0
    part.hidden_mask[9:10, 2:8] = 255
    next_mask = editor.repair_mask(0)
    editor.accept_repair(0, TeleaBackend().propose(project.source, next_mask))
    assert np.array_equal(part.generated[first_mask > 0], previous[first_mask > 0])
    assert np.all(part.generated_mask[(first_mask > 0) | (next_mask > 0)] == 255)
    # Later visible edits take priority over any generated region.
    part.mask[first_mask > 0] = 255
    assert np.array_equal(layer_pixels(project, part)[first_mask > 0], project.source[first_mask > 0])


def test_repair_persistence_and_provenance(project, tmp_path):
    editor = setup_repair(project)
    proposal = TeleaBackend().propose(project.source, editor.repair_mask(0))
    editor.accept_repair(0, proposal)
    path = tmp_path / "repair.l2split"
    save_project(project, path)
    loaded = load_project(path)
    assert loaded.history == project.history
    for key in ("mask", "hidden_mask", "generated", "generated_mask"):
        assert np.array_equal(getattr(loaded.parts[0], key), getattr(project.parts[0], key))
    target = tmp_path / "export"
    export_png(loaded, target)
    manifest = json.loads((target / "manifest.json").read_text())
    assert (target / manifest["parts"][0]["generated_mask"]).is_file()
    assert manifest["history"][0]["backend"] == "opencv-telea"


def test_schema_one_still_loads(project, tmp_path):
    Editor(project).add_part()
    path = tmp_path / "old.l2split"
    save_project(project, path)
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    data = json.loads(entries["manifest.json"])
    data["schema_version"] = 1
    data.pop("history")
    entries["manifest.json"] = json.dumps(data).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    assert load_project(path).history == []


def test_psd_layers_srgb_transparency_and_order(project, tmp_path):
    editor = Editor(project)
    a, b = editor.add_part(), editor.add_part()
    a.name, b.name = "顔", "髪"
    a.mask[:, :8] = 255
    b.mask[:, 8:] = 255
    target = tmp_path / "parts.psd"
    PsdExporter().export(project, target)
    psd = PSDImage.open(target)
    assert psd.color_mode == ColorMode.RGB and psd.depth == 8
    assert psd.size == project.size
    assert [p.name for p in psd] == ["顔", "髪"]
    assert all(not p.has_mask() for p in psd)
    profile = ImageCms.ImageCmsProfile(io.BytesIO(psd.image_resources.get_data(Resource.ICC_PROFILE)))
    assert "sRGB" in ImageCms.getProfileDescription(profile)
    for actual, expected in zip(psd, project.parts, strict=True):
        assert np.array_equal(np.array(actual.topil()), layer_pixels(project, expected))
    reconstructed = np.array(psd.composite().convert("RGBA"))
    visible = project.source[..., 3] > 0
    assert np.max(np.abs(reconstructed[visible].astype(int) - project.source[visible].astype(int))) <= 1
    with pytest.raises(ValueError):
        PsdExporter().export(project, target)


def test_psd_visibility_and_profile_conversion(project, tmp_path):
    editor = Editor(project)
    part = editor.add_part()
    part.mask[:] = 255
    part.visible = False
    project.icc_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    path = tmp_path / "hidden.psd"
    PsdExporter().export(project, path)
    assert not PSDImage.open(path)[0].visible


def test_checksum_rejects_unknown_model(tmp_path):
    path = tmp_path / "bad.pt"
    path.write_bytes(b"unknown weights")
    with pytest.raises(ValueError, match="チェックサム"):
        verify_lama(path)


def test_download_verification_failure_never_publishes(tmp_path, monkeypatch):
    from contextlib import closing
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: closing(io.BytesIO(b"bad weights")))
    target = tmp_path / "model.pt"
    with pytest.raises(ValueError, match="チェックサム"):
        download_lama(target)
    assert not list(tmp_path.iterdir())


def test_psd_failed_publication_leaves_no_partial(project, tmp_path, monkeypatch):
    part = Editor(project).add_part()
    part.mask[:] = 255
    def fail(*args):
        raise OSError("simulated publication failure")
    monkeypatch.setattr("live2d_semi_auto.exporters.os.link", fail)
    with pytest.raises(OSError):
        PsdExporter().export(project, tmp_path / "test.psd")
    assert not list(tmp_path.glob("*.psd"))


@pytest.mark.ai
def test_real_lama_keeps_source_and_records_transform():
    path = os.environ.get("LAMA_MODEL_PATH")
    if not path:
        pytest.skip("Set LAMA_MODEL_PATH explicitly to run real neural inference")
    pytest.importorskip("torch")
    project = import_image(Path(__file__).parents[1] / "samples" / "original_character.png")
    target = np.zeros(project.source.shape[:2], np.uint8)
    target[180:280, 500:600] = 255
    result = LaMaBackend(path).propose(project.source, target)
    assert result.metadata["ai"] is True
    assert result.metadata["padding"] == [4, 4]
    assert result.pixels.shape == project.source.shape
    assert np.array_equal(result.pixels[target == 0], project.source[target == 0])
    assert np.any(result.pixels[target > 0] != project.source[target > 0])
