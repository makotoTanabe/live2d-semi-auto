"""Non-destructive aligned assets, deterministic archives and undo behavior."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
import zipfile

import numpy as np
from PIL import Image, ImageCms
import pytest

from live2d_semi_auto.application import Editor
from live2d_semi_auto.core import Part, composite, layer_pixels, validate
from live2d_semi_auto.inference import PartsProposal
from live2d_semi_auto.infrastructure import export_png, load_project, save_project, require_matching_profiles


def readonly(array):
    array.flags.writeable = False
    return array


def aligned_proposal(project, name="face"):
    atlas = np.zeros((6, 8, 4), dtype=np.uint8)
    atlas[1:4, 2:6] = [20, 60, 220, 255]
    atlas[2, 3] = [90, 180, 30, 128]
    atlas = readonly(atlas)
    digest = hashlib.sha256(atlas.tobytes()).hexdigest()
    asset = readonly(atlas[1:4, 2:6].copy())
    asset_mask = readonly(np.full(asset.shape[:2], 255, dtype=np.uint8))
    artwork = np.zeros_like(project.source)
    artwork[4:7, 5:9] = asset
    mask = np.zeros(project.source.shape[:2], dtype=np.uint8)
    mask[4:7, 5:9] = 255
    info = {
        "atlas_hash": digest, "crop": [2, 1, 6, 4], "source_size": [8, 6],
        "destination_size": list(project.size), "matrix": [[1, 0, 5], [0, 1, 4]],
        "scale": 1, "offset": [5, 4],
        "anchors": [{"source": [0, 0], "destination": [5, 4]},
                    {"source": [3, 2], "destination": [8, 6]}],
    }
    part = Part(name, mask, "face", artwork=readonly(artwork), asset=asset,
                asset_mask=asset_mask, alignment=info)
    return PartsProposal([part], {"backend": "alignment-fixture", "checks": {"ok": True}},
                         {digest: atlas})


def test_aligned_artwork_composites_its_own_pixels_and_preserves_source(project):
    source = project.source.copy()
    editor = Editor(project)
    proposal = aligned_proposal(project)
    editor.accept_parts(proposal)
    accepted = editor.project.parts[0]
    pixels = layer_pixels(editor.project, accepted)
    assert np.array_equal(pixels[4:7, 5:9], proposal.parts[0].asset)
    assert np.array_equal(composite(editor.project)[4:7, 5:9], accepted.artwork[4:7, 5:9])
    assert np.all(pixels[:4, :, 3] == 0)
    assert np.array_equal(editor.project.source, source)
    # Human mask edits affect aligned artwork, rather than exposing reference-image RGB.
    accepted.mask[5, 6] = 128
    assert layer_pixels(editor.project, accepted)[5, 6, 3] == 64
    assert np.array_equal(layer_pixels(editor.project, accepted)[5, 6, :3], [90, 180, 30])


def test_alignment_archive_roundtrip_is_self_contained(project, tmp_path):
    editor = Editor(project)
    editor.accept_parts(aligned_proposal(project, "顔"))
    editor.project.parts[0].visible = False
    editor.project.parts[0].mask[4, 5] = 0
    path = tmp_path / "aligned.l2split"
    save_project(editor.project, path)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["schema_version"] == 3
        assert len(manifest["assets"]) == 1
        for key in ("artwork", "asset", "asset_mask"):
            assert manifest["parts"][0][key] in archive.namelist()
    loaded = load_project(path)
    before, after = editor.project.parts[0], loaded.parts[0]
    assert (after.id, after.name, after.visible) == (before.id, "顔", False)
    assert after.alignment == before.alignment
    for key in ("mask", "artwork", "asset", "asset_mask"):
        assert np.array_equal(getattr(after, key), getattr(before, key))
    assert after.mask.flags.writeable
    for key in ("artwork", "asset", "asset_mask"):
        assert not getattr(after, key).flags.writeable
    for digest, atlas in loaded.assets.items():
        assert np.array_equal(atlas, editor.project.assets[digest])
        assert not atlas.flags.writeable
    assert loaded.history == editor.project.history
    assert validate(loaded) == []
    assert np.array_equal(composite(loaded), composite(editor.project))


def test_aligned_visible_pixels_override_repairs_before_and_after_roundtrip(project, tmp_path):
    editor = Editor(project)
    editor.accept_parts(aligned_proposal(project))
    part = editor.project.parts[0]
    generated = np.zeros_like(project.source)
    generated[5, 6] = [255, 0, 0, 255]
    generated[9, 11] = [0, 255, 0, 255]
    generated_mask = np.zeros_like(part.mask)
    generated_mask[5, 6] = generated_mask[9, 11] = 255
    part.generated = readonly(generated)
    part.generated_mask = readonly(generated_mask)
    expected = layer_pixels(editor.project, part)
    assert np.array_equal(expected[5, 6], part.artwork[5, 6])
    assert np.array_equal(expected[9, 11], [0, 255, 0, 255])
    target = tmp_path / "aligned-repaired.l2split"
    save_project(editor.project, target)
    loaded = load_project(target)
    assert np.array_equal(layer_pixels(loaded, loaded.parts[0]), expected)


def test_alignment_png_export_contains_originals_and_transform(project, tmp_path):
    editor = Editor(project)
    editor.accept_parts(aligned_proposal(project))
    target = tmp_path / "export"
    export_png(editor.project, target)
    manifest = json.loads((target / "manifest.json").read_text())
    item = manifest["parts"][0]
    assert item["alignment"] == editor.project.parts[0].alignment
    for key in ("asset", "asset_mask"):
        with Image.open(target / item[key]) as image:
            assert np.array_equal(np.array(image), getattr(editor.project.parts[0], key))
    for digest, relative in manifest["assets"].items():
        with Image.open(target / relative) as image:
            assert np.array_equal(np.array(image), editor.project.assets[digest])
    with Image.open(target / item["file"]) as image:
        assert np.array_equal(np.array(image), layer_pixels(editor.project, editor.project.parts[0]))


def test_accept_alignment_preserves_ids_and_manual_edits_then_undo(project):
    editor = Editor(project)
    manual = editor.add_part()
    manual.name = "face"
    manual.mask[2, 2] = 255
    proposal = aligned_proposal(project)
    expected_id = proposal.parts[0].id
    editor.accept_parts(proposal)
    accepted = editor.project.parts[1]
    assert accepted.id == expected_id
    assert accepted.name == "face_2"
    assert accepted.artwork is proposal.parts[0].artwork
    assert editor.project.parts[0].id == manual.id
    assert editor.project.parts[0].mask[2, 2] == 255
    proposal.parts[0].mask[:] = 0
    proposal.parts[0].alignment["anchors"][0]["source"][0] = 100
    proposal.metadata["checks"]["ok"] = False
    assert np.any(accepted.mask)
    assert accepted.alignment["anchors"][0]["source"][0] == 0
    assert editor.project.history[-1]["checks"]["ok"] is True
    editor.undo()
    assert [part.id for part in editor.project.parts] == [manual.id]
    assert not editor.project.assets
    editor.redo()
    assert editor.project.parts[1].id == expected_id
    assert editor.project.parts[1].artwork is accepted.artwork


def test_snapshot_copies_editable_metadata_and_shares_immutable_assets(project):
    editor = Editor(project)
    editor.accept_parts(aligned_proposal(project))
    snapshot = editor.project.snapshot()
    original, copied = editor.project.parts[0], snapshot.parts[0]
    assert copied.artwork is original.artwork
    assert copied.asset is original.asset
    assert copied.asset_mask is original.asset_mask
    assert snapshot.assets is not editor.project.assets
    digest = next(iter(snapshot.assets))
    assert snapshot.assets[digest] is editor.project.assets[digest]
    copied.mask[:] = 0
    copied.alignment["anchors"][0]["source"][0] = 99
    snapshot.assets.clear()
    assert np.any(original.mask)
    assert original.alignment["anchors"][0]["source"][0] == 0
    assert editor.project.assets


@pytest.mark.parametrize("collision", ["existing_id", "incoming_id", "atlas_hash"])
def test_accept_alignment_rejects_collisions_without_changing_edits(project, collision):
    editor = Editor(project)
    manual = editor.add_part()
    manual.mask[2, 2] = 255
    proposal = aligned_proposal(project)
    if collision == "existing_id":
        proposal.parts[0].id = manual.id
    elif collision == "incoming_id":
        proposal.parts.append(replace(proposal.parts[0], name="face_again"))
    else:
        digest = next(iter(proposal.assets))
        editor.project.assets[digest] = readonly(np.zeros_like(proposal.assets[digest]))
    undo_count = len(editor.undo_stack)
    with pytest.raises(ValueError):
        editor.accept_parts(proposal)
    assert len(editor.undo_stack) == undo_count
    assert [part.id for part in editor.project.parts] == [manual.id]
    assert editor.project.parts[0].mask[2, 2] == 255


@pytest.mark.parametrize("problem", ["partial", "canvas", "crop", "original", "matrix", "matrix_type", "missing_atlas", "mutable"])
def test_alignment_validation_rejects_incomplete_or_corrupt_provenance(project, problem):
    proposal = aligned_proposal(project)
    project.parts = proposal.parts
    project.assets = proposal.assets
    part = project.parts[0]
    if problem == "partial":
        part.asset_mask = None
    elif problem == "canvas":
        part.alignment["destination_size"] = [99, 99]
    elif problem == "crop":
        part.alignment["crop"][2] = 99
    elif problem == "original":
        part.asset = readonly(np.zeros_like(part.asset))
    elif problem == "matrix":
        part.alignment["matrix"][0][0] = float("nan")
    elif problem == "matrix_type":
        part.alignment["matrix"][0][0] = "1"
    elif problem == "missing_atlas":
        project.assets.clear()
    else:
        part.artwork = part.artwork.copy()
    assert validate(project)


@pytest.mark.parametrize("schema", [1, 2])
def test_older_archives_remain_readable(project, tmp_path, schema):
    editor = Editor(project)
    part = editor.add_part()
    part.mask[2:5, 2:5] = 255
    path = tmp_path / "legacy.l2split"
    save_project(project, path)
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    data = json.loads(entries["manifest.json"])
    data["schema_version"] = schema
    data.pop("assets")
    if schema == 1:
        data.pop("history")
    entries["manifest.json"] = json.dumps(data).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    loaded = load_project(path)
    assert loaded.parts[0].id == part.id
    assert loaded.parts[0].artwork is None
    assert loaded.parts[0].alignment is None
    assert loaded.assets == {}
    assert np.array_equal(layer_pixels(loaded, loaded.parts[0]), layer_pixels(project, part))


def test_manual_alignment_moves_edited_mask_reversibly(project):
    editor = Editor(project)
    editor.accept_parts(aligned_proposal(project))
    part = editor.project.parts[0]
    part.mask[4, 5] = 0
    before = deepcopy(part.alignment)
    before_pixels = layer_pixels(editor.project, part)
    asset = part.asset
    editor.adjust_alignment(0, offset=(2, 1))
    moved = editor.project.parts[0]
    assert moved.id == part.id
    assert moved.asset is asset
    assert moved.mask[5, 7] == 0
    assert np.array_equal(layer_pixels(editor.project, moved)[5:8, 7:11], before_pixels[4:7, 5:9])
    editor.undo()
    assert editor.project.parts[0].id == part.id
    assert editor.project.parts[0].alignment == before
    assert np.array_equal(layer_pixels(editor.project, editor.project.parts[0]), before_pixels)
    editor.redo()
    assert editor.project.parts[0].id == part.id
    assert np.array_equal(editor.project.parts[0].mask, moved.mask)


def test_failed_alignment_adjustment_preserves_history_and_repairs(project):
    editor = Editor(project)
    editor.accept_parts(aligned_proposal(project))
    part = editor.project.parts[0]
    before = deepcopy(part.alignment)
    undo_count = len(editor.undo_stack)
    with pytest.raises(ValueError):
        editor.adjust_alignment(0, scale=0)
    assert len(editor.undo_stack) == undo_count
    assert editor.project.parts[0].alignment == before
    part.hidden_mask = np.zeros_like(part.mask)
    part.hidden_mask[9, 9] = 255
    with pytest.raises(ValueError, match="補完"):
        editor.adjust_alignment(0, offset=(1, 0))
    assert len(editor.undo_stack) == undo_count
    assert editor.project.parts[0].alignment == before


def test_canonical_edited_mask_survives_adjustment_archive_and_undo(project, tmp_path):
    editor = Editor(project)
    editor.accept_parts(aligned_proposal(project))
    editor.project.parts[0].mask[4, 5] = 0
    baseline = editor.project.parts[0].mask.copy()
    original_id = editor.project.parts[0].id
    editor.adjust_alignment(0, angle=17)
    part = editor.project.parts[0]
    assert part.alignment_edit_mask is not None
    assert not part.alignment_edit_mask.flags.writeable
    assert np.array_equal(part.alignment_edit_mask, baseline)
    assert part.alignment["edit_mask_matrix"] == [[1, 0, 5], [0, 1, 4]]
    snapshot = editor.project.snapshot()
    assert snapshot.parts[0].alignment_edit_mask is part.alignment_edit_mask
    path = tmp_path / "canonical-mask.l2split"
    save_project(editor.project, path)
    loaded = load_project(path)
    assert loaded.parts[0].id == original_id
    assert np.array_equal(loaded.parts[0].alignment_edit_mask, baseline)
    assert not loaded.parts[0].alignment_edit_mask.flags.writeable
    loaded_editor = Editor(loaded)
    loaded_editor.adjust_alignment(0, angle=-17)
    assert np.array_equal(loaded_editor.project.parts[0].mask, baseline)
    canonical = loaded_editor.project.parts[0].alignment_edit_mask
    for _ in range(4):
        loaded_editor.adjust_alignment(0, angle=17)
        loaded_editor.adjust_alignment(0, angle=-17)
    assert np.array_equal(loaded_editor.project.parts[0].mask, baseline)
    assert loaded_editor.project.parts[0].alignment_edit_mask is canonical
    loaded_editor.undo()
    assert loaded_editor.project.parts[0].id == original_id
    assert loaded_editor.project.parts[0].alignment_edit_mask is canonical
    loaded_editor.redo()
    assert np.array_equal(loaded_editor.project.parts[0].mask, baseline)
    output = tmp_path / "canonical-export"
    export_png(loaded_editor.project, output)
    manifest = json.loads((output / "manifest.json").read_text())
    with Image.open(output / manifest["parts"][0]["alignment_edit_mask"]) as image:
        assert np.array_equal(np.array(image), baseline)


@pytest.mark.parametrize("problem", ["mutable", "shape", "matrix", "missing_alignment"])
def test_invalid_canonical_edited_masks_rejected(project, problem):
    proposal = aligned_proposal(project)
    project.parts, project.assets = proposal.parts, proposal.assets
    part = project.parts[0]
    part.alignment_edit_mask = readonly(part.mask.copy())
    part.alignment["edit_mask_matrix"] = deepcopy(part.alignment["matrix"])
    if problem == "mutable":
        part.alignment_edit_mask = part.alignment_edit_mask.copy()
    elif problem == "shape":
        part.alignment_edit_mask = readonly(np.zeros((3, 3), dtype=np.uint8))
    elif problem == "matrix":
        part.alignment["edit_mask_matrix"] = [[1, 0, True], [0, 1, 0]]
    else:
        part.artwork = part.asset = part.asset_mask = part.alignment = None
    assert validate(project)


@pytest.mark.parametrize("reference_profile,atlas_profile,accepted", [
    (None, None, True), (b"same profile", b"same profile", True),
    (b"profile A", b"profile B", False), (None, b"profile A", False),
])
def test_input_profile_guard_is_explicit_and_preserves_originals(project, reference_profile, atlas_profile, accepted):
    reference = replace(project, icc_profile=reference_profile)
    atlas = replace(project, icc_profile=atlas_profile)
    before = project.source.copy()
    if accepted:
        require_matching_profiles(reference, atlas)
    else:
        with pytest.raises(ValueError, match="sRGB.*まだ送信されていません"):
            require_matching_profiles(reference, atlas)
    assert np.array_equal(project.source, before)


def test_matching_profile_preserved_in_original_atlas_and_crop_assets(project, tmp_path):
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    project.icc_profile = profile
    editor = Editor(project)
    editor.accept_parts(aligned_proposal(project))
    path = tmp_path / "profile.l2split"
    save_project(editor.project, path)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        relative_paths = list(manifest["assets"].values()) + [manifest["parts"][0]["asset"]]
        for relative in relative_paths:
            with Image.open(io.BytesIO(archive.read(relative))) as image:
                assert image.info["icc_profile"] == profile
    target = tmp_path / "profile-export"
    export_png(editor.project, target)
    manifest = json.loads((target / "manifest.json").read_text())
    relative_paths = list(manifest["assets"].values()) + [manifest["parts"][0]["asset"]]
    for relative in relative_paths:
        with Image.open(target / relative) as image:
            assert image.info["icc_profile"] == profile
