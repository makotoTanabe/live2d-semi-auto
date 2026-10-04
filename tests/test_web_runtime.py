import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageCms
import pytest

from live2d_semi_auto.core import Part, Project, layer_pixels
from live2d_semi_auto.application import Editor
from live2d_semi_auto.web_export import (
    ROLE_CHOICES, export_web_bundle, get_runtime_config, layer_sprite,
    normalize_runtime_config, runtime_model,
)


@pytest.fixture
def project():
    source = np.zeros((40, 32, 4), dtype=np.uint8)
    source[4:30, 5:27] = [180, 105, 90, 210]
    source.flags.writeable = False
    body = np.zeros((40, 32), dtype=np.uint8)
    body[18:30, 5:27] = 255
    face = np.zeros_like(body)
    face[4:18, 8:25] = 128
    return Project(source, "original.png", "a" * 64, parts=[
        Part("maid_body", body, kind="torso_costume", id="body"),
        Part("neck_face", face, kind="face_skin", id="face"),
    ])


def test_model_preserves_canvas_order_ids_and_bounds(project):
    model = runtime_model(project)
    assert model["canvas"] == [32, 40]
    assert [(x["id"], x["role"], x["z_order"], x["bounds"]) for x in model["layers"]] == [
        ("body", "body", 0, [5, 18, 27, 30]), ("face", "face", 1, [8, 4, 25, 18])]
    assert model["groups"]["head"]["pivot"] == [16.5, 17.16]
    assert set(model["expressions"]) == {"neutral", "smile", "angry", "cry", "surprised", "shy"}
    assert model["motions"] == ["idle", "nod", "shake", "greeting"]
    assert model["roleChoices"] == ROLE_CHOICES
    project.parts[0].name = "new display name"
    assert runtime_model(project)["layers"][0]["asset"] == model["layers"][0]["asset"]


def test_crop_recomposes_original_layer_and_does_not_modify_source(project):
    originals = project.source.copy()
    mask = project.parts[1].mask.copy()
    cropped = layer_sprite(project, project.parts[1])
    assert cropped.shape == (14, 17, 4)
    assert np.array_equal(cropped, layer_pixels(project, project.parts[1])[4:18, 8:25])
    assert not np.shares_memory(cropped, project.source)
    cropped[:] = 0
    assert np.array_equal(project.source, originals)
    assert np.array_equal(project.parts[1].mask, mask)
    assert not project.source.flags.writeable


def test_generated_hidden_pixels_are_cropped_but_visible_artwork_wins(project):
    part = project.parts[1]
    part.generated = np.zeros_like(project.source)
    part.generated[2:7, 3:12] = [20, 220, 30, 200]
    part.generated_mask = np.zeros_like(part.mask)
    part.generated_mask[2:7, 3:12] = 150
    bounds = runtime_model(project)["layers"][1]["bounds"]
    assert bounds == [3, 2, 25, 18]
    assert np.array_equal(layer_sprite(project, part), layer_pixels(project, part)[2:18, 3:25])


def test_empty_and_hidden_layers(project):
    empty = Part("empty", np.zeros((40, 32), dtype=np.uint8), id="empty")
    project.parts.append(empty)
    project.parts[0].visible = False
    model = runtime_model(project)
    assert model["omittedEmptyLayers"] == ["empty"]
    assert model["layers"][0]["visible"] is False
    assert layer_sprite(project, empty).shape == (1, 1, 4)


def test_settings_read_last_operation_and_keep_history_unchanged(project):
    project.history = [
        {"operation": "web-runtime-settings", "config": {"strength": .4}},
        {"operation": "edit"},
        {"operation": "web-runtime-settings", "config": {"strength": .8}},
    ]
    config = get_runtime_config(project)
    config["strength"] = 0
    assert project.history[-1]["config"]["strength"] == .8
    assert runtime_model(project)["config"]["strength"] == .8


def test_deleted_part_stored_binding_is_ignored_and_undo_restores_it(project):
    project.history.append({"operation": "web-runtime-settings", "config": {
        "bindings": {"body": {"role": "arm_right"}, "face": {"role": "face"}}}})
    editor = Editor(project)
    editor.delete(0)
    model = runtime_model(editor.project)
    assert list(model["config"]["bindings"]) == ["face"]
    assert "body" in editor.project.history[-1]["config"]["bindings"]
    with pytest.raises(ValueError, match="存在しない"):
        runtime_model(editor.project, {"bindings": {"body": {"role": "body"}}})
    editor.undo()
    assert runtime_model(editor.project)["layers"][0]["role"] == "arm_right"
    editor.redo()
    assert list(runtime_model(editor.project)["config"]["bindings"]) == ["face"]


def test_manual_roles_and_pivots_override_autodetection(project):
    model = runtime_model(project, {"bindings": {"body": {"role": "arm_right", "pivot": [10, 20]}},
                                     "headPivot": [16, 20], "bodyPivot": [16, 30],
                                     "meshResolution": 8, "strength": .7})
    assert model["layers"][0]["role"] == "arm_right"
    assert model["layers"][0]["pivot"] == [10, 20]
    assert model["groups"]["head"]["pivot"] == [16, 20]
    assert not project.history


def test_hair_clip_is_head_accessory_and_not_lip(project):
    project.parts[0].name = "hair_clip"
    project.parts[0].kind = "cross_hairpin"
    assert runtime_model(project)["layers"][0]["role"] == "accessory_head"


@pytest.mark.parametrize("config", [
    {"bindings": {"unknown": {"role": "eye"}}},
    {"bindings": {"face": {"role": "unsupported"}}},
    {"bindings": {"face": {"role": "eye", "unexpected": True}}},
    {"bindings": []}, {"strength": float("nan")}, {"strength": True},
    {"strength": 10**500}, {"headPivot": [float("inf"), 2]},
    {"headPivot": [1]}, {"meshResolution": 3.5}, {"meshResolution": True},
    {"surpriseOption": 4}, [],
])
def test_invalid_settings_fail_without_mutation(project, config):
    with pytest.raises(ValueError):
        normalize_runtime_config(project, config)
    assert not project.history


def test_finite_configuration_is_clamped(project):
    result = normalize_runtime_config(project, {"meshResolution": 100, "strength": 100,
                                                "headPivot": [-100, 100]})
    assert result["meshResolution"] == 8
    assert result["strength"] == 2
    assert result["headPivot"] == [0, 40]


def test_srgb_profile_keeps_transparency(project):
    original = layer_sprite(project, project.parts[1])
    project.icc_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    assert np.array_equal(layer_sprite(project, project.parts[1]), original)


def test_bundle_contains_only_safe_paths_and_escaped_inline_model(project, tmp_path):
    project.parts[1].id = "../../<script>"
    project.parts[1].name = '</script><script>alert("x")</script>'
    path = tmp_path / "character.zip"
    export_web_bundle(project, path)
    with zipfile.ZipFile(path) as archive:
        model = json.loads(archive.read("model.json"))
        names = archive.namelist()
        assert {"model.json", "runtime.js", "index.html", "README.md"}.issubset(names)
        assert not any(".." in name or "<" in name or name.startswith("/") for name in names)
        html = archive.read("index.html").decode()
        assert project.parts[1].name not in html
        assert "\\u003c/script\\u003e" in html
        embedded = html.split('<script type="application/json" id="model">')[1].split("</script>")[0]
        assert json.loads(embedded) == model
        assert 'id="motion"' in html
        for layer in model["layers"]:
            pixels = np.array(Image.open(io.BytesIO(archive.read(layer["asset"]))))
            assert pixels.shape[:2] == (layer["bounds"][3] - layer["bounds"][1],
                                       layer["bounds"][2] - layer["bounds"][0])
    assert not project.history


def test_export_protects_existing_file_and_failed_export_leaves_no_destination(project, tmp_path):
    path = tmp_path / "character.zip"
    path.write_bytes(b"existing original")
    with pytest.raises(ValueError, match="既存"):
        export_web_bundle(project, path)
    assert path.read_bytes() == b"existing original"
    project.icc_profile = b"bad ICC profile"
    with pytest.raises(Exception):
        export_web_bundle(project, tmp_path / "failed.zip")
    assert {p.name for p in tmp_path.iterdir()} == {"character.zip"}


def test_export_rejects_wrong_extension_or_all_invisible(project, tmp_path):
    with pytest.raises(ValueError, match=".zip"):
        export_web_bundle(project, tmp_path / "source.png")
    for part in project.parts:
        part.visible = False
    with pytest.raises(ValueError, match="表示"):
        export_web_bundle(project, tmp_path / "invisible.zip")


def test_runtime_is_classic_script_with_no_remote_dependencies():
    script = Path(__file__).parents[1].joinpath("src/live2d_semi_auto/web_static/runtime.js").read_text()
    assert "global.Live2DWeb=" in script
    assert "fetch(" not in script
    assert "innerHTML" not in script
