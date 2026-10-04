import base64
import io
import json

import cv2
import numpy as np
from PIL import Image
import pytest

from live2d_semi_auto.alignment import (
    GPTAlignmentBackend, adjust_part, extraction_mask, fit_similarity, warp_asset,
)
from live2d_semi_auto.core import Part, Project, layer_pixels, validate


def response(parts):
    return {"choices": [{"message": {"content": json.dumps({"parts": parts})}}]}


def item(**changes):
    data = {"name": "face", "kind": "face", "atlas_bbox": [250, 250, 750, 750],
            "anchors": [{"source": [250, 250], "target": [250, 250]},
                        {"source": [750, 250], "target": [750, 250]}],
            "confidence": 0.9, "notes": "Inspect the candidate."}
    data.update(changes)
    return data


def square():
    pixels = np.zeros((32, 32, 4), np.uint8)
    pixels[8:24, 8:24] = [230, 60, 100, 255]
    return pixels


def propose(parts=None, *, atlas=None, reference=None, background_mode="alpha"):
    atlas = square() if atlas is None else atlas
    reference = atlas.copy() if reference is None else reference
    backend = GPTAlignmentBackend(transport=lambda body: response(parts or [item()]))
    return backend.propose_alignment(reference, atlas, background_mode=background_mode)


def test_known_similarity_rotation_scale_and_translation():
    source = np.array([[0, 0], [3, 0], [0, 4]], dtype=float)
    target = np.array([[10, 20], [10, 26], [2, 20]], dtype=float)
    matrix, metadata = fit_similarity(source, target)
    np.testing.assert_allclose(matrix, [[0, -2, 10], [2, 0, 20]], atol=1e-12)
    assert metadata["scale"] == pytest.approx(2)
    assert metadata["rotation_degrees"] == pytest.approx(90)
    assert metadata["anchor_residual_px"] == pytest.approx(0)


@pytest.mark.parametrize("source,target", [
    ([[1, 1], [1, 1]], [[2, 2], [3, 3]]),
    ([[0, 0], [1, 1]], [[2, 2], [2, 2]]),
    ([[0, 0], [1, 1]], [[2, 2], [float("nan"), 3]]),
    ([[0, 0]], [[2, 2]]),
    ([[0, 0], [1, 0]], [[0, 0], [1000, 0]]),
])
def test_degenerate_or_extreme_anchors_rejected(source, target):
    with pytest.raises(ValueError):
        fit_similarity(source, target)


def test_unequal_canvases_and_exact_crop_local_transform():
    atlas = np.zeros((40, 80, 4), np.uint8)
    atlas[5:25, 10:30] = [220, 120, 50, 255]
    reference = np.zeros((100, 120, 4), np.uint8)
    candidate = item(atlas_bbox=[125, 125, 375, 625], anchors=[
        {"source": [125, 125], "target": [1000 / 3, 300]},
        {"source": [375, 125], "target": [2000 / 3, 300]},
    ])
    proposal = propose([candidate], atlas=atlas, reference=reference)
    part = proposal.parts[0]
    np.testing.assert_allclose(part.alignment["matrix"], [[2, 0, 40], [0, 2, 30]], atol=1e-10)
    assert part.alignment["crop"] == [10, 5, 30, 25]
    assert part.alignment["source_size"] == [80, 40]
    assert part.alignment["destination_size"] == [120, 100]
    assert part.asset.shape == (20, 20, 4)
    assert part.artwork.shape == reference.shape
    assert np.array_equal(part.asset, atlas[5:25, 10:30])
    assert np.array_equal(proposal.assets[part.alignment["atlas_hash"]], atlas)
    assert not part.asset.flags.writeable and not part.asset_mask.flags.writeable
    assert not part.artwork.flags.writeable
    project = Project(reference, "reference", "hash", parts=proposal.parts, assets=proposal.assets)
    assert validate(project) == []


def test_two_images_are_sent_with_independent_resize_metadata():
    atlas = np.zeros((600, 1200, 4), np.uint8)
    atlas[150:450, 300:900] = 255
    reference = np.full((1200, 600, 4), 255, np.uint8)
    received = []
    backend = GPTAlignmentBackend(transport=lambda body: received.append(body) or response([item()]))
    proposal = backend.propose_alignment(reference, atlas)
    body = received[0]
    urls = [block["image_url"]["url"] for block in body["messages"][0]["content"] if block["type"] == "image_url"]
    assert len(urls) == 2
    for url, size in zip(urls, [(512, 1024), (1024, 512)]):
        with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
            assert image.size == size and image.mode == "RGB"
    assert body["response_format"]["json_schema"]["strict"]
    assert proposal.metadata["sent_sizes"] == {"reference": [512, 1024], "atlas": [1024, 512]}


@pytest.mark.parametrize("model,limit,reasoning", [
    ("gpt-5.6-luna", 8192, "low"), ("gpt-4.1", 4096, None),
])
def test_reasoning_and_output_budget_are_model_appropriate(model, limit, reasoning):
    received = []
    backend = GPTAlignmentBackend(model=model, max_completion_tokens=limit,
                                  transport=lambda body: received.append(body) or response([item()]))
    backend.propose_alignment(square(), square())
    assert received[0]["max_completion_tokens"] == limit
    assert received[0].get("reasoning_effort") == reasoning


@pytest.mark.parametrize("limit", [True, 100, 32769, 8192.0])
def test_invalid_output_budget_rejected_before_network(limit):
    with pytest.raises(ValueError, match="出力上限"):
        GPTAlignmentBackend(max_completion_tokens=limit)


def test_truncated_model_response_explains_output_limit():
    truncated = {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
    backend = GPTAlignmentBackend(transport=lambda body: truncated)
    with pytest.raises(ValueError, match="出力トークン上限"):
        backend.propose_alignment(square(), square())


def test_alpha_is_not_squared_and_original_pixels_are_unchanged():
    atlas = square()
    atlas[8:24, 8:24, 3] = 128
    before = atlas.copy()
    proposal = propose(atlas=atlas)
    part = proposal.parts[0]
    assert part.artwork[12, 12, 3] == 128
    assert part.mask[12, 12] == 255
    project = Project(atlas, "source", "hash", parts=proposal.parts, assets=proposal.assets)
    assert layer_pixels(project, part)[12, 12, 3] == 128
    assert np.array_equal(atlas, before)


def test_premultiplied_warp_preserves_color_at_transparent_edge():
    asset = np.array([[[255, 0, 0, 255], [0, 0, 0, 0]]], np.uint8)
    result = warp_asset(asset, np.full((1, 2), 255, np.uint8),
                        np.array([[1, 0, 0.5], [0, 1, 0]], float), (4, 2))
    assert result[0, 1].tolist() == [255, 0, 0, 128]


def test_white_background_flood_preserves_enclosed_white_artwork():
    asset = np.full((7, 7, 4), 255, np.uint8)
    asset[1:6, 1:6, :3] = [220, 40, 80]
    asset[3, 3, :3] = 255
    mask = extraction_mask(asset, "white")
    assert not mask[0].any() and not mask[:, 0].any()
    assert mask[3, 3] == 255
    assert mask[1, 1] == 255
    assert extraction_mask(asset, "alpha").min() == 255
    assert np.all(asset[0] == 255)


@pytest.mark.parametrize("changes", [
    {"atlas_bbox": [0, 0, True, 1000]},
    {"confidence": True}, {"confidence": float("nan")}, {"confidence": 2},
    {"name": "\nface"}, {"kind": ""}, {"name": ""},
    {"anchors": [{"source": [250, True], "target": [250, 250]},
                 {"source": [750, 250], "target": [750, 250]}]},
    {"anchors": [{"source": [100, 250], "target": [250, 250]},
                 {"source": [750, 250], "target": [750, 250]}]},
    {"anchors": [{"source": [250, 250], "target": [250, 250]}]},
    {"anchors": [{"source": [250, 250], "target": [250, 250]},
                 {"source": [250, 250], "target": [750, 250]}]},
    {"unexpected": "value"},
])
def test_invalid_model_output_rejected_without_changes(changes):
    atlas = square()
    before = atlas.copy()
    with pytest.raises(ValueError):
        propose([item(**changes)], atlas=atlas)
    assert np.array_equal(atlas, before)


def test_duplicate_names_empty_transparent_parts_and_refusal_rejected():
    with pytest.raises(ValueError):
        propose([item(), item(name=" FACE ")])
    with pytest.raises(ValueError, match="非透明"):
        propose(atlas=np.zeros((32, 32, 4), np.uint8))
    backend = GPTAlignmentBackend(transport=lambda body: {"choices": [{"message": {"refusal": "declined"}}]})
    with pytest.raises(ValueError, match="候補を返しません"):
        backend.propose_alignment(square(), square())


def test_low_confidence_and_bad_anchor_fit_are_inspectable_warnings():
    candidate = item(confidence=0.3, anchors=[
        {"source": [250, 250], "target": [250, 250]},
        {"source": [750, 250], "target": [750, 250]},
        {"source": [250, 750], "target": [750, 750]},
    ])
    part = propose([candidate]).parts[0]
    assert "low-confidence" in part.alignment["warnings"]
    assert "anchor-mismatch" in part.alignment["warnings"]
    assert part.alignment["anchor_residual_px"] > 3


def test_repeated_adjustments_rewarp_original_without_artwork_drift():
    part = propose().parts[0]
    before = part.artwork.copy()
    anchors = part.alignment["anchors"]
    changed = part
    for _ in range(10):
        changed = adjust_part(changed, (32, 32), angle=17)
        changed = adjust_part(changed, (32, 32), angle=-17)
    assert changed.id == part.id
    assert changed.asset is part.asset and changed.asset_mask is part.asset_mask
    assert np.array_equal(changed.artwork, before)
    assert np.array_equal(changed.mask, part.mask)
    assert changed.alignment["anchors"] == anchors
    assert len(changed.alignment["manual_adjustments"]) == 20
    assert part.alignment["manual_adjustments"] == []


def asymmetric_part():
    atlas = np.zeros((64, 64, 4), np.uint8)
    atlas[8:56, 8:56] = [230, 60, 100, 255]
    atlas[8:32, 36:56] = 0
    candidate = item(atlas_bbox=[0, 0, 1000, 1000], anchors=[
        {"source": [125, 125], "target": [125, 125]},
        {"source": [875, 125], "target": [875, 125]},
    ])
    return propose([candidate], atlas=atlas).parts[0]


def test_asymmetric_rotations_use_stable_asset_pivot():
    part = asymmetric_part()
    changed = part
    for _ in range(20):
        changed = adjust_part(changed, (64, 64), angle=17)
        changed = adjust_part(changed, (64, 64), angle=-17)
    np.testing.assert_allclose(changed.alignment["matrix"], part.alignment["matrix"], atol=1e-10)
    assert np.array_equal(changed.artwork, part.artwork)
    assert np.array_equal(changed.mask, part.mask)


def test_repeated_edited_mask_rotations_do_not_erode_cuts():
    part = asymmetric_part()
    part.mask[15:45, 18:22] = 0
    before = part.mask.copy()
    changed = part
    for _ in range(10):
        changed = adjust_part(changed, (64, 64), angle=17)
        changed = adjust_part(changed, (64, 64), angle=-17)
    assert np.array_equal(changed.mask, before)
    assert np.array_equal(changed.artwork, part.artwork)
    assert np.array_equal(changed.alignment_edit_mask, before)
    assert not changed.alignment_edit_mask.flags.writeable
    assert part.alignment_edit_mask is None
    np.testing.assert_allclose(changed.alignment["edit_mask_matrix"], part.alignment["matrix"])


def test_new_manual_edit_recaptures_canonical_mask_at_current_matrix():
    part = asymmetric_part()
    part.mask[15:45, 18:22] = 0
    turned = adjust_part(part, (64, 64), angle=17)
    first_origin = turned.alignment_edit_mask
    turned.mask[25:35, 25:35] = 0
    before = turned.mask.copy()
    changed = adjust_part(turned, (64, 64), angle=5)
    assert changed.alignment_edit_mask is not first_origin
    assert np.array_equal(changed.alignment_edit_mask, before)
    np.testing.assert_allclose(changed.alignment["edit_mask_matrix"], turned.alignment["matrix"])
    restored = adjust_part(changed, (64, 64), angle=-5)
    assert np.array_equal(restored.mask, before)


def test_clipped_manual_cut_is_restored_when_part_returns_to_canvas():
    part = asymmetric_part()
    part.mask[12:25, 10:12] = 0
    before = part.mask.copy()
    shifted = adjust_part(part, (64, 64), offset=(-20, 0))
    assert np.array_equal(shifted.mask, np.where(shifted.artwork[..., 3] > 0, 255, 0))
    restored = adjust_part(shifted, (64, 64), offset=(20, 0))
    assert np.array_equal(restored.mask, before)
    assert restored.alignment_edit_mask is shifted.alignment_edit_mask


def test_manual_mask_cuts_move_with_artwork_instead_of_resetting():
    part = propose().parts[0]
    part.mask[:, 12] = 0
    shifted = adjust_part(part, (32, 32), offset=(2, 0))
    expected = cv2.warpAffine(part.mask, np.array([[1, 0, 2], [0, 1, 0]], float), (32, 32),
                              flags=cv2.INTER_NEAREST)
    assert np.array_equal(shifted.mask, expected)
    assert shifted.mask[15, 14] == 0 and shifted.artwork[15, 14, 3] == 255
    assert shifted.alignment["manual_adjustments"][-1]["edited_mask_preserved"]


def test_adjust_rejects_plain_parts_repairs_extreme_values_and_offcanvas():
    with pytest.raises(ValueError):
        adjust_part(Part("plain", np.full((32, 32), 255, np.uint8)), (32, 32))
    part = propose().parts[0]
    for settings in [{"scale": True}, {"scale": -1}, {"angle": float("inf")},
                     {"offset": (1000, 0)}]:
        with pytest.raises(ValueError):
            adjust_part(part, (32, 32), **settings)
    part.hidden_mask = np.zeros((32, 32), np.uint8)
    changed = adjust_part(part, (32, 32), offset=(1, 0))
    assert changed.hidden_mask is None
    part.hidden_mask[10, 10] = 255
    with pytest.raises(ValueError, match="補完"):
        adjust_part(part, (32, 32), offset=(1, 0))


def test_opaque_atlas_requires_explicit_white_background_mode():
    atlas = np.full((32, 32, 4), 255, np.uint8)
    atlas[9:23, 9:23, :3] = [220, 60, 100]
    with pytest.raises(ValueError, match="透明"):
        propose(atlas=atlas)
    assert np.any(propose(atlas=atlas, background_mode="white").parts[0].mask)


def test_partially_offcanvas_crop_warns_and_manual_fit_updates_warning():
    part = propose().parts[0]
    changed = adjust_part(part, (32, 32), offset=(-12, 0))
    assert "transformed-crop-outside-canvas" in changed.alignment["warnings"]
    assert changed.alignment["transformed_crop_bbox"][0] == pytest.approx(-4)
    assert changed.alignment["initial_fit"]["anchor_residual_px"] == pytest.approx(0)
    assert changed.alignment["anchor_residual_px"] == pytest.approx(12)
    fitted = adjust_part(changed, (32, 32), offset=(12, 0))
    assert "transformed-crop-outside-canvas" not in fitted.alignment["warnings"]
