"""HTTP workflow checks: independent sessions, reversible edits and safe jobs."""

import io
import json
import threading
import time
import zipfile

import numpy as np
from PIL import Image, ImageCms
from psd_tools import PSDImage
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from live2d_semi_auto.core import Part
from live2d_semi_auto.inference import PartsProposal
from live2d_semi_auto.web import create_app


def png(size=(20, 16)):
    pixels = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    pixels[2:-2, 2:-2] = [192, 80, 128, 255]
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, "PNG")
    return stream.getvalue()


@pytest.fixture
def client():
    app = create_app(allowed_hosts=["testserver"])
    with TestClient(app) as client:
        token = client.post("/api/session").json()["session"]
        client.headers["X-Session-Token"] = token
        yield client


def imported(client):
    response = client.post("/api/import/image", files={"file": ("art.png", png(), "image/png")})
    assert response.status_code == 200, response.text
    return response.json()


def painted(client):
    imported(client)
    part = client.post("/api/parts").json()["parts"][0]
    response = client.post(f"/api/parts/{part['id']}/mask", json={
        "tool": "lasso", "points": [[2, 2], [17, 2], [17, 13], [2, 13]],
    })
    assert response.status_code == 200, response.text
    return part["id"]


def ready(client, job_id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = client.get(f"/api/jobs/{job_id}").json()
        if result["status"] != "running":
            return result
        threading.Event().wait(0.01)
    pytest.fail("Background job did not finish")


def test_sessions_are_independent_and_expire():
    app = create_app(allowed_hosts=["testserver"], session_ttl=10, max_sessions=2)
    with TestClient(app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        assert client.get("/api/state").status_code == 401
        first = client.post("/api/session").json()["session"]
        second = client.post("/api/session").json()["session"]
        assert first != second
        assert client.post("/api/session").status_code == 429
        client.headers["X-Session-Token"] = first
        imported(client)
        client.headers["X-Session-Token"] = second
        assert client.get("/api/state").json()["has_project"] is False
        app.state.sessions[first].accessed -= 11
        client.headers["X-Session-Token"] = first
        assert client.get("/api/state").status_code == 401
        assert client.post("/api/session").status_code == 200


def test_foreign_origins_and_hosts_are_rejected(client):
    assert client.post("/api/session", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/session", headers={"Origin": "http://testserver"}).status_code == 200
    assert client.get("/api/health", headers={"Host": "evil.example"}).status_code == 400
    assert client.post("/api/session", headers={"Origin": "null"}).status_code == 403


def test_edit_mask_undo_rename_and_stable_ids(client):
    part_id = painted(client)
    state = client.get("/api/state").json()
    assert state["parts"][0]["bounds"] == [2, 2, 18, 14]
    assert state["can_undo"] is True
    response = client.patch(f"/api/parts/{part_id}", json={"name": "face", "kind": "face", "visible": False})
    assert response.json()["parts"][0]["id"] == part_id
    assert response.json()["parts"][0]["visible"] is False
    restored = client.post("/api/history", json={"direction": "undo"}).json()
    assert restored["parts"][0]["visible"] is True
    restored = client.post("/api/history", json={"direction": "undo"}).json()
    assert restored["parts"][0]["bounds"] is None
    assert restored["parts"][0]["id"] == part_id
    redone = client.post("/api/history", json={"direction": "redo"}).json()
    assert redone["parts"][0]["bounds"] == [2, 2, 18, 14]
    mask = Image.open(io.BytesIO(client.get(f"/api/image/mask.png?part_id={part_id}").content))
    assert mask.mode == "L" and np.asarray(mask)[3, 3] == 255
    before = client.get("/api/state").json()["revision"]
    assert client.post(f"/api/parts/{part_id}/mask", json={"points": [[-1, 2]], "radius": 3}).status_code == 422
    assert client.get("/api/state").json()["revision"] == before
    assert client.patch(f"/api/parts/{part_id}", json={"visible": "false"}).status_code == 422


def test_export_roundtrip_and_psd_pixels(client):
    part_id = painted(client)
    client.patch(f"/api/parts/{part_id}", json={"name": "顔"})
    archive = client.post("/api/export/project")
    assert archive.status_code == 200
    assert "project.l2split" in archive.headers["content-disposition"]
    saved = client.post("/api/import/project", files={"file": ("saved.l2split", archive.content)})
    assert saved.status_code == 200, saved.text
    assert saved.json()["parts"][0]["id"] == part_id
    assert saved.json()["parts"][0]["name"] == "顔"
    expected = np.array(Image.open(io.BytesIO(client.get("/api/image/composite.png").content)))
    result = client.post("/api/export/png")
    with zipfile.ZipFile(io.BytesIO(result.content)) as package:
        manifest = json.loads(package.read("manifest.json"))
        assert manifest["parts"][0]["id"] == part_id
        actual = np.array(Image.open(io.BytesIO(package.read(manifest["parts"][0]["file"]))))
        np.testing.assert_array_equal(actual, expected)
    result = client.post("/api/export/psd")
    assert result.status_code == 200, result.text
    psd = PSDImage.open(io.BytesIO(result.content))
    assert psd[0].name == "顔"
    psd_pixels = np.array(psd.composite())
    np.testing.assert_array_equal(psd_pixels[..., 3], expected[..., 3])
    np.testing.assert_array_equal(psd_pixels[expected[..., 3] > 0], expected[expected[..., 3] > 0])


def test_color_proposals_require_review_and_undo(client):
    imported(client)
    job = client.post("/api/proposals/color", json={"count": 2}).json()
    result = ready(client, job["id"])
    assert result["status"] == "ready"
    assert client.get("/api/state").json()["parts"] == []
    assert client.post("/api/parts").status_code == 409
    assert client.get(result["preview_url"]).status_code == 200
    accepted = client.post(f"/api/jobs/{job['id']}/accept").json()
    assert accepted["parts"]
    assert accepted["active_job"] is None
    assert client.post(f"/api/jobs/{job['id']}/accept").status_code == 409
    undone = client.post("/api/history", json={"direction": "undo"}).json()
    assert undone["parts"] == []


def test_cancelled_job_discards_result_and_keeps_new_edits(client, monkeypatch):
    imported(client)
    started, release = threading.Event(), threading.Event()
    def delayed(self, source):
        started.set()
        assert release.wait(3)
        return PartsProposal([Part("proposal", np.full(source.shape[:2], 255, np.uint8))], {})
    monkeypatch.setattr("live2d_semi_auto.web.ColorPartsBackend.propose_parts", delayed)
    job = client.post("/api/proposals/color", json={"count": 2}).json()
    assert started.wait(2)
    client.post(f"/api/jobs/{job['id']}/reject")
    part = client.post("/api/parts").json()["parts"][0]
    release.set()
    client.app.state.sessions[client.headers["X-Session-Token"]].jobs[job["id"]].future.result(3)
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "cancelled"
    assert client.get("/api/state").json()["parts"][0]["id"] == part["id"]
    assert client.post(f"/api/jobs/{job['id']}/accept").status_code == 409


def test_failed_job_keeps_manual_work_and_never_exposes_key(client, monkeypatch):
    part_id = painted(client)
    calls = []
    def fail(self, source):
        calls.append(True)
        raise ValueError("failure sk-test-secret-do-not-return")
    monkeypatch.setattr("live2d_semi_auto.web.GPTPartsBackend.propose_parts", fail)
    secret = "sk-test-secret-do-not-return"
    rejected = client.post("/api/proposals/gpt", json={"api_key": secret, "consent": False})
    assert rejected.status_code == 422 and not calls
    invalid = client.post("/api/proposals/gpt", json={"api_key": secret, "consent": "yes"})
    assert invalid.status_code == 422 and secret not in invalid.text
    job = client.post("/api/proposals/gpt", json={"api_key": secret, "consent": True}).json()
    result = ready(client, job["id"])
    assert result["status"] == "failed" and calls
    assert secret not in json.dumps(result)
    state = client.get("/api/state").json()
    assert secret not in json.dumps(state)
    assert state["parts"][0]["id"] == part_id
    assert client.post("/api/parts").status_code == 200


def test_stale_job_cannot_replace_project(client, monkeypatch):
    imported(client)
    job = client.post("/api/proposals/color", json={"count": 2}).json()
    assert ready(client, job["id"])["status"] == "ready"
    session = client.app.state.sessions[client.headers["X-Session-Token"]]
    session.revision += 1  # Independent version guard, beyond normal mutation freeze.
    assert client.post(f"/api/jobs/{job['id']}/accept").status_code == 409
    assert client.get("/api/state").json()["parts"] == []
    assert client.get("/api/state").json()["active_job"] is None


def test_invalid_import_does_not_replace_project(client):
    part_id = painted(client)
    assert client.post("/api/import/image", files={"file": ("bad.png", b"invalid")}).status_code == 422
    assert client.post("/api/import/image", files={"file": ("huge.png", png((4097, 1)))}).status_code == 422
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("manifest.json", "{}")
        package.writestr("../escape.png", png())
    assert client.post("/api/import/project", files={"file": ("bad.l2split", archive.getvalue())}).status_code == 422
    assert client.get("/api/state").json()["parts"][0]["id"] == part_id
    assert client.get("/api/image/layer.png?part_id=missing").status_code == 404


def test_archive_pixel_limits_apply_to_non_png_names_and_duplicate_references(client):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("manifest.json", json.dumps({"source": "pixels.bin", "parts": [], "assets": {}}))
        package.writestr("pixels.bin", png((4097, 1)))
    result = client.post("/api/import/project", files={"file": ("bad.l2split", archive.getvalue())})
    assert result.status_code == 422
    assert "4096" in result.text


def test_runtime_binding_deletion_and_undo_stays_valid(client):
    part_id = painted(client)
    client.patch("/api/runtime/config", json={"bindings": {part_id: {"role": "face"}}})
    result = client.delete(f"/api/parts/{part_id}")
    assert result.status_code == 200
    assert result.json()["runtime_config"]["bindings"] == {}
    result = client.post("/api/history", json={"direction": "undo"})
    assert result.json()["parts"][0]["id"] == part_id
    assert result.json()["runtime_config"]["bindings"][part_id]["role"] == "face"


def test_runtime_sprites_do_not_reapply_original_color_profile(client, monkeypatch):
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
    stream = io.BytesIO()
    Image.new("RGBA", (20, 16), (20, 40, 60, 255)).save(stream, "PNG", icc_profile=profile)
    result = client.post("/api/import/image", files={"file": ("profile.png", stream.getvalue())})
    assert result.status_code == 200
    part_id = client.post("/api/parts").json()["parts"][0]["id"]
    converted = np.full((2, 3, 4), [64, 80, 96, 255], np.uint8)
    monkeypatch.setattr("live2d_semi_auto.web_export.layer_sprite", lambda project, part: converted)
    source = Image.open(io.BytesIO(client.get("/api/image/source.png").content))
    assert source.info["icc_profile"] == profile
    sprite = Image.open(io.BytesIO(client.get(f"/api/runtime/layers/{part_id}.png").content))
    assert "icc_profile" not in sprite.info
    np.testing.assert_array_equal(np.array(sprite), converted)


def test_upload_size_is_bounded_before_multipart_parsing(client, monkeypatch):
    monkeypatch.setattr("live2d_semi_auto.web.MAX_UPLOAD", 32)
    response = client.post("/api/import/image", files={"file": ("art.png", png())})
    assert response.status_code == 413
    response = client.post("/api/import/image", content=b"x" * (1024 * 1024 + 33))
    assert response.status_code == 413


def test_job_queue_is_bounded_after_repeated_cancellation(client, monkeypatch):
    imported(client)
    release = threading.Event()
    def delayed(self, source):
        assert release.wait(3)
        return PartsProposal([Part("proposal", np.full(source.shape[:2], 255, np.uint8))], {})
    monkeypatch.setattr("live2d_semi_auto.web.ColorPartsBackend.propose_parts", delayed)
    # Two running jobs cannot be aborted; two more are queued. Cancelling queued
    # jobs releases their slots, while each running job still occupies one.
    jobs = []
    for _ in range(2):
        response = client.post("/api/proposals/color", json={"count": 2})
        assert response.status_code == 200
        jobs.append(response.json()["id"])
        threading.Event().wait(0.02)
        client.post(f"/api/jobs/{jobs[-1]}/reject")
    queued_jobs = []
    for _ in range(2):
        token = client.post("/api/session").json()["session"]
        client.headers["X-Session-Token"] = token
        imported(client)
        response = client.post("/api/proposals/color", json={"count": 2})
        assert response.status_code == 200
        queued_jobs.append((token, response.json()["id"]))
    token = client.post("/api/session").json()["session"]
    client.headers["X-Session-Token"] = token
    imported(client)
    assert client.post("/api/proposals/color", json={"count": 2}).status_code == 429
    for token, job_id in queued_jobs:
        client.headers["X-Session-Token"] = token
        client.post(f"/api/jobs/{job_id}/reject")
    release.set()
    for session in client.app.state.sessions.values():
        for job_id in jobs:
            if job_id in session.jobs:
                session.jobs[job_id].future.result(3)


def test_editing_budget_matches_project_import_budget(client, monkeypatch):
    monkeypatch.setattr("live2d_semi_auto.web.MAX_ARCHIVE_PIXELS", 20 * 16 * 2)
    part_id = painted(client)  # Source + one mask exactly fills budget.
    state = client.get("/api/state").json()
    assert client.post("/api/parts").status_code == 422
    assert client.post(f"/api/parts/{part_id}/mask", json={
        "hidden": True, "points": [[0, 0]], "radius": 1,
    }).status_code == 422
    assert client.get("/api/state").json()["revision"] == state["revision"]
    exported = client.post("/api/export/project")
    assert exported.status_code == 200
    loaded = client.post("/api/import/project", files={"file": ("project.l2split", exported.content)})
    assert loaded.status_code == 200
    assert loaded.json()["parts"][0]["id"] == part_id


def test_over_budget_proposal_never_replaces_manual_work(client, monkeypatch):
    part_id = painted(client)
    monkeypatch.setattr("live2d_semi_auto.web.MAX_ARCHIVE_PIXELS", 20 * 16 * 2)
    job = client.post("/api/proposals/color", json={"count": 2}).json()
    result = ready(client, job["id"])
    assert result["status"] == "failed"
    assert "候補数" in result["message"]
    assert client.get("/api/state").json()["parts"][0]["id"] == part_id


def test_downloadable_project_must_meet_reimport_size_limit(client, monkeypatch):
    part_id = painted(client)
    monkeypatch.setattr("live2d_semi_auto.web.MAX_UPLOAD", 100)
    result = client.post("/api/export/project")
    assert result.status_code == 422
    assert "32MiB" in result.text
    assert client.get("/api/state").json()["parts"][0]["id"] == part_id


def test_runtime_roles_are_reversible_and_saved(client):
    part_id = painted(client)
    result = client.patch("/api/runtime/config", json={"bindings": {part_id: {"role": "face"}}})
    assert result.status_code == 200, result.text
    assert result.json()["runtime_config"]["bindings"][part_id]["role"] == "face"
    model = client.get("/api/runtime/model")
    assert model.status_code == 200
    sprite = client.get(f"/api/runtime/layers/{part_id}.png")
    assert sprite.status_code == 200
    assert Image.open(io.BytesIO(sprite.content)).size == (16, 12)
    bad = client.patch("/api/runtime/config", json={"bindings": {"unknown": {"role": "face"}}})
    assert bad.status_code == 422
    saved = client.post("/api/export/project").content
    client.post("/api/history", json={"direction": "undo"})
    assert client.get("/api/state").json()["runtime_config"]["bindings"].get(part_id, {}).get("role") != "face"
    loaded = client.post("/api/import/project", files={"file": ("project.l2split", saved)})
    assert loaded.json()["runtime_config"]["bindings"][part_id]["role"] == "face"
    bundle = client.post("/api/export/web")
    assert bundle.status_code == 200, bundle.text
    with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
        assert "index.html" in archive.namelist()
        assert any(name.endswith(".json") for name in archive.namelist())


def test_alignment_requires_upload_consent_before_backend(client, monkeypatch):
    imported(client)
    def forbidden(*args, **kwargs):
        pytest.fail("Remote inference ran without consent")
    monkeypatch.setattr("live2d_semi_auto.web.GPTAlignmentBackend.propose_alignment", forbidden)
    response = client.post("/api/proposals/alignment", files={"file": ("atlas.png", png())},
                           data={"consent": "false", "api_key": "sk-test-secret"})
    assert response.status_code == 422
    assert "sk-test-secret" not in response.text


def test_alignment_and_adjustment_are_previewed_then_accepted(client, monkeypatch):
    source = png((32, 32))
    assert client.post("/api/import/image", files={"file": ("source.png", source)}).status_code == 200
    response = {"choices": [{"message": {"content": json.dumps({"parts": [{
        "name": "face", "kind": "face", "atlas_bbox": [0, 0, 1000, 1000],
        "anchors": [{"source": [250, 250], "target": [250, 250]},
                    {"source": [750, 250], "target": [750, 250]}],
        "confidence": 0.9, "notes": "Synthetic corresponding anchors",
    }]})}}]}
    monkeypatch.setattr("live2d_semi_auto.web.GPTAlignmentBackend._request", lambda self, body: response)
    submitted = client.post("/api/proposals/alignment", files={"file": ("atlas.png", source)},
                            data={"consent": "true", "api_key": "sk-test-ephemeral", "background": "alpha"})
    assert submitted.status_code == 200, submitted.text
    job_id = submitted.json()["id"]
    assert ready(client, job_id)["status"] == "ready"
    accepted = client.post(f"/api/jobs/{job_id}/accept").json()
    part = accepted["parts"][0]
    assert part["aligned"] is True and part["bounds"] == [2, 2, 30, 30]
    original_matrix = part["alignment"]["matrix"]
    submitted = client.post(f"/api/parts/{part['id']}/adjustment", json={"offset": [1, 0]})
    adjustment_id = submitted.json()["id"]
    assert ready(client, adjustment_id)["status"] == "ready"
    assert client.get("/api/state").json()["parts"][0]["alignment"]["matrix"] == original_matrix
    adjusted = client.post(f"/api/jobs/{adjustment_id}/accept").json()["parts"][0]
    assert adjusted["id"] == part["id"] and adjusted["bounds"] == [3, 2, 31, 30]
    undone = client.post("/api/history", json={"direction": "undo"}).json()["parts"][0]
    assert undone["alignment"]["matrix"] == original_matrix
    saved = client.post("/api/export/project")
    assert b"sk-test-ephemeral" not in saved.content
    restored = client.post("/api/import/project", files={"file": ("saved.l2split", saved.content)})
    assert restored.status_code == 200
    assert restored.json()["parts"][0]["alignment"]["matrix"] == original_matrix


def test_telea_repair_is_reviewable_and_preserves_visible_pixels(client):
    part_id = painted(client)
    response = client.post(f"/api/parts/{part_id}/mask", json={
        "hidden": True, "radius": 1, "points": [[0, 0]],
    })
    assert response.status_code == 200
    job = client.post("/api/proposals/repair", json={"part_id": part_id, "backend": "telea"}).json()
    result = ready(client, job["id"])
    assert result["status"] == "ready", result
    assert client.get("/api/state").json()["parts"][0]["generated"] is False
    state = client.post(f"/api/jobs/{job['id']}/accept").json()
    assert state["parts"][0]["generated"] is True
    source = np.asarray(Image.open(io.BytesIO(client.get("/api/image/source.png").content)))
    layer = np.asarray(Image.open(io.BytesIO(client.get(f"/api/image/layer.png?part_id={part_id}").content)))
    np.testing.assert_array_equal(source[2:14, 2:18], layer[2:14, 2:18])
