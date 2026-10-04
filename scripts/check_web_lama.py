"""Reproduce true local LaMa repair through the running Web application's API.

Uses only localhost HTTP and the existing NumPy/Pillow application dependencies.
The fixture withholds a visible hair region; it does not prove hidden anatomy.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import secrets
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zipfile

import numpy as np
from PIL import Image

from live2d_semi_auto.models import LAMA_SHA256


def pixels(raw):
    with Image.open(io.BytesIO(raw)) as image:
        return np.array(image)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check(base_url: str, project_path: Path, output: Path, *, timeout: float = 120):
    parsed = urlsplit(base_url)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("この確認スクリプトはlocalhostのHTTPサーバーだけに接続できます。")
    base_url = base_url.rstrip("/")
    if output.exists():
        raise ValueError("既存の成果物を守るため、新しい出力フォルダーを指定してください。")
    input_bytes = project_path.read_bytes()
    output.mkdir(parents=True)
    (output / "input.l2split").write_bytes(input_bytes)
    token = None

    def request(path, *, method="GET", data=None, content_type=None):
        headers = {}
        if token:
            headers["X-Session-Token"] = token
        if content_type:
            headers["Content-Type"] = content_type
        if isinstance(data, dict):
            data = json.dumps(data).encode()
            headers["Content-Type"] = "application/json"
        req = Request(base_url + path, data=data, method=method, headers=headers)
        try:
            with urlopen(req, timeout=timeout) as response:
                raw = response.read()
                if response.headers.get("Content-Type", "").startswith("application/json"):
                    return json.loads(raw)
                return raw
        except HTTPError as exc:
            # Endpoint is loopback-only; responses contain sanitized API errors.
            message = exc.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"Web API returned HTTP {exc.code}: {message}") from None

    def upload_project(raw, name="project.l2split"):
        boundary = "l2web-" + secrets.token_hex(16)
        data = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n").encode()
        data += raw + f"\r\n--{boundary}--\r\n".encode()
        return request("/api/import/project", method="POST", data=data,
                       content_type=f"multipart/form-data; boundary={boundary}")

    def ready(job):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = request(f"/api/jobs/{job['id']}")
            if state["status"] == "ready":
                return state
            if state["status"] != "running":
                raise RuntimeError(f"LaMa job {state['status']}: {state['message']}")
            time.sleep(0.15)
        raise TimeoutError("LaMa候補の作成が時間内に終わりませんでした。")

    def png(path, filename):
        raw = request(path)
        (output / filename).write_bytes(raw)
        return pixels(raw)

    started = time.monotonic()
    assert request("/api/health") == {"status": "ok"}
    session = request("/api/session", method="POST")
    token = session["session"]
    initial = upload_project(input_bytes, "input.l2split")
    write_json(output / "initial_state.json", initial)
    if not initial["capabilities"]["lama_available"]:
        raise ValueError("サーバーのLAMA_MODEL_PATHに検証済みLaMaモデルを設定してください。")
    if len(initial["parts"]) != 1 or initial["parts"][0]["generated"]:
        raise ValueError("未補完の1パーツfixture（samples/results/inpainting/before.l2split）を指定してください。")
    part_id = initial["parts"][0]["id"]
    source = png("/api/image/source.png", "source.png")
    before = png("/api/image/composite.png", "before.png")
    visible = png(f"/api/image/mask.png?part_id={part_id}", "visible_mask.png")
    hidden = png(f"/api/image/hidden.png?part_id={part_id}", "hidden_mask.png")
    assert np.any(hidden) and not np.any((hidden > 0) & (visible > 0))

    first = request("/api/proposals/repair", method="POST", data={"part_id": part_id, "backend": "lama"})
    first = ready(first)
    write_json(output / "first_proposal.json", first)
    assert first["metadata"]["backend"] == "big-lama-torchscript"
    assert first["metadata"]["ai"] is True
    assert first["metadata"]["model_sha256"] == LAMA_SHA256
    png(first["preview_url"], "first_preview.png")
    np.testing.assert_array_equal(pixels(request("/api/image/composite.png")), before)
    rejected = request(f"/api/jobs/{first['id']}/reject", method="POST")
    write_json(output / "rejected_state.json", rejected)
    rejected_pixels = png("/api/image/composite.png", "after_rejection.png")
    np.testing.assert_array_equal(rejected_pixels, before)
    assert rejected["revision"] == initial["revision"]
    assert rejected["source_hash"] == initial["source_hash"]
    assert rejected["parts"][0]["generated"] is False

    second = request("/api/proposals/repair", method="POST", data={"part_id": part_id, "backend": "lama"})
    second = ready(second)
    write_json(output / "second_proposal.json", second)
    preview = png(second["preview_url"], "accepted_preview.png")
    accepted = request(f"/api/jobs/{second['id']}/accept", method="POST")
    write_json(output / "accepted_state.json", accepted)
    after = png("/api/image/composite.png", "after.png")
    after_source = png("/api/image/source.png", "source_after.png")
    np.testing.assert_array_equal(after, preview)
    np.testing.assert_array_equal(after_source, source)
    np.testing.assert_array_equal(after[visible > 0], source[visible > 0])
    assert accepted["source_hash"] == initial["source_hash"]
    assert accepted["parts"][0]["id"] == part_id
    assert accepted["parts"][0]["generated"] is True
    assert np.all(after[..., 3][hidden > 0] == 255)
    assert np.all(before[..., 3][hidden > 0] == 0)

    saved = request("/api/export/project", method="POST")
    (output / "accepted.l2split").write_bytes(saved)
    with zipfile.ZipFile(io.BytesIO(saved)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        part = manifest["parts"][0]
        assert any(entry.get("operation") == "inpaint" and entry.get("ai") is True
                   and entry.get("model_sha256") == LAMA_SHA256 for entry in manifest["history"])
        generated_mask = pixels(archive.read(part["generated_mask"]))
        np.testing.assert_array_equal(generated_mask, hidden)
        (output / "generated_mask.png").write_bytes(archive.read(part["generated_mask"]))
        (output / "generated.png").write_bytes(archive.read(part["generated"]))
        write_json(output / "accepted_manifest.json", manifest)
    reloaded = upload_project(saved, "accepted.l2split")
    write_json(output / "reloaded_state.json", reloaded)
    roundtrip = png("/api/image/composite.png", "roundtrip.png")
    np.testing.assert_array_equal(roundtrip, after)
    assert reloaded["source_hash"] == initial["source_hash"]
    assert reloaded["parts"][0]["id"] == part_id
    assert reloaded["parts"][0]["generated"] is True
    reexported = request("/api/export/project", method="POST")
    (output / "roundtrip.l2split").write_bytes(reexported)
    with zipfile.ZipFile(io.BytesIO(reexported)) as archive:
        restored = json.loads(archive.read("manifest.json"))
        assert restored["history"] == manifest["history"]
        restored_part = restored["parts"][0]
        np.testing.assert_array_equal(pixels(archive.read(restored_part["generated"])),
                                      pixels((output / "generated.png").read_bytes()))
        np.testing.assert_array_equal(pixels(archive.read(restored_part["generated_mask"])), hidden)
    assert project_path.read_bytes() == input_bytes
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "transport": "localhost HTTP Web API; no external AI endpoint",
        "fixture": "samples/results/inpainting/before.l2split",
        "purpose": "Synthetic withholding of visible hair pixels; not ground-truth hidden anatomy",
        "source_hash": initial["source_hash"], "part_id": part_id,
        "size": initial["size"], "target_pixels": int(np.count_nonzero(hidden)),
        "inpainting": second["metadata"], "elapsed_seconds": round(time.monotonic() - started, 3),
        "checks": {"true_local_neural_inference": True, "preview_not_auto_adopted": True,
                   "rejection_preserved_state": True, "accepted_matches_preview": True,
                   "source_pixels_and_hash_preserved": True, "visible_pixels_preserved": True,
                   "hidden_region_filled": True, "stable_part_id": True,
                   "generated_pixels_masks_history_roundtrip": True, "input_archive_unchanged": True},
        "artifacts_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in sorted(output.iterdir()) if path.is_file()},
    }
    write_json(output / "report.json", report)
    print(json.dumps({"output": str(output), "checks": report["checks"],
                      "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="実LaMaをWeb API経由でプレビュー・採用・保存確認")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--project", type=Path, default=Path("samples/results/inpainting/before.l2split"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    check(args.base_url, args.project, args.output, timeout=args.timeout)
