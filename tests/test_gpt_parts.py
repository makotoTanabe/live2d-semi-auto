import base64
import io
import json

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QMessageBox
import pytest

from live2d_semi_auto.application import Editor
from live2d_semi_auto.gpt_parts import GPTPartsBackend


def response(parts):
    return {"choices": [{"message": {"content": json.dumps({"parts": parts})}}]}


def test_gpt_structured_proposals_and_source_preservation(project):
    received = []
    def transport(body):
        received.append(body)
        return response([{"name": "face", "kind": "face", "bbox": [100, 100, 900, 900]}])
    before = project.source.copy()
    proposal = GPTPartsBackend(transport=transport).propose_parts(project.source)
    body = received[0]
    assert body["response_format"]["json_schema"]["strict"]
    encoded = body["messages"][0]["content"][1]["image_url"]["url"].split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        assert image.mode == "RGB" and image.size == project.size
    assert proposal.parts[0].name == "face"
    assert proposal.metadata["parts"][0]["canvas_bbox"] == [1, 1, 15, 11]
    assert np.any(proposal.parts[0].mask)
    assert np.all(proposal.parts[0].mask[project.source[..., 3] == 0] == 0)
    assert np.array_equal(project.source, before)
    assert project.parts == []
    editor = Editor(project)
    editor.accept_parts(proposal)
    assert editor.project.history[-1]["backend"] == "gpt-vision-grabcut"
    editor.undo()
    assert editor.project.parts == []


@pytest.mark.parametrize("parts", [
    [], [{"name": "face", "kind": "face", "bbox": [900, 0, 100, 1000]}],
    [{"name": "face", "kind": "face", "bbox": [0, 0, 1001, 1000]}],
    [{"name": "face", "kind": "face", "bbox": [0, 0, True, 1000]}],
    [{"name": "", "kind": "face", "bbox": [0, 0, 1000, 1000]}],
    [{"name": "face", "kind": "face", "bbox": [0, 0, 1000, 1000]}] * 2,
])
def test_invalid_gpt_response_does_not_change_project(project, parts):
    backend = GPTPartsBackend(transport=lambda body: response(parts))
    with pytest.raises(ValueError):
        backend.propose_parts(project.source)
    assert project.parts == []


def test_gpt_refusal(project):
    backend = GPTPartsBackend(transport=lambda body: {"choices": [{"message": {"refusal": "cannot analyze"}}]})
    with pytest.raises(ValueError, match="候補を返しません"):
        backend.propose_parts(project.source)


def test_gpt_credentials_missing_no_network(project, monkeypatch):
    monkeypatch.delenv("GPT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("must not send without credentials"))
    with pytest.raises(ValueError, match="GPT_API_KEY"):
        GPTPartsBackend().propose_parts(project.source)


def test_gpt_image_resize_transform():
    source = np.full((600, 1200, 4), 255, np.uint8)
    proposal = GPTPartsBackend(transport=lambda body: response([
        {"name": "whole", "kind": "test", "bbox": [0, 0, 1000, 1000]},
    ])).propose_parts(source)
    assert proposal.metadata["sent_size"] == [1024, 512]
    assert proposal.metadata["parts"][0]["canvas_bbox"] == [0, 0, 1200, 600]
    assert proposal.metadata["parts"][0]["mask_method"] == "rectangle-fallback"
