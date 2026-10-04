"""Exercise the generated character without relying on semantic AI labels."""

from pathlib import Path

import numpy as np
from PIL import Image

from live2d_semi_auto.application import Editor
from live2d_semi_auto.core import composite
from live2d_semi_auto.infrastructure import export_png, import_image, load_project, save_project


def test_original_character_roundtrip(tmp_path):
    path = Path(__file__).parents[1] / "samples" / "original_character.png"
    before = path.read_bytes()
    project = import_image(path)
    editor = Editor(project)
    top, bottom = editor.add_part(), editor.add_part()
    midpoint = project.size[1] // 2
    top.name, bottom.name = "top", "bottom"
    top.mask[:midpoint] = 255
    bottom.mask[midpoint:] = 255
    saved = tmp_path / "sample.l2split"
    save_project(project, saved)
    restored = load_project(saved)
    assert np.array_equal(composite(restored), project.source)
    exported = tmp_path / "export"
    export_png(restored, exported)
    with Image.open(exported / "preview.png") as preview:
        assert np.array_equal(np.array(preview), project.source)
    assert path.read_bytes() == before
