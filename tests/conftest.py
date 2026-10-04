import os

import numpy as np
from PIL import Image
import pytest

from live2d_semi_auto.infrastructure import import_image


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def project(tmp_path):
    pixels = np.zeros((12, 16, 4), dtype=np.uint8)
    pixels[2:10, 2:14] = [230, 110, 140, 255]
    pixels[4:8, 6:10] = [100, 50, 80, 128]
    path = tmp_path / "character.png"
    Image.fromarray(pixels).save(path)
    return import_image(path)
