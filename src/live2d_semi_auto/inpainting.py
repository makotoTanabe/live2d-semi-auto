"""Local repair backends. The domain stays independent of neural models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .core import mask_bounds
from .models import LAMA_SHA256, verify_lama


@dataclass
class RepairProposal:
    pixels: np.ndarray
    mask: np.ndarray
    metadata: dict


class InpaintingBackend(Protocol):
    def propose(self, source: np.ndarray, mask: np.ndarray) -> RepairProposal: ...


def check_mask(source: np.ndarray, mask: np.ndarray):
    if mask.shape != source.shape[:2] or mask.dtype != np.uint8 or not np.any(mask):
        raise ValueError("補完する領域を補完マスクに描いてください。")
    if np.all(mask > 0):
        raise ValueError("補完にはマスクの外側に参照画素が必要です。")


class TeleaBackend:
    """Traditional image processing, deliberately not described as AI."""

    def propose(self, source: np.ndarray, mask: np.ndarray) -> RepairProposal:
        check_mask(source, mask)
        rgb = cv2.inpaint(source[..., :3], mask, 3, cv2.INPAINT_TELEA)
        pixels = source.copy()
        pixels[mask > 0, :3] = rgb[mask > 0]
        pixels[mask > 0, 3] = 255
        return RepairProposal(pixels, mask.copy(), {"backend": "opencv-telea", "ai": False})


class LaMaBackend:
    """Pinned TorchScript LaMa on CPU; no automatic network access."""

    def __init__(self, path: str | Path):
        self.path = path

    def propose(self, source: np.ndarray, mask: np.ndarray) -> RepairProposal:
        check_mask(source, mask)
        path = verify_lama(self.path)
        try:
            import torch
        except ImportError as exc:
            raise ValueError("AI補完には uv sync --extra ai を実行してください。") from exc
        left, top, right, bottom = mask_bounds(mask)
        left, top = max(0, left - 64), max(0, top - 64)
        right, bottom = min(source.shape[1], right + 64), min(source.shape[0], bottom + 64)
        width, height = right - left, bottom - top
        scale = min(1.0, 512 / max(width, height))
        size = max(1, round(width * scale)), max(1, round(height * scale))
        rgb = cv2.resize(source[top:bottom, left:right, :3], size, interpolation=cv2.INTER_AREA)
        target = cv2.resize(mask[top:bottom, left:right], size, interpolation=cv2.INTER_NEAREST)
        if not np.any(target):
            raise ValueError("補完領域が縮小後に消えました。領域を広げてください。")
        pad_x, pad_y = (-size[0]) % 8, (-size[1]) % 8
        rgb = np.pad(rgb, ((0, pad_y), (0, pad_x), (0, 0)), mode="edge")
        target = np.pad(target, ((0, pad_y), (0, pad_x)), mode="constant")
        torch.set_num_threads(4)
        model = torch.jit.load(str(path), map_location="cpu").eval()
        image_tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None] / 255
        mask_tensor = torch.from_numpy((target > 0).astype(np.float32))[None, None]
        with torch.inference_mode():
            output = model(image_tensor, mask_tensor)[0].permute(1, 2, 0).numpy()
        if not np.all(np.isfinite(output)):
            raise ValueError("AI補完結果に不正な画素が含まれています。")
        rgb = np.rint(np.clip(output[:size[1], :size[0]], 0, 1) * 255).astype(np.uint8)
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        pixels = source.copy()
        crop = pixels[top:bottom, left:right]
        selected = mask[top:bottom, left:right] > 0
        crop[selected, :3] = rgb[selected]
        crop[selected, 3] = 255
        return RepairProposal(pixels, mask.copy(), {
            "backend": "big-lama-torchscript", "ai": True, "model_sha256": LAMA_SHA256,
            "source_size": [source.shape[1], source.shape[0]], "crop": [left, top, right, bottom],
            "inference_size": list(size), "padding": [pad_x, pad_y],
            "scale": [size[0] / width, size[1] / height], "offset": [left, top],
        })
