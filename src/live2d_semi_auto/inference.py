"""Explicit local proposals; no model downloads or network requests."""

from dataclasses import dataclass, field
from typing import Protocol

import cv2

import numpy as np

from .core import Part


class SegmentationBackend(Protocol):
    def propose(self, source: np.ndarray) -> np.ndarray: ...


class AlphaBackend:
    """A baseline, not semantic segmentation: propose nontransparent pixels."""

    def propose(self, source: np.ndarray) -> np.ndarray:
        return np.where(source[..., 3] > 0, 255, 0).astype(np.uint8)


@dataclass
class PartsProposal:
    parts: list[Part]
    metadata: dict
    assets: dict[str, np.ndarray] = field(default_factory=dict)


class ColorPartsBackend:
    """Deterministic color clustering; proposals have no semantic labels."""

    def __init__(self, count: int = 8):
        if not 2 <= count <= 16:
            raise ValueError("候補数は2〜16にしてください。")
        self.count = count

    def propose_parts(self, source: np.ndarray) -> PartsProposal:
        lab = cv2.cvtColor(source[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)
        foreground = source[..., 3] > 0
        pixels = lab[foreground]
        if not len(pixels):
            raise ValueError("非透明の画素がありません。")
        samples = pixels[::max(1, len(pixels) // 16384)]
        # Farthest-point initialization makes results repeatable without RNG.
        centers = [samples.mean(axis=0)]
        for _ in range(self.count - 1):
            distances = np.min([np.sum((samples - c) ** 2, axis=1) for c in centers], axis=0)
            if distances.max() < 1:
                break
            centers.append(samples[distances.argmax()])
        centers = np.asarray(centers)
        for _ in range(15):
            labels = np.argmin([np.sum((samples - c) ** 2, axis=1) for c in centers], axis=0)
            updated = np.array([samples[labels == i].mean(axis=0) if np.any(labels == i) else c
                                for i, c in enumerate(centers)])
            if np.allclose(updated, centers, atol=0.1):
                break
            centers = updated
        labels = np.zeros(source.shape[:2], dtype=np.uint8)
        # Work in row chunks: memory stays bounded on large canvases.
        for y in range(0, source.shape[0], 128):
            chunk = lab[y:y + 128]
            labels[y:y + 128] = np.argmin(
                [np.sum((chunk - c) ** 2, axis=2) for c in centers], axis=0)
        parts = []
        for index in range(len(centers)):
            mask = np.where(foreground & (labels == index), 255, 0).astype(np.uint8)
            if np.any(mask):
                parts.append(Part(f"color_region_{len(parts) + 1:02d}", mask, "color_region"))
        return PartsProposal(parts, {"backend": "color-clusters-v1", "requested_count": self.count,
                                     "actual_count": len(parts), "semantic_labels": False,
                                     "canvas": [source.shape[1], source.shape[0]]})
