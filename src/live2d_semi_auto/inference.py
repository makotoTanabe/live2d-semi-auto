"""Explicit local proposals; no model downloads or network requests."""

from typing import Protocol

import numpy as np


class SegmentationBackend(Protocol):
    def propose(self, source: np.ndarray) -> np.ndarray: ...


class AlphaBackend:
    """A baseline, not semantic segmentation: propose nontransparent pixels."""

    def propose(self, source: np.ndarray) -> np.ndarray:
        return np.where(source[..., 3] > 0, 255, 0).astype(np.uint8)
