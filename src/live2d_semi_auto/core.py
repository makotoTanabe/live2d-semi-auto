"""Domain data and image operations. No UI or model-specific dependencies."""

from dataclasses import dataclass, field, replace
from uuid import uuid4

import numpy as np


@dataclass
class Part:
    name: str
    mask: np.ndarray
    kind: str = "custom"
    id: str = field(default_factory=lambda: uuid4().hex)
    visible: bool = True


@dataclass
class Project:
    source: np.ndarray
    source_name: str
    source_hash: str
    parts: list[Part] = field(default_factory=list)
    icc_profile: bytes | None = None
    original_path: str | None = None

    @property
    def size(self) -> tuple[int, int]:
        return self.source.shape[1], self.source.shape[0]

    def snapshot(self) -> "Project":
        # Source is read-only; only editable masks need copies for undo.
        return replace(self, parts=[replace(p, mask=p.mask.copy()) for p in self.parts])


def validate(project: Project, *, for_export: bool = False) -> list[str]:
    errors = []
    source = project.source
    if source.ndim != 3 or source.shape[2] != 4 or source.dtype != np.uint8:
        return ["原画は8bit RGBA画像である必要があります。"]
    if not source.shape[0] or not source.shape[1]:
        errors.append("キャンバスが空です。")
    ids, names = set(), set()
    for part in project.parts:
        if not part.id or part.id in ids:
            errors.append("パーツIDが空または重複しています。")
        ids.add(part.id)
        name = part.name.strip()
        if not name or name.casefold() in names:
            errors.append("パーツ名が空または重複しています。")
        names.add(name.casefold())
        if not part.kind.strip():
            errors.append(f"{part.name}: 種類が空です。")
        if part.mask.shape != source.shape[:2] or part.mask.dtype != np.uint8:
            errors.append(f"{part.name}: マスクとキャンバスが一致しません。")
        elif for_export and not np.any(part.mask):
            errors.append(f"{part.name}: マスクが空です。")
    if for_export and not project.parts:
        errors.append("出力するパーツがありません。")
    return errors


def layer_pixels(project: Project, part: Part) -> np.ndarray:
    result = project.source.copy()
    result[..., 3] = (
        result[..., 3].astype(np.uint16) * part.mask.astype(np.uint16) // 255
    ).astype(np.uint8)
    return result


def composite(project: Project) -> np.ndarray:
    """Straight-alpha source-over, parts stored bottom to top."""
    rgb = np.zeros((*project.source.shape[:2], 3), dtype=np.float32)
    alpha = np.zeros((*project.source.shape[:2], 1), dtype=np.float32)
    for part in project.parts:
        if not part.visible:
            continue
        pixels = layer_pixels(project, part).astype(np.float32) / 255
        a = pixels[..., 3:4]
        rgb = pixels[..., :3] * a + rgb * (1 - a)
        alpha = a + alpha * (1 - a)
    rgb = np.divide(rgb, alpha, out=np.zeros_like(rgb), where=alpha > 0)
    return np.rint(np.concatenate((rgb, alpha), axis=2) * 255).astype(np.uint8)


def mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1
