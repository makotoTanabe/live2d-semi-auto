"""Domain data and image operations. No UI or model-specific dependencies."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
import re
from uuid import uuid4

import numpy as np


@dataclass
class Part:
    name: str
    mask: np.ndarray
    kind: str = "custom"
    id: str = field(default_factory=lambda: uuid4().hex)
    visible: bool = True
    hidden_mask: np.ndarray | None = None
    generated: np.ndarray | None = None
    generated_mask: np.ndarray | None = None
    artwork: np.ndarray | None = None
    asset: np.ndarray | None = None
    asset_mask: np.ndarray | None = None
    alignment: dict | None = None
    alignment_edit_mask: np.ndarray | None = None


@dataclass
class Project:
    source: np.ndarray
    source_name: str
    source_hash: str
    parts: list[Part] = field(default_factory=list)
    icc_profile: bytes | None = None
    original_path: str | None = None
    history: list[dict] = field(default_factory=list)
    assets: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        return self.source.shape[1], self.source.shape[0]

    def snapshot(self) -> "Project":
        # Source is read-only; only editable masks need copies for undo.
        return replace(self, parts=[replace(
            p, mask=p.mask.copy(),
            hidden_mask=p.hidden_mask.copy() if p.hidden_mask is not None else None,
            alignment=deepcopy(p.alignment),
        ) for p in self.parts], history=deepcopy(self.history), assets=dict(self.assets))


def _rgba(pixels: object, shape: tuple | None = None) -> bool:
    return (isinstance(pixels, np.ndarray) and pixels.dtype == np.uint8
            and pixels.ndim == 3 and pixels.shape[2] == 4
            and pixels.shape[0] > 0 and pixels.shape[1] > 0
            and (shape is None or pixels.shape == shape))


def _finite_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (OverflowError, ValueError):
        return False


def _valid_matrix(values: object) -> bool:
    if (not isinstance(values, list) or len(values) != 2
            or not all(isinstance(row, list) and len(row) == 3
                       and all(_finite_number(value) for value in row) for row in values)):
        return False
    matrix = np.asarray(values, dtype=np.float64)
    return bool(abs(np.linalg.det(matrix[:, :2])) >= 1e-12)


def _alignment_errors(project: Project, part: Part) -> list[str]:
    """Validate storage/coordinate provenance without depending on any model."""
    fields = (part.artwork, part.asset, part.asset_mask, part.alignment)
    if all(value is None for value in fields):
        return ([f"{part.name}: 編集基準マスクには配置情報が必要です。"]
                if part.alignment_edit_mask is not None else [])
    label = f"{part.name}: "
    if any(value is None for value in fields):
        return [label + "配置画像・元パーツ・抽出マスク・配置情報を一緒に保存してください。"]
    errors = []
    if not _rgba(part.artwork, project.source.shape):
        errors.append(label + "配置画像はキャンバスと同じ8bit RGBAである必要があります。")
    elif part.artwork.flags.writeable:
        errors.append(label + "配置画像は読み取り専用である必要があります。")
    if not _rgba(part.asset):
        errors.append(label + "元パーツは8bit RGBA画像である必要があります。")
    elif part.asset.flags.writeable:
        errors.append(label + "元パーツは読み取り専用である必要があります。")
    if (not isinstance(part.asset_mask, np.ndarray) or part.asset_mask.dtype != np.uint8
            or part.asset_mask.ndim != 2
            or (_rgba(part.asset) and part.asset_mask.shape != part.asset.shape[:2])):
        errors.append(label + "抽出マスクと元パーツの寸法・形式が一致しません。")
    elif part.asset_mask.flags.writeable:
        errors.append(label + "抽出マスクは読み取り専用である必要があります。")
    info = part.alignment
    if not isinstance(info, dict):
        return errors + [label + "配置情報が不正です。"]
    try:
        json.dumps(info, allow_nan=False)
    except (TypeError, ValueError):
        errors.append(label + "配置情報は有限数値を含むJSONである必要があります。")
    digest = info.get("atlas_hash")
    atlas = project.assets.get(digest) if isinstance(digest, str) else None
    if atlas is None:
        errors.append(label + "元パーツシートがプロジェクトにありません。")
    source_size = info.get("source_size")
    destination_size = info.get("destination_size")
    if destination_size != list(project.size):
        errors.append(label + "配置先キャンバスの寸法が一致しません。")
    if _rgba(atlas) and source_size != [atlas.shape[1], atlas.shape[0]]:
        errors.append(label + "元パーツシートの寸法が一致しません。")
    crop = info.get("crop")
    valid_crop = (isinstance(crop, list) and len(crop) == 4
                  and all(type(value) is int for value in crop)
                  and 0 <= crop[0] < crop[2] and 0 <= crop[1] < crop[3])
    if not valid_crop:
        errors.append(label + "元パーツの切り出し範囲が不正です。")
    elif atlas is not None and _rgba(atlas):
        x0, y0, x1, y1 = crop
        if x1 > atlas.shape[1] or y1 > atlas.shape[0]:
            errors.append(label + "切り出し範囲が元パーツシートの外にあります。")
        elif _rgba(part.asset) and not np.array_equal(part.asset, atlas[y0:y1, x0:x1]):
            errors.append(label + "元パーツが記録された切り出し範囲と一致しません。")
    if not _valid_matrix(info.get("matrix")):
        errors.append(label + "配置変換行列が不正です。")
    if part.alignment_edit_mask is not None:
        edited_mask = part.alignment_edit_mask
        if (not isinstance(edited_mask, np.ndarray) or edited_mask.dtype != np.uint8
                or edited_mask.shape != project.source.shape[:2]):
            errors.append(label + "編集基準マスクとキャンバスの寸法・形式が一致しません。")
        elif edited_mask.flags.writeable:
            errors.append(label + "編集基準マスクは読み取り専用である必要があります。")
        if not _valid_matrix(info.get("edit_mask_matrix")):
            errors.append(label + "編集基準マスクの変換行列が不正です。")
    scale, offset = info.get("scale"), info.get("offset")
    if not _finite_number(scale) or scale <= 0:
        errors.append(label + "配置倍率が不正です。")
    if (not isinstance(offset, list) or len(offset) != 2
            or not all(_finite_number(value) for value in offset)):
        errors.append(label + "配置オフセットが不正です。")
    if not isinstance(info.get("anchors"), list):
        errors.append(label + "配置の基準点がありません。")
    return errors


def validate(project: Project, *, for_export: bool = False) -> list[str]:
    errors = []
    source = project.source
    if not isinstance(source, np.ndarray) or source.ndim != 3 or source.shape[2] != 4 or source.dtype != np.uint8:
        return ["原画は8bit RGBA画像である必要があります。"]
    if not source.shape[0] or not source.shape[1]:
        errors.append("キャンバスが空です。")
    if not isinstance(project.assets, dict):
        return errors + ["元パーツシートの一覧が不正です。"]
    for digest, pixels in project.assets.items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            errors.append("元パーツシートのSHA-256が不正です。")
        if not _rgba(pixels):
            errors.append("元パーツシートは8bit RGBA画像である必要があります。")
        elif pixels.flags.writeable:
            errors.append("元パーツシートは読み取り専用である必要があります。")
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
        elif for_export and not np.any(part.mask) and (
            part.generated_mask is None or not np.any(part.generated_mask)
        ):
            errors.append(f"{part.name}: マスクが空です。")
        for label, mask in (("補完領域", part.hidden_mask), ("生成領域", part.generated_mask)):
            if mask is not None and (mask.shape != source.shape[:2] or mask.dtype != np.uint8):
                errors.append(f"{part.name}: {label}とキャンバスが一致しません。")
        if (part.generated is None) != (part.generated_mask is None):
            errors.append(f"{part.name}: 生成画像と生成領域を両方保存する必要があります。")
        if part.generated is not None and (
            part.generated.shape != source.shape or part.generated.dtype != np.uint8
        ):
            errors.append(f"{part.name}: 生成画像とキャンバスが一致しません。")
        errors.extend(_alignment_errors(project, part))
    if for_export and not project.parts:
        errors.append("出力するパーツがありません。")
    return errors


def layer_pixels(project: Project, part: Part) -> np.ndarray:
    result = (part.artwork if part.artwork is not None else project.source).copy()
    result[..., 3] = (
        result[..., 3].astype(np.uint16) * part.mask.astype(np.uint16) // 255
    ).astype(np.uint8)
    if part.generated is not None and part.generated_mask is not None:
        # Visible source artwork always wins, even after later mask editing.
        hidden = (part.generated_mask > 0) & (part.mask == 0)
        result[hidden] = part.generated[hidden]
        result[..., 3][hidden] = (
            part.generated[..., 3][hidden].astype(np.uint16)
            * part.generated_mask[hidden].astype(np.uint16) // 255
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
