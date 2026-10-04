"""Image import, atomic project archives, and PNG packages."""

import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

import numpy as np
from PIL import Image, ImageOps

from .core import Part, Project, composite, layer_pixels, mask_bounds, validate


SCHEMA_VERSION = 1


def import_image(path: str | Path) -> Project:
    path = Path(path)
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        if image.format not in {"PNG", "JPEG", "WEBP"}:
            raise ValueError("PNG・JPEG・WebP画像を選択してください。")
        profile = image.info.get("icc_profile")
        pixels = np.array(ImageOps.exif_transpose(image).convert("RGBA"))
    pixels.flags.writeable = False
    return Project(pixels, path.name, hashlib.sha256(raw).hexdigest(),
                   icc_profile=profile, original_path=str(path.resolve()))


def _png(pixels: np.ndarray, profile: bytes | None = None) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, "PNG", icc_profile=profile)
    return stream.getvalue()


def _check(project: Project, *, for_export: bool = False) -> None:
    errors = validate(project, for_export=for_export)
    if errors:
        raise ValueError("\n".join(errors))


def save_project(project: Project, path: str | Path) -> None:
    """Self-contained zip; replace metadata and assets together atomically."""
    _check(project)
    path = Path(path)
    if path.suffix.lower() != ".l2split":
        raise ValueError("プロジェクトは .l2split で保存してください。")
    if project.original_path and (
        path.resolve() == Path(project.original_path).resolve()
        or (path.exists() and Path(project.original_path).exists()
            and os.path.samefile(path, project.original_path))
    ):
        raise ValueError("元画像への上書きは禁止されています。")
    manifest = {
        "schema_version": SCHEMA_VERSION, "application_version": "0.1.0",
        "canvas": list(project.size), "source_name": project.source_name,
        "source_hash": project.source_hash, "source": "source.png", "parts": [],
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("source.png", _png(project.source, project.icc_profile))
            for order, part in enumerate(project.parts):
                asset = f"masks/{order}.png"
                archive.writestr(asset, _png(part.mask))
                manifest["parts"].append({
                    "id": part.id, "name": part.name, "kind": part.kind,
                    "visible": part.visible, "z_order": order, "mask": asset,
                })
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False))
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_project(path: str | Path) -> Project:
    try:
        with zipfile.ZipFile(path) as archive:
            data = json.loads(archive.read("manifest.json"))
            if type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
                raise ValueError("未対応のプロジェクト形式です。")
            with Image.open(io.BytesIO(archive.read(data["source"]))) as image:
                source = np.array(image.convert("RGBA"))
                profile = image.info.get("icc_profile")
            if data["canvas"] != [source.shape[1], source.shape[0]]:
                raise ValueError("保存されたキャンバスと原画の寸法が一致しません。")
            parts = []
            for order, item in enumerate(data["parts"]):
                if type(item["z_order"]) is not int or item["z_order"] != order:
                    raise ValueError("レイヤー順が不正です。")
                if not all(isinstance(item[key], str) for key in ("id", "name", "kind")):
                    raise ValueError("パーツ情報が不正です。")
                if type(item["visible"]) is not bool:
                    raise ValueError("表示状態が不正です。")
                with Image.open(io.BytesIO(archive.read(item["mask"]))) as image:
                    if image.mode != "L":
                        raise ValueError("マスクは8bitグレースケールである必要があります。")
                    mask = np.array(image)
                parts.append(Part(item["name"], mask, item["kind"], item["id"], item["visible"]))
            if not isinstance(data["source_name"], str) or not isinstance(data["source_hash"], str):
                raise ValueError("原画情報が不正です。")
            source.flags.writeable = False
            project = Project(source, data["source_name"], data["source_hash"], parts, profile)
            _check(project)
            return project
    except (KeyError, TypeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ValueError("プロジェクトが破損しているか、必要な画像・マスクがありません。") from exc


def export_png(project: Project, directory: str | Path) -> None:
    """Write a new package only; never overwrite existing files/directories."""
    _check(project, for_export=True)
    directory = Path(directory)
    if directory.exists():
        raise ValueError("既存の出力を守るため、存在しない新しい出力フォルダーを指定してください。")
    staging = Path(tempfile.mkdtemp(prefix=".l2split-export-", dir=directory.parent))
    try:
        (staging / "parts").mkdir()
        manifest = {"schema_version": SCHEMA_VERSION, "canvas": list(project.size),
                    "source_name": project.source_name, "source_hash": project.source_hash,
                    "parts": []}
        for order, part in enumerate(project.parts):
            safe = re.sub(r"[^\w-]+", "_", part.name, flags=re.UNICODE).strip("_")[:64] or "part"
            relative = f"parts/{order:03d}_{safe}.png"
            (staging / relative).write_bytes(_png(layer_pixels(project, part), project.icc_profile))
            manifest["parts"].append({
                "id": part.id, "name": part.name, "kind": part.kind, "z_order": order,
                "visible": part.visible, "file": relative, "bounds": mask_bounds(part.mask),
            })
        (staging / "preview.png").write_bytes(_png(composite(project), project.icc_profile))
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        # Reserve target so a competing export cannot replace its contents.
        directory.mkdir()
        try:
            for child in staging.iterdir():
                shutil.move(str(child), directory / child.name)
        except Exception:
            shutil.rmtree(directory)
            raise
    finally:
        shutil.rmtree(staging)
