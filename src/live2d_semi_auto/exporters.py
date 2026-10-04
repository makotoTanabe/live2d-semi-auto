"""Exporter boundary: core state has no PSD-library dependency."""

import io
import os
from pathlib import Path
import tempfile
from typing import Protocol

from PIL import Image, ImageCms
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer
from psd_tools.constants import Resource, Tag
from psd_tools.psd.image_resources import ImageResource

from .core import Project, layer_pixels, validate


class Exporter(Protocol):
    def export(self, project: Project, path: str | Path) -> None: ...


class PsdExporter:
    def export(self, project: Project, path: str | Path) -> None:
        errors = validate(project, for_export=True)
        if errors:
            raise ValueError("\n".join(errors))
        path = Path(path)
        if path.suffix.lower() != ".psd":
            raise ValueError("PSDは .psd で保存してください。")
        if path.exists():
            raise ValueError("既存ファイルを守るため、新しいPSDファイル名を指定してください。")
        if max(project.size) > 30000:
            raise ValueError("PSDのキャンバス上限は30000pxです。")
        srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        source_profile = ImageCms.ImageCmsProfile(io.BytesIO(project.icc_profile)) if project.icc_profile else None
        # RGBA is represented by RGB PSD color mode plus transparency channels.
        psd = PSDImage.new("RGBA", project.size, depth=8)
        psd.background_color = None
        psd.image_resources[Resource.ICC_PROFILE] = ImageResource(
            key=Resource.ICC_PROFILE, data=srgb.tobytes())
        for index, part in enumerate(project.parts):
            image = Image.fromarray(layer_pixels(project, part))
            if source_profile:
                image = ImageCms.profileToProfile(image, source_profile, srgb, outputMode="RGBA")
            # PSD legacy Pascal names use MacRoman; the Unicode tag is canonical.
            layer = PixelLayer.frompil(image, psd, name=f"Layer_{index + 1:03d}")
            layer.tagged_blocks.set_data(Tag.UNICODE_LAYER_NAME, part.name)
            layer.visible = part.visible
        psd.tagged_blocks.set_data(Tag.SAVING_MERGED_TRANSPARENCY)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".psd", delete=False) as file:
                temporary = Path(file.name)
            psd.save(temporary)
            with temporary.open("rb") as file:
                os.fsync(file.fileno())
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
