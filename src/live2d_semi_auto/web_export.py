"""Non-destructive web-character models and a dependency-free browser bundle.

Coordinates are source-canvas pixels, top-left origin. Textures are cropped but
bounds retain their original canvas placement. This format is independent of
Cubism and does not generate or consume proprietary model files.
"""

from copy import deepcopy
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from PIL import Image, ImageCms

from .core import Part, Project, validate


ROLE_CHOICES = ["body", "face", "hair_back", "hair_front", "hair_side", "eye",
                "iris", "eyebrow", "mouth", "accessory_head", "accessory_body",
                "arm_left", "arm_right", "leg", "static"]
EXPRESSION_PRESETS = {
    "neutral": {"eye": 1, "mouth": 0, "smile": 0, "brow": 0, "blush": 0, "tear": 0},
    "smile": {"eye": .7, "mouth": .3, "smile": 1, "brow": .2, "blush": .15, "tear": 0},
    "angry": {"eye": .7, "mouth": .25, "smile": -.8, "brow": -1, "blush": .15, "tear": 0},
    "cry": {"eye": .5, "mouth": .65, "smile": -1, "brow": .8, "blush": .1, "tear": 1},
    "surprised": {"eye": 1.15, "mouth": .9, "smile": 0, "brow": 1, "blush": 0, "tear": 0},
    "shy": {"eye": .75, "mouth": .15, "smile": .5, "brow": .4, "blush": 1, "tear": 0},
}


def _number(value, label, minimum, maximum):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{label} は有限の数値で指定してください。")
    return min(maximum, max(minimum, float(value)))


def _point(value, label, size):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} は [x, y] で指定してください。")
    return [_number(value[0], label, 0, size[0]), _number(value[1], label, 0, size[1])]


def get_runtime_config(project: Project) -> dict:
    for entry in reversed(project.history):
        if entry.get("operation") == "web-runtime-settings":
            config = deepcopy(entry.get("config", {}))
            # A desktop/core edit can remove parts without changing this adapter's
            # stored history. Ignore stale bindings when reading; retain history
            # so Undo restores both the part and its original animation settings.
            if isinstance(config, dict) and isinstance(config.get("bindings"), dict):
                current_ids = {part.id for part in project.parts}
                config["bindings"] = {key: binding for key, binding in config["bindings"].items()
                                      if key in current_ids}
            return config
    return {}


def normalize_runtime_config(project: Project, config: dict | None = None) -> dict:
    """Reject malformed references; clamp finite motion settings to safe ranges."""
    config = get_runtime_config(project) if config is None else config
    if not isinstance(config, dict):
        raise ValueError("Webモデル設定はオブジェクトで指定してください。")
    allowed = {"bindings", "headPivot", "bodyPivot", "meshResolution", "strength"}
    if set(config) - allowed:
        raise ValueError("未対応のWebモデル設定があります。")
    bindings = config.get("bindings", {})
    if not isinstance(bindings, dict):
        raise ValueError("パーツの動作割り当てが不正です。")
    ids = {part.id for part in project.parts}
    cleaned = {}
    for part_id, binding in bindings.items():
        if part_id not in ids:
            raise ValueError("動作割り当てが存在しないパーツを参照しています。")
        if not isinstance(binding, dict) or set(binding) - {"role", "pivot"}:
            raise ValueError("パーツの動作割り当てが不正です。")
        if binding.get("role") not in ROLE_CHOICES:
            raise ValueError("パーツの動作種類が不正です。")
        cleaned[part_id] = {"role": binding["role"]}
        if "pivot" in binding:
            cleaned[part_id]["pivot"] = _point(binding["pivot"], "パーツ支点", project.size)
    resolution = config.get("meshResolution", 4)
    if type(resolution) is not int:
        raise ValueError("メッシュ分割数は整数で指定してください。")
    result = {"bindings": cleaned, "meshResolution": min(8, max(2, resolution)),
              "strength": _number(config.get("strength", 1), "動作の強さ", 0, 2)}
    for key in ("headPivot", "bodyPivot"):
        if key in config:
            result[key] = _point(config[key], key, project.size)
    return result


def _role(part: Part) -> str:
    label = f"{part.kind} {part.name}".casefold().replace("-", "_")
    # Classify accessories first: "hair_clip" contains the substring "lip".
    if any(word in label for word in ("hairpin", "headband", "head_band", "hair_clip", "髪飾", "カチューシャ")):
        return "accessory_head"
    if any(word in label for word in ("eyebrow", "brow", "眉")):
        return "eyebrow"
    if any(word in label for word in ("mouth", "lips", "口")) or "lip" in label.replace("_", " ").split():
        return "mouth"
    if any(word in label for word in ("iris", "pupil", "瞳")):
        return "iris"
    if any(word in label for word in ("eye", "目")):
        return "eye"
    if any(word in label for word in ("hair", "髪")):
        if any(word in label for word in ("back", "rear", "後")):
            return "hair_back"
        if any(word in label for word in ("side", "strand", "横", "毛束")):
            return "hair_side"
        return "hair_front"
    if any(word in label for word in ("face", "head", "顔")):
        return "face"
    if any(word in label for word in ("arm", "hand", "腕", "手")):
        return "arm_right" if any(word in label for word in ("right", "_r", "右")) else "arm_left"
    if any(word in label for word in ("leg", "shoe", "足", "脚", "靴")):
        return "leg"
    if any(word in label for word in ("bow", "ribbon", "accessory", "リボン", "アクセ")):
        return "accessory_body"
    if any(word in label for word in ("body", "torso", "clothes", "costume", "服", "胴")):
        return "body"
    return "static"


def _bounds(project: Project, part: Part) -> list[int] | None:
    base = part.artwork if part.artwork is not None else project.source
    visible = ((base[..., 3].astype(np.uint16) * part.mask.astype(np.uint16)) // 255) > 0
    if part.generated is not None and part.generated_mask is not None:
        visible |= ((part.mask == 0) &
                    ((part.generated[..., 3].astype(np.uint16) * part.generated_mask.astype(np.uint16)) // 255 > 0))
    yy, xx = np.nonzero(visible)
    if not len(xx):
        return None
    return [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]


def layer_sprite(project: Project, part: Part) -> np.ndarray:
    """Only allocate cropped pixels. Original and accepted artwork stay untouched."""
    bounds = _bounds(project, part)
    if bounds is None:
        return np.zeros((1, 1, 4), dtype=np.uint8)
    x0, y0, x1, y1 = bounds
    crop = np.s_[y0:y1, x0:x1]
    base = part.artwork if part.artwork is not None else project.source
    pixels = base[crop].copy()
    mask = part.mask[crop]
    pixels[..., 3] = (pixels[..., 3].astype(np.uint16) * mask.astype(np.uint16) // 255).astype(np.uint8)
    if part.generated is not None:
        generated_mask = part.generated_mask[crop]
        hidden = (mask == 0) & (generated_mask > 0)
        generated = part.generated[crop]
        pixels[hidden] = generated[hidden]
        pixels[..., 3][hidden] = (generated[..., 3][hidden].astype(np.uint16)
                                 * generated_mask[hidden].astype(np.uint16) // 255).astype(np.uint8)
    if project.icc_profile:
        original = ImageCms.ImageCmsProfile(io.BytesIO(project.icc_profile))
        srgb = ImageCms.createProfile("sRGB")
        image = ImageCms.profileToProfile(Image.fromarray(pixels), original, srgb, outputMode="RGBA")
        pixels = np.asarray(image).copy()
    return pixels


def runtime_model(project: Project, config: dict | None = None) -> dict:
    errors = validate(project)
    if errors:
        raise ValueError("\n".join(errors))
    config = normalize_runtime_config(project, config)
    width, height = project.size
    layers = []
    omitted = []
    for index, part in enumerate(project.parts):
        bounds = _bounds(project, part)
        if bounds is None:
            omitted.append(part.id)
            continue
        binding = config["bindings"].get(part.id, {})
        role = binding.get("role", _role(part))
        filename = sha256(part.id.encode("utf-8")).hexdigest()[:32]
        layer = {"id": part.id, "name": part.name, "kind": part.kind, "role": role,
                 "visible": bool(part.visible), "z_order": index,
                 "asset": f"parts/{filename}.png", "bounds": bounds}
        if "pivot" in binding:
            layer["pivot"] = binding["pivot"]
        layers.append(layer)
    faces = [layer for layer in layers if layer["role"] == "face"]
    if faces:
        face = faces[0]["bounds"]
        head_pivot = [(face[0] + face[2]) / 2, face[1] + (face[3] - face[1]) * .94]
    else:
        head_pivot = [width * .5, height * .38]
    head_pivot = config.get("headPivot", head_pivot)
    body_pivot = config.get("bodyPivot", [width * .5, height * .82])
    return {"version": 1, "format": "live2d-semi-auto-web-character", "canvas": [width, height],
            "colorSpace": "sRGB", "layers": layers, "omittedEmptyLayers": omitted,
            "roleChoices": list(ROLE_CHOICES), "motions": ["idle", "nod", "shake", "greeting"],
            "groups": {"head": {"pivot": head_pivot}, "body": {"pivot": body_pivot}},
            "config": config, "expressions": deepcopy(EXPRESSION_PRESETS),
            "parameters": {"angleX": [-1, 1], "angleY": [-1, 1], "angleZ": [-1, 1],
                           "eyeOpen": [0, 1], "mouthOpen": [0, 1], "breath": [0, 1]},
            "provenance": {"sourceName": project.source_name, "sourceSha256": project.source_hash,
                           "sourceArtworkPreserved": True,
                           "animation": "editable procedural mesh rig; no Cubism dependency",
                           "expressionOverlays": "procedural blush and tears"}}


def _bundle_html(model):
    # These escapes prevent names/source metadata from ending a script element.
    encoded = json.dumps(model, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Web Character</title><style>body{font:16px system-ui;margin:0;background:#eef2f8;color:#203044}main{max-width:900px;margin:auto;padding:20px}canvas{display:block;width:100%;height:65vh;background:linear-gradient(135deg,#fff,#e8edf7);border-radius:18px;touch-action:none}fieldset{border:0;padding:10px 0;display:flex;gap:12px;flex-wrap:wrap}label{display:flex;align-items:center;gap:6px}button,select{font:inherit;padding:6px}#status{min-height:1.5em}</style><main>
<canvas id="character" width="900" height="750" aria-label="Animated character"></canvas>
<fieldset><label>表情 <select id="expression"><option value="neutral">通常</option><option value="smile">笑顔</option><option value="angry">怒り</option><option value="cry">泣き</option><option value="surprised">驚き</option><option value="shy">照れ</option></select></label><label>モーション <select id="motion"><option value="idle">待機</option><option value="nod">うなずく</option><option value="shake">首を振る</option><option value="greeting">挨拶</option></select></label><label><input id="blink" type="checkbox" checked>自動瞬き</label><label><input id="idle" type="checkbox" checked>呼吸・揺れ</label><label><input id="track" type="checkbox" checked>ポインター追従</label></fieldset>
<fieldset><label>左右 <input id="angleX" type="range" min="-1" max="1" value="0" step=".01"></label><label>上下 <input id="angleY" type="range" min="-1" max="1" value="0" step=".01"></label><label>傾き <input id="angleZ" type="range" min="-1" max="1" value="0" step=".01"></label><label>口 <input id="mouthOpen" type="range" min="0" max="1" value="0" step=".01"></label><button id="reset">リセット</button></fieldset><p id="status">画像を読み込み中…</p></main>
<script src="runtime.js"></script><script type="application/json" id="model">''' + encoded + '''</script><script>
const model=JSON.parse(document.getElementById('model').textContent);
const canvas=document.getElementById('character');
const character=new Live2DWeb.Character(canvas,model);
window.character=character;
function resizeCanvas(){
  const bounds=canvas.getBoundingClientRect();
  if(bounds.width<=0||bounds.height<=0)return;
  const ratio=Math.min(window.devicePixelRatio||1,8192/bounds.width,8192/bounds.height);
  character.resize(bounds.width*ratio,bounds.height*ratio);
  if(!character.running)character.render(0);
}
const resizeObserver=window.ResizeObserver?new ResizeObserver(resizeCanvas):null;
if(resizeObserver)resizeObserver.observe(canvas);
window.addEventListener('resize',resizeCanvas);
resizeCanvas();
character.load().then(()=>{resizeCanvas();character.start();document.getElementById('status').textContent='ポインターを動かすと顔が追従します。';}).catch(error=>document.getElementById('status').textContent=error.message);
document.getElementById('expression').onchange=event=>character.setExpression(event.target.value);
document.getElementById('motion').onchange=event=>character.setMotion(event.target.value);
for(const [id,method] of [['blink','setAutoBlink'],['idle','setIdle'],['track','setPointerTracking']])document.getElementById(id).onchange=event=>character[method](event.target.checked);
for(const id of ['angleX','angleY','angleZ','mouthOpen'])document.getElementById(id).oninput=event=>character.setParameters({[id]:Number(event.target.value)});
document.getElementById('reset').onclick=()=>{character.setParameters({angleX:0,angleY:0,angleZ:0,eyeOpen:1,mouthOpen:0,breath:0});character.setExpression('neutral');character.setMotion('idle');document.getElementById('expression').value='neutral';document.getElementById('motion').value='idle';for(const id of ['angleX','angleY','angleZ','mouthOpen'])document.getElementById(id).value='0';};
</script></html>'''


def export_web_bundle(project: Project, path: str | Path, config: dict | None = None) -> None:
    path = Path(path)
    if path.suffix.lower() != ".zip":
        raise ValueError("Webモデルは .zip で保存してください。")
    if path.exists():
        raise ValueError("既存ファイルを守るため、新しいWebモデルファイル名を指定してください。")
    model = runtime_model(project, config)
    if not model["layers"] or not any(layer["visible"] for layer in model["layers"]):
        raise ValueError("Webモデルに表示できるパーツがありません。")
    parts = {part.id: part for part in project.parts}
    runtime = Path(__file__).with_name("web_static").joinpath("runtime.js").read_bytes()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".zip", delete=False) as file:
            temporary = Path(file.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("model.json", json.dumps(model, ensure_ascii=False, indent=2, allow_nan=False))
            archive.writestr("runtime.js", runtime)
            archive.writestr("index.html", _bundle_html(model))
            archive.writestr("README.md", """# Web character bundle

Extract all files together and open index.html, or serve the directory with any static server.
No API keys, backend, CDN, Cubism SDK, or network services are needed for animation.
Original artwork is represented by independent cropped sRGB RGBA textures.
Motion uses an editable procedural triangle-mesh rig. Blush/tears are drawn overlays.

Embedding on your site (serve model/assets from the same origin, or configure CORS):

```html
<canvas id="avatar" width="600" height="800"></canvas>
<script src="character/runtime.js"></script>
<script>
fetch('character/model.json').then(r => r.json()).then(async model => {
  const avatar = new Live2DWeb.Character(document.querySelector('#avatar'), model,
    {baseUrl: 'character/'});
  await avatar.load();
  const resize = () => {
    const bounds = avatar.canvas.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const ratio = Math.min(window.devicePixelRatio || 1,
      8192 / bounds.width, 8192 / bounds.height);
    avatar.resize(bounds.width * ratio, bounds.height * ratio);
    if (!avatar.running) avatar.render(0);
  };
  const observer = window.ResizeObserver ? new ResizeObserver(resize) : null;
  observer?.observe(avatar.canvas);
  window.addEventListener('resize', resize); // Also supports browsers without ResizeObserver.
  resize();
  avatar.start();
  avatar.setExpression('smile');
  avatar.setParameters({mouthOpen: 0.5});
  window.avatar = avatar;
  // On component unmount: observer?.disconnect();
  // window.removeEventListener('resize', resize); avatar.destroy();
});
</script>
```

Character methods: load(), start(), stop(), destroy(), render(timestampMilliseconds),
setParameters(values), getParameters(), setExpression(name), setMotion(name), setAutoBlink(boolean),
setIdle(boolean), setPointerTracking(boolean), resize(width, height).
resize uses backing pixels; match CSS dimensions times devicePixelRatio to preserve
the character's proportions. Parameter ranges are in model.json.
Greeting waves an arm when assigned arm_right; otherwise it moves the head gently.
Names/IDs are metadata; filenames are hashes of IDs. Artwork rights remain with their owner.
""")
            for layer in model["layers"]:
                image = Image.fromarray(layer_sprite(project, parts[layer["id"]]))
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                archive.writestr(layer["asset"], buffer.getvalue())
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
