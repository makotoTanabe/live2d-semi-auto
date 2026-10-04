"""Local HTTP adapter for editing and playing an embeddable 2D character.

Artwork remains in a bounded, expiring in-memory session. Remote proposals are
explicit, reviewable jobs; no UI or web dependency enters the domain modules.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import tempfile
import threading
import time
from urllib.parse import urlsplit
import zipfile

import numpy as np
from PIL import Image
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt
from starlette.background import BackgroundTask
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .alignment import GPTAlignmentBackend
from .application import Editor
from .core import Project, composite, layer_pixels, mask_bounds
from .exporters import PsdExporter
from .gpt_parts import GPTPartsBackend, GPTRequestError
from .inference import ColorPartsBackend, PartsProposal
from .inpainting import LaMaBackend, RepairProposal, TeleaBackend
from .infrastructure import export_png, import_image, load_project, require_matching_profiles, save_project


MAX_UPLOAD = 32 * 1024 * 1024
MAX_DIMENSION = 4096
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_PIXELS = 64 * 1024 * 1024
SESSION_TTL = 60 * 60
MAX_SESSIONS = 16


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PartUpdate(Input):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    kind: str | None = Field(default=None, min_length=1, max_length=255)
    visible: StrictBool | None = None


class Move(Input):
    offset: StrictInt = Field(ge=-128, le=128)


class Stroke(Input):
    points: list[tuple[StrictInt, StrictInt]] = Field(min_length=1, max_length=4096)
    radius: StrictInt = Field(default=8, ge=1, le=256)
    erase: StrictBool = False
    hidden: StrictBool = False
    tool: str = "brush"


class History(Input):
    direction: str


class Sample(Input):
    name: str = "alignment"


class ColorRequest(Input):
    count: StrictInt = Field(default=8, ge=2, le=16)


class GPTRequest(Input):
    model: str | None = Field(default=None, min_length=1, max_length=100)
    api_key: str | None = Field(default=None, min_length=1, max_length=512, repr=False)
    consent: StrictBool = False


class RepairRequest(Input):
    part_id: str
    backend: str = "telea"


class Adjustment(Input):
    scale: float = Field(default=1, ge=0.01, le=100)
    angle: float = Field(default=0, ge=-36000, le=36000)
    offset: tuple[float, float] = (0, 0)


@dataclass
class Job:
    id: str
    kind: str
    revision: int
    status: str = "running"
    message: str = "候補を作成しています。"
    result: object = None
    preview: Project | None = None
    part_id: str | None = None
    future: object = None


@dataclass
class Session:
    editor: Editor | None = None
    revision: int = 0
    accessed: float = field(default_factory=time.monotonic)
    lock: threading.RLock = field(default_factory=threading.RLock)
    jobs: dict[str, Job] = field(default_factory=dict)
    active: str | None = None


def _safe(value):
    """Return JSON metadata without credentials, server paths or image payloads."""
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()
                if not any(word in str(k).lower() for word in ("api_key", "authorization", "secret", "original_path"))}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, str):
        if "data:image/" in value or re.search(r"sk-[A-Za-z0-9_-]+", value):
            return "[非公開]"
        return value[:4096]
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


def _parts(project: Project) -> list[dict]:
    return [{"id": part.id, "name": part.name, "kind": part.kind,
             "visible": part.visible, "bounds": mask_bounds(part.mask),
             "hidden_bounds": mask_bounds(part.hidden_mask) if part.hidden_mask is not None else None,
             "generated": part.generated is not None, "aligned": part.alignment is not None,
             "alignment": _safe(part.alignment)} for part in project.parts]


def _project_budget(project: Project, *, extra_pixels: int = 0, extra_parts: int = 0) -> None:
    """Apply the same resource budget to editing and archive loading.

    A downloadable project must remain reloadable. Count all stored arrays,
    even when their pixel buffers are shared internally by immutable snapshots.
    """
    if len(project.parts) + extra_parts > 128:
        raise ValueError("パーツ数の上限は128です。不要なパーツを削除してから再試行してください。")
    pixels = extra_pixels + project.source.shape[0] * project.source.shape[1]
    pixels += sum(asset.shape[0] * asset.shape[1] for asset in project.assets.values())
    for part in project.parts:
        for field_name in ("mask", "hidden_mask", "generated", "generated_mask", "artwork",
                           "asset", "asset_mask", "alignment_edit_mask"):
            image = getattr(part, field_name)
            if image is not None:
                pixels += image.shape[0] * image.shape[1]
    if pixels > MAX_ARCHIVE_PIXELS:
        raise ValueError("元画像・パーツ・マスクの合計がWeb版の総画素数上限を超えます。"
                         "編集内容は保持されています。候補数を減らすか、元ファイルを残して小さい画像を読み込んでください。")


def _project(session: Session) -> Editor:
    if session.editor is None:
        raise HTTPException(409, "先に画像またはプロジェクトを読み込んでください。")
    return session.editor


def _mutable(session: Session) -> Editor:
    editor = _project(session)
    if session.active and session.jobs[session.active].status in {"running", "ready"}:
        raise HTTPException(409, "処理候補を採用または破棄してから編集してください。")
    return editor


def _index(editor: Editor, part_id: str) -> int:
    for index, part in enumerate(editor.project.parts):
        if part.id == part_id:
            return index
    raise HTTPException(404, "指定されたパーツはありません。")


def _png(pixels: np.ndarray, profile=None) -> Response:
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, "PNG", icc_profile=profile)
    return Response(stream.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


def _image_check(raw: bytes) -> None:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("PNG・JPEG・WebP画像を選択してください。")
            if max(image.size) > MAX_DIMENSION or min(image.size) < 1:
                raise ValueError("画像は縦横4096px以内にしてください。")
            image.verify()
    except (OSError, Image.DecompressionBombError, SyntaxError) as exc:
        raise ValueError("画像を読み込めません。PNG・JPEG・WebPを確認してください。") from exc


def _archive_check(raw: bytes) -> None:
    """Inspect ZIP sizes and PNG headers before domain archive decoding."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(members) > 512 or len(names) != len(set(names)) or "manifest.json" not in names:
                raise ValueError("プロジェクト内のファイル数または構成が不正です。")
            total_bytes = 0
            for member in members:
                path = PurePosixPath(member.filename.replace("\\", "/"))
                if (path.is_absolute() or ".." in path.parts or ":" in member.filename
                        or member.flag_bits & 1):
                    raise ValueError("プロジェクト内のファイル名または暗号化形式が不正です。")
                total_bytes += member.file_size
                if (member.file_size > MAX_UPLOAD or total_bytes > MAX_ARCHIVE_BYTES
                        or member.file_size > max(1, member.compress_size) * 1000):
                    raise ValueError("展開後のプロジェクトが大きすぎます。")
                if member.filename == "manifest.json" and member.file_size > 1024 * 1024:
                    raise ValueError("プロジェクトのメタデータが大きすぎます。")
            manifest = json.loads(archive.read("manifest.json"))
            parts, assets = manifest.get("parts"), manifest.get("assets", {})
            if not isinstance(parts, list) or len(parts) > 128 or not isinstance(assets, dict):
                raise ValueError("パーツ・元パーツシートの一覧が不正です。")
            references = [manifest.get("source"), *assets.values()]
            image_fields = ("mask", "hidden_mask", "generated", "generated_mask", "artwork",
                            "asset", "asset_mask", "alignment_edit_mask")
            for part in parts:
                if not isinstance(part, dict) or "mask" not in part:
                    raise ValueError("パーツの画像参照が不正です。")
                references.extend(part[key] for key in image_fields if key in part)
            total_pixels = 0
            dimensions = {}
            for reference in references:
                if not isinstance(reference, str) or reference not in names:
                    raise ValueError("プロジェクトに必要な画像がありません。")
                if reference not in dimensions:
                    with archive.open(reference) as stream, Image.open(stream) as image:
                        if max(image.size) > MAX_DIMENSION or min(image.size) < 1:
                            raise ValueError("プロジェクト画像は縦横4096px以内にしてください。")
                        dimensions[reference] = image.width * image.height
                # Count repeated references too: load_project creates each decoded array.
                total_pixels += dimensions[reference]
                if total_pixels > MAX_ARCHIVE_PIXELS:
                    raise ValueError("プロジェクト内の画像・マスクの総画素数が大きすぎます。")
    except (zipfile.BadZipFile, OSError, Image.DecompressionBombError, json.JSONDecodeError, AttributeError) as exc:
        raise ValueError("プロジェクトが破損しているか、対応するZIP形式ではありません。") from exc


async def _upload(file: UploadFile) -> tuple[bytes, str]:
    try:
        raw = await file.read(MAX_UPLOAD + 1)
    finally:
        await file.close()
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "アップロードは32MiB以内にしてください。")
    if not raw:
        raise HTTPException(422, "ファイルが空です。")
    name = (file.filename or "image.png").replace("\\", "/").rsplit("/", 1)[-1][:255]
    return raw, name or "image.png"


def _import(raw: bytes, name: str, *, archive: bool = False) -> Project:
    (_archive_check if archive else _image_check)(raw)
    with tempfile.TemporaryDirectory(prefix="l2web-import-") as directory:
        path = Path(directory) / ("project.l2split" if archive else "source.png")
        path.write_bytes(raw)
        project = load_project(path) if archive else import_image(path)
    project.original_path = None
    if not archive:
        project.source_name = name
        project.source_hash = sha256(raw).hexdigest()
    return project


def create_app(*, allowed_hosts: list[str] | None = None, session_ttl: float = SESSION_TTL,
               max_sessions: int = MAX_SESSIONS, sample_root: str | Path | None = None) -> FastAPI:
    """Create one isolated adapter; tests can explicitly permit their test host."""
    sessions: dict[str, Session] = {}
    registry_lock = threading.RLock()
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="l2web")
    job_slots = threading.BoundedSemaphore(4)

    @asynccontextmanager
    async def lifespan(app):
        yield
        for session in sessions.values():
            with session.lock:
                for job in session.jobs.values():
                    if job.status in {"running", "ready"}:
                        job.status = "cancelled"
                        job.result = job.preview = None
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Live2D Semi-Auto Web", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.sessions = sessions
    app.state.executor = executor
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts or ["localhost", "127.0.0.1", "[::1]"])

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        if request.url.path.startswith("/api/") and request.method in {"POST", "PATCH"}:
            # Bound the entire body before multipart parsing can spool unbounded
            # chunked uploads. Starlette's cached request replays this body once.
            size, chunks = 0, []
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD + 1024 * 1024:
                    return JSONResponse({"detail": "リクエストは33MiB以内にしてください。"}, status_code=413)
                chunks.append(chunk)
            request._body = b"".join(chunks)
        origin = request.headers.get("origin")
        if request.url.path.startswith("/api/") and origin:
            parsed = urlsplit(origin)
            expected = urlsplit(str(request.base_url))
            try:
                matches = (parsed.scheme in {"http", "https"} and parsed.scheme == expected.scheme
                           and parsed.hostname == expected.hostname and parsed.port == expected.port
                           and not parsed.username and not parsed.password)
            except ValueError:
                matches = False
            if not matches:
                return JSONResponse({"detail": "別のWebサイトからのAPI操作は許可されていません。"}, status_code=403)
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        message = str(exc) if isinstance(exc, (ValueError, GPTRequestError)) else "操作に失敗しました。"
        return JSONResponse({"detail": _safe(message)}, status_code=422)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # FastAPI's default includes rejected input; it may contain a pasted API key.
        errors = [{"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
                  for error in exc.errors()]
        return JSONResponse({"detail": errors}, status_code=422)

    def clean_sessions():
        expired = [token for token, session in sessions.items() if time.monotonic() - session.accessed > session_ttl]
        for token in expired:
            session = sessions.pop(token)
            with session.lock:
                for job in session.jobs.values():
                    if job.status in {"running", "ready"}:
                        job.status = "cancelled"
                        job.result = job.preview = None
                        if job.future:
                            job.future.cancel()

    def current(request: Request) -> Session:
        token = request.headers.get("x-session-token", "")
        with registry_lock:
            clean_sessions()
            session = sessions.get(token)
            if session is None:
                raise HTTPException(401, "編集セッションがありません。再接続してプロジェクトを読み込んでください。")
            session.accessed = time.monotonic()
            return session

    def state(session: Session):
        from .web_export import normalize_runtime_config
        editor = session.editor
        project = editor.project if editor else None
        active = session.jobs.get(session.active)
        return {"has_project": project is not None, "revision": session.revision,
                "source_name": project.source_name if project else None,
                "source_hash": project.source_hash if project else None,
                "size": list(project.size) if project else None, "parts": _parts(project) if project else [],
                "can_undo": bool(editor and editor.undo_stack), "can_redo": bool(editor and editor.redo_stack),
                "capabilities": {"model": os.environ.get("GPT_MODEL", "gpt-4.1"),
                                 "gpt_configured": bool(os.environ.get("GPT_API_KEY") or os.environ.get("OPENAI_API_KEY")),
                                 "lama_available": Path(os.environ.get("LAMA_MODEL_PATH", "")).is_file()},
                "active_job": {"id": active.id, "kind": active.kind, "status": active.status} if active else None,
                "runtime_config": normalize_runtime_config(project) if project else None}

    def changed(session: Session):
        session.revision += 1
        return state(session)

    def replace_project(session: Session, project: Project):
        from .web_export import normalize_runtime_config
        if session.active:
            raise HTTPException(409, "処理候補を破棄してから別の画像を読み込んでください。")
        normalize_runtime_config(project)
        _project_budget(project)
        session.editor = Editor(project)
        return changed(session)

    def get_job(session: Session, job_id: str) -> Job:
        job = session.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "処理候補がありません。")
        return job

    def job_state(job: Job):
        metadata = getattr(job.result, "metadata", {})
        return {"id": job.id, "kind": job.kind, "status": job.status, "message": job.message,
                "parts": _parts(job.preview) if job.preview else [], "metadata": _safe(metadata),
                "preview_url": f"/api/jobs/{job.id}/preview.png" if job.status == "ready" else None}

    def submit(session: Session, kind: str, operation, *, part_id: str | None = None):
        _mutable(session)
        snapshot = session.editor.project.snapshot()
        if not job_slots.acquire(blocking=False):
            raise HTTPException(429, "処理待ちの上限に達しました。実行中の処理が終わってから再試行してください。")
        job = Job(secrets.token_urlsafe(18), kind, session.revision, part_id=part_id)
        # Keep at most eight recent results; old candidates cannot retain images.
        while len(session.jobs) >= 8:
            session.jobs.pop(next(iter(session.jobs)))
        session.jobs[job.id] = job
        session.active = job.id

        def run():
            try:
                result = operation(snapshot)
                candidate = Editor(snapshot)
                if isinstance(result, PartsProposal):
                    candidate.accept_parts(result)
                elif isinstance(result, RepairProposal):
                    candidate.accept_repair(_index(candidate, part_id), result)
                elif isinstance(result, Project):
                    candidate = Editor(result)
                else:
                    raise ValueError("推論候補の形式が不正です。")
                _project_budget(candidate.project)
                with session.lock:
                    if job.status != "running" or session.revision != job.revision:
                        job.status = "cancelled"
                        job.message = "編集状態が変わったため候補を破棄しました。"
                    else:
                        job.result, job.preview = result, candidate.project
                        job.status, job.message = "ready", "候補を確認し、採用または破棄してください。"
            except Exception as exc:
                with session.lock:
                    if job.status == "running":
                        job.status = "failed"
                        job.message = _safe(str(exc)) if isinstance(exc, ValueError) else "処理に失敗しました。編集内容は保持されています。"
                        session.active = None
            finally:
                with session.lock:
                    if job.status == "cancelled" and session.active == job.id:
                        session.active = None
        try:
            job.future = executor.submit(run)
            job.future.add_done_callback(lambda future: job_slots.release())
        except Exception:
            job_slots.release()
            session.jobs.pop(job.id, None)
            session.active = None
            raise
        return job_state(job)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/session")
    def new_session():
        with registry_lock:
            clean_sessions()
            if len(sessions) >= max_sessions:
                raise HTTPException(429, "編集セッション数の上限に達しました。既存セッションを再利用してください。")
            token = secrets.token_urlsafe(32)
            sessions[token] = Session()
            return {"session": token, "state": state(sessions[token])}

    @app.get("/api/state")
    def read_state(session: Session = Depends(current)):
        with session.lock:
            return state(session)

    @app.post("/api/import/image")
    async def upload_image(file: UploadFile = File(...), session: Session = Depends(current)):
        raw, name = await _upload(file)
        project = _import(raw, name)
        with session.lock:
            return replace_project(session, project)

    @app.post("/api/import/project")
    async def upload_project(file: UploadFile = File(...), session: Session = Depends(current)):
        raw, name = await _upload(file)
        project = _import(raw, name, archive=True)
        with session.lock:
            return replace_project(session, project)

    @app.post("/api/sample")
    def sample(body: Sample, session: Session = Depends(current)):
        locations = {"original": "original_character.png", "alignment": "results/alignment/project.l2split",
                     "automatic": "results/automatic/project.l2split"}
        if body.name not in locations:
            raise ValueError("サンプルは original・alignment・automatic から選んでください。")
        root = Path(sample_root) if sample_root else Path(__file__).resolve().parents[2] / "samples"
        path = root / locations[body.name]
        if not path.is_file():
            raise HTTPException(404, "このインストールにはサンプルがありません。リポジトリのsamplesから読み込んでください。")
        project = _import(path.read_bytes(), path.name, archive=path.suffix == ".l2split")
        with session.lock:
            return replace_project(session, project)

    @app.get("/api/image/{view}.png")
    def image(view: str, part_id: str | None = None, session: Session = Depends(current)):
        with session.lock:
            editor = _project(session)
            project = editor.project
            if view == "source":
                pixels = project.source
            elif view == "composite":
                pixels = composite(project)
            elif view == "difference":
                pixels = np.abs(project.source.astype(np.int16) - composite(project).astype(np.int16)).astype(np.uint8)
                pixels[..., 3] = 255
            elif view in {"layer", "mask", "hidden"}:
                part = project.parts[_index(editor, part_id)]
                pixels = (layer_pixels(project, part) if view == "layer" else
                          part.mask if view == "mask" else
                          part.hidden_mask if part.hidden_mask is not None else np.zeros_like(part.mask))
            else:
                raise HTTPException(404, "表示モードがありません。")
            return _png(pixels, project.icc_profile if pixels.ndim == 3 else None)

    @app.post("/api/parts")
    def add_part(session: Session = Depends(current)):
        with session.lock:
            editor = _mutable(session)
            _project_budget(editor.project, extra_pixels=editor.project.source.shape[0] * editor.project.source.shape[1],
                            extra_parts=1)
            editor.add_part()
            return changed(session)

    @app.patch("/api/parts/{part_id}")
    def update_part(part_id: str, body: PartUpdate, session: Session = Depends(current)):
        with session.lock:
            editor = _mutable(session)
            index = _index(editor, part_id)
            part = editor.project.parts[index]
            editor.update_part(index, body.name if body.name is not None else part.name,
                               body.kind if body.kind is not None else part.kind,
                               body.visible if body.visible is not None else part.visible)
            return changed(session)

    @app.delete("/api/parts/{part_id}")
    def delete_part(part_id: str, session: Session = Depends(current)):
        from .web_export import get_runtime_config
        with session.lock:
            editor = _mutable(session)
            editor.delete(_index(editor, part_id))
            config = get_runtime_config(editor.project)
            if part_id in config.get("bindings", {}):
                config["bindings"].pop(part_id)
                editor.project.history.append({"operation": "web-runtime-settings", "config": config})
            return changed(session)

    @app.post("/api/parts/{part_id}/move")
    def move_part(part_id: str, body: Move, session: Session = Depends(current)):
        with session.lock:
            editor = _mutable(session)
            editor.move(_index(editor, part_id), body.offset)
            return changed(session)

    @app.post("/api/parts/{part_id}/mask")
    def stroke(part_id: str, body: Stroke, session: Session = Depends(current)):
        with session.lock:
            editor = _mutable(session)
            index = _index(editor, part_id)
            width, height = editor.project.size
            if any(not 0 <= x < width or not 0 <= y < height for x, y in body.points):
                raise ValueError("描画座標はキャンバス内にしてください。")
            if body.tool not in {"brush", "lasso"} or (body.tool == "lasso" and len(body.points) < 3):
                raise ValueError("ブラシまたは3点以上の投げ縄を指定してください。")
            if body.hidden and editor.project.parts[index].hidden_mask is None:
                _project_budget(editor.project, extra_pixels=width * height)
            if body.tool == "lasso":
                editor.polygon(index, body.points, body.erase, body.hidden)
            else:
                editor.checkpoint()
                for start, end in zip(body.points, body.points[1:] or body.points):
                    editor.paint(index, start, end, body.radius, body.erase, body.hidden)
            return changed(session)

    @app.post("/api/history")
    def history(body: History, session: Session = Depends(current)):
        with session.lock:
            editor = _mutable(session)
            if body.direction not in {"undo", "redo"}:
                raise ValueError("undo または redo を指定してください。")
            getattr(editor, body.direction)()
            return changed(session)

    @app.post("/api/proposals/color")
    def color(body: ColorRequest, session: Session = Depends(current)):
        with session.lock:
            backend = ColorPartsBackend(body.count)
            return submit(session, "color", lambda project: backend.propose_parts(project.source))

    @app.post("/api/proposals/gpt")
    def gpt(body: GPTRequest, session: Session = Depends(current)):
        if body.consent is not True:
            raise HTTPException(422, "完成画像をapi.openai.comに送信する同意が必要です。API料金が発生します。")
        backend = GPTPartsBackend(model=body.model, api_key=body.api_key)
        def operation(project):
            try:
                return backend.propose_parts(project.source)
            finally:
                backend.api_key = None
        with session.lock:
            return submit(session, "gpt", operation)

    @app.post("/api/proposals/alignment")
    async def alignment(file: UploadFile = File(...), model: str | None = Form(None),
                        background: str = Form("alpha"), consent: bool = Form(False),
                        api_key: str | None = Form(None), session: Session = Depends(current)):
        if consent is not True:
            await file.close()
            raise HTTPException(422, "完成画像とパーツシートをapi.openai.comに送信する同意が必要です。API料金が発生します。")
        if background not in {"alpha", "white"} or (model is not None and not 1 <= len(model) <= 100):
            await file.close()
            raise ValueError("モデル名または背景モードが不正です。")
        if api_key is not None and not 1 <= len(api_key) <= 512:
            await file.close()
            raise ValueError("APIキーの形式を確認してください。")
        raw, name = await _upload(file)
        atlas = _import(raw, name)
        backend = GPTAlignmentBackend(model=model, api_key=api_key)
        with session.lock:
            project = _mutable(session).project
            require_matching_profiles(project, atlas)
            def operation(snapshot):
                try:
                    return backend.propose_alignment(snapshot.source, atlas.source, atlas_name=name,
                                                     atlas_hash=atlas.source_hash, background_mode=background)
                finally:
                    backend.api_key = None
            return submit(session, "alignment", operation)

    @app.post("/api/proposals/repair")
    def repair(body: RepairRequest, session: Session = Depends(current)):
        with session.lock:
            editor = _mutable(session)
            index = _index(editor, body.part_id)
            mask = editor.repair_mask(index)
            if body.backend == "telea":
                backend = TeleaBackend()
            elif body.backend == "lama":
                backend = LaMaBackend(os.environ.get("LAMA_MODEL_PATH", ""))
            else:
                raise ValueError("補完方式は telea または lama を指定してください。")
            return submit(session, "repair", lambda project: backend.propose(project.source, mask),
                          part_id=body.part_id)

    @app.post("/api/parts/{part_id}/adjustment")
    def adjustment(part_id: str, body: Adjustment, session: Session = Depends(current)):
        with session.lock:
            editor = _mutable(session)
            index = _index(editor, part_id)
            def operation(project):
                candidate = Editor(project)
                candidate.adjust_alignment(index, scale=body.scale, angle=body.angle, offset=body.offset)
                return candidate.project
            return submit(session, "adjustment", operation, part_id=part_id)

    @app.get("/api/jobs/{job_id}")
    def read_job(job_id: str, session: Session = Depends(current)):
        with session.lock:
            return job_state(get_job(session, job_id))

    @app.get("/api/jobs/{job_id}/preview.png")
    def job_preview(job_id: str, session: Session = Depends(current)):
        with session.lock:
            job = get_job(session, job_id)
            if job.status != "ready" or job.preview is None:
                raise HTTPException(409, "候補プレビューはまだ利用できません。")
            return _png(composite(job.preview), job.preview.icc_profile)

    @app.post("/api/jobs/{job_id}/accept")
    def accept(job_id: str, session: Session = Depends(current)):
        with session.lock:
            job = get_job(session, job_id)
            if job.status != "ready" or job.preview is None or session.active != job.id:
                raise HTTPException(409, "この候補は採用できません。")
            if job.revision != session.revision:
                job.status, session.active = "cancelled", None
                job.result = job.preview = None
                raise HTTPException(409, "編集状態が変わりました。候補を再作成してください。")
            editor = _project(session)
            editor.checkpoint()
            editor.project = job.preview
            job.status, job.message = "accepted", "候補を採用しました。Undoで戻せます。"
            job.result = job.preview = None
            session.active = None
            return changed(session)

    @app.post("/api/jobs/{job_id}/reject")
    def reject(job_id: str, session: Session = Depends(current)):
        with session.lock:
            job = get_job(session, job_id)
            if job.status in {"running", "ready"}:
                job.status = "cancelled" if job.status == "running" else "rejected"
                job.message = "候補を破棄しました。実行中のAPI処理・料金は取り消せません。"
                job.result = job.preview = None
                if job.future:
                    job.future.cancel()
            if session.active == job.id:
                session.active = None
            return state(session)

    @app.get("/api/runtime/model")
    def runtime_model(session: Session = Depends(current)):
        from .web_export import runtime_model
        with session.lock:
            return runtime_model(_project(session).project)

    @app.get("/api/runtime/layers/{part_id}.png")
    def sprite(part_id: str, session: Session = Depends(current)):
        from .web_export import layer_sprite
        with session.lock:
            editor = _project(session)
            # layer_sprite already converts original-profile colors to sRGB.
            # Browser PNG defaults to sRGB; tagging it with the original ICC
            # would incorrectly apply that conversion a second time.
            return _png(layer_sprite(editor.project, editor.project.parts[_index(editor, part_id)]))

    @app.patch("/api/runtime/config")
    def runtime_config(body: dict, session: Session = Depends(current)):
        from .web_export import get_runtime_config, normalize_runtime_config
        with session.lock:
            editor = _mutable(session)
            merged = {**get_runtime_config(editor.project), **body}
            config = normalize_runtime_config(editor.project, merged)
            editor.checkpoint()
            editor.project.history.append({"operation": "web-runtime-settings", "config": config})
            return changed(session)

    @app.post("/api/export/{format}")
    def export(format: str, session: Session = Depends(current)):
        from .web_export import export_web_bundle
        with session.lock:
            snapshot = _project(session).project.snapshot()
            _project_budget(snapshot)
        names = {"project": "project.l2split", "png": "parts.zip", "psd": "parts.psd", "web": "web-character.zip"}
        if format not in names:
            raise ValueError("出力形式は project・png・psd・web を指定してください。")
        directory = tempfile.TemporaryDirectory(prefix="l2web-export-")
        path = Path(directory.name) / names[format]
        try:
            if format == "project":
                save_project(snapshot, path)
                if path.stat().st_size > MAX_UPLOAD:
                    raise ValueError("保存プロジェクトがWeb版の32MiB上限を超えます。編集内容は保持されています。"
                                     "元ファイルを残して小さい画像を使うか、PNG・Web素材として出力してください。")
                _archive_check(path.read_bytes())
            elif format == "psd":
                PsdExporter().export(snapshot, path)
            elif format == "web":
                export_web_bundle(snapshot, path)
            else:
                package = Path(directory.name) / "png"
                export_png(snapshot, package)
                with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for asset in sorted(package.rglob("*")):
                        if asset.is_file():
                            archive.write(asset, asset.relative_to(package).as_posix())
        except Exception:
            directory.cleanup()
            raise
        return FileResponse(path, filename=names[format], media_type="application/octet-stream",
                            background=BackgroundTask(directory.cleanup))

    static = Path(__file__).with_name("web_static")
    if static.is_dir():
        app.mount("/", StaticFiles(directory=static, html=True), name="web")
    return app


def main():
    parser = argparse.ArgumentParser(description="ブラウザーで編集・再生できるローカルWebアプリ")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    args = parser.parse_args()
    import uvicorn
    hosts = ["localhost", "127.0.0.1", "[::1]"]
    if args.host not in {"0.0.0.0", "::"}:
        hosts.append(args.host)
    uvicorn.run(create_app(allowed_hosts=hosts), host=args.host, port=args.port, access_log=False)
