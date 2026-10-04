"""Desktop view; all edits go through application services."""

import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGraphicsPathItem,
    QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox, QPushButton,
    QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget,
    QProgressDialog,
)

from .application import Editor
from .core import composite, layer_pixels, validate
from .exporters import PsdExporter
from .gpt_parts import GPTPartsBackend
from .inference import AlphaBackend, ColorPartsBackend
from .inpainting import LaMaBackend, TeleaBackend
from .infrastructure import export_png, import_image, load_project, save_project


def pixmap(pixels: np.ndarray) -> QPixmap:
    pixels = np.ascontiguousarray(pixels)
    height, width = pixels.shape[:2]
    image = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format_RGBA8888)
    return QPixmap.fromImage(image.copy())


class Worker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, operation, parent):
        super().__init__(parent)
        self.operation = operation

    def run(self):
        try:
            result = self.operation()
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.completed.emit(result)


class Canvas(QGraphicsView):
    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.window = window
        self.setScene(QGraphicsScene(self))
        self.picture = QGraphicsPixmapItem()
        self.scene().addItem(self.picture)
        self.outline = QGraphicsPathItem()
        self.outline.setPen(QPen(QColor("#80e5ff"), 0))
        self.scene().addItem(self.outline)
        self.setBackgroundBrush(QColor("#555555"))
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.previous = None
        self.pan_position = None
        self.points = []

    def show_pixels(self, pixels: np.ndarray):
        self.picture.setPixmap(pixmap(pixels))
        self.scene().setSceneRect(self.picture.boundingRect())

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        zoom = self.transform().m11() * factor
        if 0.05 <= zoom <= 32:
            self.scale(factor, factor)
        event.accept()

    def point(self, event):
        pos = self.mapToScene(event.position().toPoint())
        return round(pos.x()), round(pos.y())

    def editable(self):
        return (self.window.editor is not None and self.window.index() >= 0
                and self.window.mode.currentIndex() == 1)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.pan_position = event.position().toPoint()
            return
        if event.button() == Qt.LeftButton and self.editable():
            point = self.point(event)
            width, height = self.window.editor.project.size
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                return
            if self.window.tool.currentIndex() >= 2:
                self.points = [point]
            else:
                self.window.editor.checkpoint()
                self.previous = point
                self.paint_to(point)
            return
        super().mousePressEvent(event)

    def paint_to(self, point):
        self.window.editor.paint(self.window.index(), self.previous, point,
                                 self.window.radius.value(), self.window.tool.currentIndex() == 1,
                                 hidden=self.window.target.currentIndex() == 1)
        self.previous = point
        self.window.refresh_canvas()

    def mouseMoveEvent(self, event):
        if self.pan_position is not None:
            position = event.position().toPoint()
            delta = position - self.pan_position
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.pan_position = position
            return
        if self.previous is not None and self.editable():
            self.paint_to(self.point(event))
            return
        if self.points and self.editable():
            self.points.append(self.point(event))
            path = QPainterPath()
            path.moveTo(*self.points[0])
            for point in self.points[1:]:
                path.lineTo(*point)
            path.closeSubpath()
            self.outline.setPath(path)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.pan_position = None
        if event.button() == Qt.LeftButton:
            if self.previous is not None and self.editable():
                self.paint_to(self.point(event))
            if self.points and self.editable():
                self.points.append(self.point(event))
                self.window.editor.polygon(self.window.index(), self.points,
                                           self.window.tool.currentIndex() == 3,
                                           hidden=self.window.target.currentIndex() == 1)
                self.outline.setPath(QPainterPath())
            self.previous, self.points = None, []
            self.window.refresh_canvas()
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.editor: Editor | None = None
        self.project_path: Path | None = None
        self.worker = None
        self.job_cancelled = False
        self.resize(1200, 800)
        self.setWindowTitle("Live2D Semi-Auto")
        toolbar = self.toolbar = self.addToolBar("ファイル・編集")
        for title, shortcut, handler in [
            ("画像を読み込む", "Ctrl+N", self.open_image),
            ("プロジェクトを開く", "Ctrl+O", self.open_project),
            ("保存", "Ctrl+S", self.save),
            ("名前を付けて保存", "Ctrl+Shift+S", lambda: self.save(True)),
            ("PNG出力", "Ctrl+E", self.export),
            ("PSD出力", "Ctrl+Shift+E", self.export_psd),
            ("Undo", "Ctrl+Z", lambda: self.history(False)),
            ("Redo", "Ctrl+Shift+Z", lambda: self.history(True)),
            ("全体表示", "Ctrl+0", self.fit),
        ]:
            action = QAction(title, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(lambda checked=False, fn=handler: self.run(fn))
            toolbar.addAction(action)

        self.parts = QListWidget()
        self.parts.currentRowChanged.connect(self.select)
        self.name, self.kind = QLineEdit(), QLineEdit()
        self.visible = QCheckBox("再合成で表示")
        form = QFormLayout()
        form.addRow("名前", self.name)
        form.addRow("種類", self.kind)
        form.addRow(self.visible)
        sidebar = QVBoxLayout()
        sidebar.addWidget(QLabel("パーツ（下の行ほど手前）"))
        sidebar.addWidget(self.parts)
        sidebar.addLayout(form)
        for title, fn in [
            ("属性を適用", self.apply_properties), ("パーツを追加", self.add_part),
            ("パーツを削除", self.delete_part),
            ("奥へ移動 ↑", lambda: self.move_part(-1)),
            ("手前へ移動 ↓", lambda: self.move_part(1)),
            ("不透明領域のマスク候補", self.propose),
            ("自動パーツ分割（色領域）", self.auto_parts),
            ("GPTでパーツ候補（外部送信）", self.gpt_parts),
            ("隠れ領域を補完", self.repair),
        ]:
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, handler=fn: self.run(handler))
            sidebar.addWidget(button)
        self.solo = QCheckBox("再合成で選択パーツのみ表示")
        self.solo.toggled.connect(self.refresh_canvas)
        sidebar.addWidget(self.solo)
        side = QWidget()
        side.setLayout(sidebar)
        side.setMaximumWidth(320)

        self.mode = QComboBox()
        self.mode.addItems(["原画", "マスク編集・オーバーレイ", "選択パーツ", "再合成", "差分"])
        self.mode.setCurrentIndex(1)
        self.mode.currentIndexChanged.connect(self.refresh_canvas)
        self.tool = QComboBox()
        self.tool.addItems(["ブラシ追加", "ブラシ消去", "なげなわ追加", "なげなわ消去"])
        self.target = QComboBox()
        self.target.addItems(["可視マスク", "補完領域"])
        self.target.currentIndexChanged.connect(self.refresh_canvas)
        self.radius = QSpinBox()
        self.radius.setRange(1, 200)
        self.radius.setValue(12)
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(45)
        self.opacity.valueChanged.connect(self.refresh_canvas)
        controls = QHBoxLayout()
        for widget in (self.mode, self.target, self.tool, QLabel("半径(px)"), self.radius,
                       QLabel("マスク濃度"), self.opacity):
            controls.addWidget(widget)
        self.canvas = Canvas(self)
        body = QVBoxLayout()
        body.addLayout(controls)
        body.addWidget(self.canvas)
        body.addWidget(QLabel("マスク編集モードで左ドラッグ：描画／中ボタンドラッグ：移動／ホイール：拡大縮小"))
        center = QWidget()
        center.setLayout(body)
        splitter = QSplitter()
        splitter.addWidget(side)
        splitter.addWidget(center)
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("画像を読み込んで、パーツを追加してください。")

    def run(self, operation):
        try:
            return operation()
        except (ValueError, OSError, RuntimeError) as exc:
            QMessageBox.warning(self, "処理を完了できませんでした",
                                f"{exc}\n\n編集内容は保持されています。入力や保存先を確認して再試行してください。")
            return False

    def index(self):
        return self.parts.currentRow()

    def require(self):
        if self.editor is None:
            raise ValueError("先に画像またはプロジェクトを開いてください。")
        return self.editor

    def require_part(self):
        editor = self.require()
        if self.index() < 0:
            raise ValueError("編集するパーツを選択してください。")
        return editor

    def discard(self):
        if self.editor is None or not self.editor.dirty:
            return True
        answer = QMessageBox.question(self, "未保存の変更",
                                      "変更を保存しますか？",
                                      QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                      QMessageBox.Save)
        if answer == QMessageBox.Save:
            return bool(self.run(self.save))
        return answer == QMessageBox.Discard

    def set_project(self, project, path=None):
        self.editor = Editor(project)
        self.project_path = Path(path) if path else None
        self.refresh()
        self.fit()

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "原画を読み込む", "", "画像 (*.png *.jpg *.jpeg *.webp)")
        if path:
            project = import_image(path)
            if self.discard():
                self.set_project(project)
                self.editor.dirty = True
                self.refresh_canvas()

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "プロジェクトを開く", "", "プロジェクト (*.l2split)")
        if path:
            project = load_project(path)
            if self.discard():
                self.set_project(project, path)

    def save(self, save_as=False):
        editor = self.require()
        path = None if save_as else self.project_path
        if path is None:
            chosen, _ = QFileDialog.getSaveFileName(self, "保存", "project.l2split", "プロジェクト (*.l2split)")
            if not chosen:
                return False
            path = Path(chosen)
            if not path.suffix:
                path = path.with_suffix(".l2split")
        save_project(editor.project, path)
        self.project_path = path
        editor.dirty = False
        self.refresh_canvas()
        self.statusBar().showMessage("プロジェクトを保存しました。")
        return True

    def export(self):
        editor = self.require()
        errors = validate(editor.project, for_export=True)
        if errors:
            raise ValueError("\n".join(errors))
        parent = QFileDialog.getExistingDirectory(self, "出力先の親フォルダーを選択")
        if not parent:
            return
        name, ok = QInputDialog.getText(self, "PNG出力", "新しい出力フォルダー名", text="parts_export")
        if ok:
            if not name.strip() or name in {".", ".."} or "/" in name or "\\" in name:
                raise ValueError("フォルダー名だけを入力してください。")
            snapshot = editor.project.snapshot()
            self.background("PNG出力中…", lambda: export_png(snapshot, Path(parent) / name),
                            lambda _: self.statusBar().showMessage("PNG出力が完了しました。"), cancellable=False)

    def export_psd(self):
        editor = self.require()
        chosen, _ = QFileDialog.getSaveFileName(self, "新しいPSDファイルへ出力", "parts.psd", "PSD (*.psd)")
        if chosen:
            path = Path(chosen)
            if not path.suffix:
                path = path.with_suffix(".psd")
            snapshot = editor.project.snapshot()
            self.background("PSD出力中…", lambda: PsdExporter().export(snapshot, path),
                            lambda _: self.statusBar().showMessage("PSD出力が完了しました。Cubismでの確認は別途必要です。"),
                            cancellable=False)

    def background(self, title, operation, on_result, *, cancellable=True):
        if self.worker is not None:
            raise ValueError("実行中の処理が完了してから再試行してください。")
        self.job_cancelled = False
        self.centralWidget().setEnabled(False)
        self.toolbar.setEnabled(False)
        progress = QProgressDialog(title, "結果を破棄" if cancellable else "", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        if not cancellable:
            progress.setCancelButton(None)
            progress.setWindowFlag(Qt.WindowCloseButtonHint, False)
        progress.canceled.connect(lambda: setattr(self, "job_cancelled", True))
        worker = self.worker = Worker(operation, self)

        def receive(result):
            progress.hide()
            if self.job_cancelled:
                self.statusBar().showMessage("結果を破棄しました。編集中のデータは変更していません。")
            else:
                self.run(lambda: on_result(result))

        def failed(message):
            progress.hide()
            if not self.job_cancelled:
                QMessageBox.warning(self, "処理を完了できませんでした", f"{message}\n編集内容は保持されています。")

        def finished():
            self.centralWidget().setEnabled(True)
            self.toolbar.setEnabled(True)
            progress.deleteLater()
            worker.deleteLater()
            self.worker = None

        worker.completed.connect(receive)
        worker.failed.connect(failed)
        worker.finished.connect(finished)
        worker.start()

    def confirm_pixels(self, title, text, pixels):
        dialog = QMessageBox(self)
        dialog.setWindowTitle(title)
        dialog.setText(text)
        dialog.setIconPixmap(pixmap(pixels).scaled(480, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        dialog.setDefaultButton(QMessageBox.No)
        return dialog.exec() == QMessageBox.Yes

    def auto_parts(self):
        editor = self.require()
        count, ok = QInputDialog.getInt(self, "自動パーツ分割", "色領域の候補数（意味的な髪・顔分類ではありません）", 8, 2, 16)
        if not ok:
            return
        source = editor.project.source

        def show(proposal):
            preview = source.copy()
            for i, part in enumerate(proposal.parts):
                color = np.array([(i * 79 + 50) % 256, (i * 131 + 90) % 256, (i * 193 + 130) % 256])
                preview[part.mask > 0, :3] = (preview[part.mask > 0, :3] * 0.35 + color * 0.65).astype(np.uint8)
            if self.confirm_pixels("自動分割候補", f"{len(proposal.parts)}個の色領域候補を追加しますか？\n既存の手動パーツは保持します。採用後に修正・Undoできます。", preview):
                editor.accept_parts(proposal)
                self.refresh()

        self.background("色領域の候補を計算中…", lambda: ColorPartsBackend(count).propose_parts(source), show)

    def repair(self):
        editor = self.require_part()
        index = self.index()
        mask = editor.repair_mask(index)
        selected, ok = QInputDialog.getItem(self, "補完方法", "ローカル補完バックエンド",
                                           ["LaMa（AI・モデルが必要）", "Telea（画像処理・AIではありません）"], 0, False)
        if not ok:
            return
        if selected.startswith("LaMa"):
            path = os.environ.get("LAMA_MODEL_PATH")
            if not path:
                path, _ = QFileDialog.getOpenFileName(self, "取得済みのLaMaモデルを選択", "", "TorchScript (*.pt)")
            if not path:
                return
            backend = LaMaBackend(path)
        else:
            backend = TeleaBackend()
        source = editor.project.source

        def show(proposal):
            temporary = editor.project.snapshot()
            temporary.parts[index].generated = proposal.pixels
            temporary.parts[index].generated_mask = proposal.mask
            before = layer_pixels(editor.project, editor.project.parts[index])
            after = layer_pixels(temporary, temporary.parts[index])
            comparison = np.concatenate((before, after), axis=1)
            if self.confirm_pixels("補完候補：左=現在／右=候補", "生成結果を採用しますか？\n原画の可視領域は保持され、Undoできます。", comparison):
                editor.accept_repair(index, proposal)
                self.mode.setCurrentIndex(2)
                self.refresh_canvas()

        self.background("隠れ領域を補完中…", lambda: backend.propose(source, mask), show)

    def gpt_parts(self):
        editor = self.require()
        backend = GPTPartsBackend()
        if not backend.api_key:
            raise ValueError("環境設定で GPT_API_KEY を設定してください。キーをチャットに貼らないでください。")
        answer = QMessageBox.question(
            self, "OpenAIへの画像送信", f"原画を最大1024pxに縮小して api.openai.com に送信します。\n"
            f"モデル: {backend.model}\n画像からパーツの分類・位置候補を得ます。APIの利用料金が発生します。送信しますか？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        source = editor.project.source

        def show(proposal):
            preview = source.copy()
            for i, part in enumerate(proposal.parts):
                tint = np.array([(i * 79 + 50) % 256, (i * 131 + 90) % 256, (i * 193 + 130) % 256])
                preview[part.mask > 0, :3] = (preview[part.mask > 0, :3] * 0.4 + tint * 0.6).astype(np.uint8)
            names = "、".join(p.name for p in proposal.parts)
            if self.confirm_pixels("GPT候補の確認", f"{names}\n分類・位置をGPTで提案し、GrabCutでマスク化した候補です。\n輪郭・名称・順序は要確認です。追加しますか？", preview):
                editor.accept_parts(proposal)
                self.refresh()

        self.background("GPTの分類・位置候補を取得中…", lambda: backend.propose_parts(source), show)

    def add_part(self):
        self.require().add_part()
        self.refresh(len(self.editor.project.parts) - 1)

    def delete_part(self):
        editor = self.require_part()
        index = self.index()
        editor.delete(index)
        self.refresh(max(0, index - 1))

    def apply_properties(self):
        editor = self.require_part()
        index = self.index()
        editor.update_part(index, self.name.text(), self.kind.text(), self.visible.isChecked())
        self.refresh(index)

    def move_part(self, offset):
        editor = self.require_part()
        self.refresh(editor.move(self.index(), offset))

    def history(self, redo):
        editor = self.require()
        index = self.index()
        editor.redo() if redo else editor.undo()
        self.refresh(index)

    def propose(self):
        editor = self.require_part()
        candidate = editor.proposal(AlphaBackend())
        # A proposal is inspectable before it can replace the selected mask.
        temporary = editor.project.snapshot()
        temporary.parts[self.index()].mask = candidate
        dialog = QMessageBox(self)
        dialog.setWindowTitle("マスク候補の確認")
        dialog.setText("非透明画素を選ぶ簡易候補です。キャラクターの意味的な分割は行いません。\n選択パーツのマスクを置き換えますか？（Undo可能）")
        dialog.setIconPixmap(pixmap(layer_pixels(temporary, temporary.parts[self.index()])).scaled(
            320, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        dialog.setDefaultButton(QMessageBox.No)
        if dialog.exec() == QMessageBox.Yes:
            editor.accept_mask(self.index(), candidate)
            self.refresh_canvas()

    def refresh(self, index=0):
        self.parts.blockSignals(True)
        self.parts.clear()
        if self.editor:
            self.parts.addItems([p.name for p in self.editor.project.parts])
            self.parts.setCurrentRow(min(max(0, index), self.parts.count() - 1))
        self.parts.blockSignals(False)
        self.select(self.index())

    def select(self, index):
        enabled = self.editor is not None and index >= 0
        for widget in (self.name, self.kind, self.visible):
            widget.setEnabled(enabled)
        if enabled:
            part = self.editor.project.parts[index]
            self.name.setText(part.name)
            self.kind.setText(part.kind)
            self.visible.setChecked(part.visible)
        else:
            self.name.clear()
            self.kind.clear()
        self.refresh_canvas()

    def refresh_canvas(self, *_):
        if self.editor is None:
            return
        project = self.editor.project
        mode, index = self.mode.currentIndex(), self.index()
        if mode in {3, 4}:
            if self.solo.isChecked() and index >= 0:
                pixels = layer_pixels(project, project.parts[index])
            else:
                pixels = composite(project)
            if mode == 4:
                pixels = np.abs(project.source.astype(np.int16) - pixels.astype(np.int16)).astype(np.uint8)
                pixels[..., 3] = 255
        elif mode == 2:
            pixels = (layer_pixels(project, project.parts[index]) if index >= 0
                      else np.zeros_like(project.source))
        else:
            pixels = project.source.copy()
            if mode == 1 and index >= 0:
                part = project.parts[index]
                mask = part.hidden_mask if self.target.currentIndex() == 1 else part.mask
                if mask is None:
                    mask = np.zeros_like(part.mask)
                alpha = mask[..., None].astype(np.float32) / 255
                alpha *= self.opacity.value() / 100
                tint = [255, 140, 40] if self.target.currentIndex() == 1 else [40, 220, 170]
                pixels[..., :3] = np.rint(pixels[..., :3] * (1 - alpha)
                                          + np.array(tint) * alpha).astype(np.uint8)
                pixels[..., 3] = np.maximum(pixels[..., 3], (alpha[..., 0] * 255).astype(np.uint8))
        self.canvas.show_pixels(pixels)
        label = self.project_path.name if self.project_path else project.source_name
        self.setWindowTitle(f"{'* ' if self.editor.dirty else ''}{label} — Live2D Semi-Auto")

    def fit(self):
        if self.editor:
            self.canvas.fitInView(self.canvas.picture.boundingRect(), Qt.KeepAspectRatio)

    def closeEvent(self, event):
        if self.worker is not None:
            event.ignore()
            self.statusBar().showMessage("処理の完了または結果の破棄後に閉じてください。")
            return
        event.accept() if self.discard() else event.ignore()
