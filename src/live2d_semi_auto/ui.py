"""Desktop view; all edits go through application services."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGraphicsPathItem,
    QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox, QPushButton,
    QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from .application import Editor
from .core import composite, layer_pixels, validate
from .inference import AlphaBackend
from .infrastructure import export_png, import_image, load_project, save_project


def pixmap(pixels: np.ndarray) -> QPixmap:
    pixels = np.ascontiguousarray(pixels)
    height, width = pixels.shape[:2]
    image = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format_RGBA8888)
    return QPixmap.fromImage(image.copy())


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
                                 self.window.radius.value(), self.window.tool.currentIndex() == 1)
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
                                           self.window.tool.currentIndex() == 3)
                self.outline.setPath(QPainterPath())
            self.previous, self.points = None, []
            self.window.refresh_canvas()
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.editor: Editor | None = None
        self.project_path: Path | None = None
        self.resize(1200, 800)
        self.setWindowTitle("Live2D Semi-Auto")
        toolbar = self.addToolBar("ファイル・編集")
        for title, shortcut, handler in [
            ("画像を読み込む", "Ctrl+N", self.open_image),
            ("プロジェクトを開く", "Ctrl+O", self.open_project),
            ("保存", "Ctrl+S", self.save),
            ("名前を付けて保存", "Ctrl+Shift+S", lambda: self.save(True)),
            ("PNG出力", "Ctrl+E", self.export),
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
        self.radius = QSpinBox()
        self.radius.setRange(1, 200)
        self.radius.setValue(12)
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(45)
        self.opacity.valueChanged.connect(self.refresh_canvas)
        controls = QHBoxLayout()
        for widget in (self.mode, self.tool, QLabel("半径(px)"), self.radius,
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
            export_png(editor.project, Path(parent) / name)
            self.statusBar().showMessage("透過PNG・manifest・プレビューを出力しました。")

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
                alpha = project.parts[index].mask[..., None].astype(np.float32) / 255
                alpha *= self.opacity.value() / 100
                pixels[..., :3] = np.rint(pixels[..., :3] * (1 - alpha)
                                          + np.array([40, 220, 170]) * alpha).astype(np.uint8)
                pixels[..., 3] = np.maximum(pixels[..., 3], (alpha[..., 0] * 255).astype(np.uint8))
        self.canvas.show_pixels(pixels)
        label = self.project_path.name if self.project_path else project.source_name
        self.setWindowTitle(f"{'* ' if self.editor.dirty else ''}{label} — Live2D Semi-Auto")

    def fit(self):
        if self.editor:
            self.canvas.fitInView(self.canvas.picture.boundingRect(), Qt.KeepAspectRatio)

    def closeEvent(self, event):
        event.accept() if self.discard() else event.ignore()
