import numpy as np
import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from live2d_semi_auto.infrastructure import load_project
from live2d_semi_auto.ui import MainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, project):
    window = MainWindow()
    window.show()
    window.set_project(project)
    app.processEvents()
    yield window
    window.editor.dirty = False
    window.close()


def test_gui_edit_save_load(window, app, tmp_path, monkeypatch):
    window.add_part()
    window.name.setText("face")
    window.kind.setText("face")
    window.apply_properties()
    original = window.editor.project.source.copy()
    window.radius.setValue(1)
    start = window.canvas.mapFromScene(4, 4)
    end = window.canvas.mapFromScene(9, 4)
    QTest.mousePress(window.canvas.viewport(), Qt.LeftButton, pos=start)
    QTest.mouseMove(window.canvas.viewport(), end)
    QTest.mouseRelease(window.canvas.viewport(), Qt.LeftButton, pos=end)
    assert np.any(window.editor.project.parts[0].mask)
    assert np.array_equal(window.editor.project.source, original)
    mask = window.editor.project.parts[0].mask.copy()
    window.history(False)
    assert not np.any(window.editor.project.parts[0].mask)
    window.history(True)
    assert np.array_equal(window.editor.project.parts[0].mask, mask)
    for mode in range(5):
        window.mode.setCurrentIndex(mode)
        app.processEvents()
        assert not window.canvas.picture.pixmap().isNull()
    target = tmp_path / "ui.l2split"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(target), ""))
    assert window.save()
    loaded = load_project(target)
    assert loaded.parts[0].name == "face"
    assert np.array_equal(loaded.parts[0].mask, mask)
    assert not window.editor.dirty
    window.set_project(loaded, target)
    assert window.parts.item(0).text() == "face"


def test_gui_rejects_close_with_unsaved_changes(window, monkeypatch):
    window.add_part()
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.Cancel)
    assert not window.close()
    assert window.isVisible()


def test_gui_no_paint_in_source_mode(window):
    window.add_part()
    window.mode.setCurrentIndex(0)
    pos = window.canvas.mapFromScene(5, 5)
    QTest.mouseClick(window.canvas.viewport(), Qt.LeftButton, pos=pos)
    assert not np.any(window.editor.project.parts[0].mask)


def test_entry_point_runs_event_loop(app):
    from live2d_semi_auto.__main__ import main
    QTimer.singleShot(100, app.quit)
    assert main() == 0
