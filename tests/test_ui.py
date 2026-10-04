import numpy as np
import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QInputDialog, QDialog, QDoubleSpinBox

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


def wait_for_worker(window):
    for _ in range(200):
        QTest.qWait(10)
        if window.worker is None:
            return
    raise AssertionError("background job did not finish")


def test_background_success_and_cancel_keep_ui_responsive(window):
    import threading
    result = []
    ready = threading.Event()
    window.background("testing", lambda: ready.wait(2) or 42, result.append)
    assert not window.centralWidget().isEnabled()
    window.job_cancelled = True
    ready.set()
    wait_for_worker(window)
    assert result == []
    assert window.centralWidget().isEnabled()
    window.background("testing", lambda: 42, result.append)
    wait_for_worker(window)
    assert result == [42]


def test_gui_hidden_mask_painting(window):
    window.add_part()
    window.target.setCurrentIndex(1)
    window.radius.setValue(1)
    pos = window.canvas.mapFromScene(5, 5)
    QTest.mouseClick(window.canvas.viewport(), Qt.LeftButton, pos=pos)
    assert not np.any(window.editor.project.parts[0].mask)
    assert window.editor.project.parts[0].hidden_mask[5, 5] == 255


def test_gpt_image_upload_requires_explicit_confirmation(window, monkeypatch):
    class Backend:
        api_key = "test-token"
        model = "test-model"
        def propose_parts(self, source):
            pytest.fail("declined artwork must not be sent")
    monkeypatch.setattr("live2d_semi_auto.ui.GPTPartsBackend", Backend)
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.No)
    window.gpt_parts()
    assert window.worker is None


def atlas_upload_setup(window, tmp_path, monkeypatch):
    import json
    from PIL import Image
    from live2d_semi_auto.alignment import GPTAlignmentBackend
    atlas_path = tmp_path / "atlas.png"
    Image.fromarray(window.editor.project.source).save(atlas_path)
    calls = []
    response = {"choices": [{"message": {"content": json.dumps({"parts": [{
        "name": "face", "kind": "face", "atlas_bbox": [100, 100, 900, 900],
        "anchors": [{"source": [200, 200], "target": [200, 200]},
                    {"source": [800, 200], "target": [800, 200]}],
        "confidence": 0.9, "notes": "test proposal",
    }]})}}]}

    class Backend(GPTAlignmentBackend):
        def __init__(self):
            super().__init__(api_key="test-token", transport=lambda _: response)
        def propose_alignment(self, *args, **kwargs):
            calls.append(True)
            return super().propose_alignment(*args, **kwargs)

    monkeypatch.setattr("live2d_semi_auto.ui.GPTAlignmentBackend", Backend)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (str(atlas_path), ""))
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a: ("透明背景", True))
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **kw: ("test-model", True))
    return calls


def test_atlas_upload_decline_never_calls_model(window, tmp_path, monkeypatch):
    calls = atlas_upload_setup(window, tmp_path, monkeypatch)
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.No)
    window.align_parts()
    assert calls == [] and window.worker is None
    assert window.editor.project.parts == []


def test_atlas_proposal_rejection_keeps_manual_work(window, tmp_path, monkeypatch):
    calls = atlas_upload_setup(window, tmp_path, monkeypatch)
    window.add_part()
    before_id = window.editor.project.parts[0].id
    history_length = len(window.editor.undo_stack)
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.Yes)
    monkeypatch.setattr(window, "confirm_alignment", lambda *a: False)
    window.align_parts()
    wait_for_worker(window)
    assert calls == [True]
    assert [p.id for p in window.editor.project.parts] == [before_id]
    assert window.editor.project.assets == {}
    assert len(window.editor.undo_stack) == history_length


def test_atlas_acceptance_matches_preview_with_existing_parts(window, tmp_path, monkeypatch):
    from live2d_semi_auto.core import composite
    calls = atlas_upload_setup(window, tmp_path, monkeypatch)
    window.add_part()
    window.editor.project.parts[0].mask[:, :5] = 255
    existing_id = window.editor.project.parts[0].id
    inspected = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.Yes)
    monkeypatch.setattr(window, "confirm_alignment", lambda *a: inspected.append(a[-1].copy()) or True)
    window.align_parts()
    wait_for_worker(window)
    assert calls == [True]
    assert len(window.editor.project.parts) == 2
    assert window.editor.project.parts[0].id == existing_id
    assert np.array_equal(composite(window.editor.project), inspected[0])
    assert window.editor.project.parts[1].artwork is not None
    window.history(False)
    assert [p.id for p in window.editor.project.parts] == [existing_id]
    assert window.editor.project.assets == {}
    window.history(True)
    assert len(window.editor.project.parts) == 2


def test_gui_alignment_adjustment_is_reviewable_and_undoable(window, tmp_path, monkeypatch):
    from live2d_semi_auto.core import composite
    atlas_upload_setup(window, tmp_path, monkeypatch)
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.Yes)
    monkeypatch.setattr(window, "confirm_alignment", lambda *a: True)
    window.align_parts()
    wait_for_worker(window)
    before = window.editor.project.parts[0].artwork.copy()
    part_id = window.editor.project.parts[0].id
    inspected = []

    def edit_dialog(dialog):
        dialog.findChildren(QDoubleSpinBox)[2].setValue(2)
        return QDialog.Accepted

    monkeypatch.setattr(QDialog, "exec", edit_dialog)
    monkeypatch.setattr(window, "confirm_alignment", lambda *a: inspected.append(a[-1].copy()) or True)
    window.adjust_alignment()
    wait_for_worker(window)
    assert not np.array_equal(window.editor.project.parts[0].artwork, before)
    assert window.editor.project.parts[0].id == part_id
    assert np.array_equal(composite(window.editor.project), inspected[0])
    window.history(False)
    assert np.array_equal(window.editor.project.parts[0].artwork, before)
