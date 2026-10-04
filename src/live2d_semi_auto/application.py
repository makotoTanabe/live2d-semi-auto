"""Editing transactions and reversible manual/inference operations."""

from collections import deque
import cv2
import numpy as np

from .core import Part, Project
from .inference import SegmentationBackend


class Editor:
    def __init__(self, project: Project):
        self.project = project
        self.undo_stack: deque[Project] = deque(maxlen=50)
        self.redo_stack: deque[Project] = deque(maxlen=50)
        self.dirty = False

    def checkpoint(self) -> None:
        self.undo_stack.append(self.project.snapshot())
        self.redo_stack.clear()
        self.dirty = True

    def undo(self) -> None:
        if self.undo_stack:
            self.redo_stack.append(self.project.snapshot())
            self.project = self.undo_stack.pop()
            self.dirty = True

    def redo(self) -> None:
        if self.redo_stack:
            self.undo_stack.append(self.project.snapshot())
            self.project = self.redo_stack.pop()
            self.dirty = True

    def add_part(self) -> Part:
        self.checkpoint()
        names = {p.name for p in self.project.parts}
        index = 1
        while f"part_{index:03d}" in names:
            index += 1
        part = Part(f"part_{index:03d}", np.zeros(self.project.source.shape[:2], dtype=np.uint8))
        self.project.parts.append(part)
        return part

    def update_part(self, index: int, name: str, kind: str, visible: bool) -> None:
        name, kind = name.strip(), kind.strip()
        if not name or not kind:
            raise ValueError("パーツ名と種類を入力してください。")
        if any(i != index and p.name.casefold() == name.casefold()
               for i, p in enumerate(self.project.parts)):
            raise ValueError("同じ名前のパーツがすでにあります。")
        self.checkpoint()
        part = self.project.parts[index]
        part.name, part.kind, part.visible = name, kind, visible

    def move(self, index: int, offset: int) -> int:
        target = index + offset
        if 0 <= target < len(self.project.parts):
            self.checkpoint()
            self.project.parts.insert(target, self.project.parts.pop(index))
            return target
        return index

    def delete(self, index: int) -> None:
        self.checkpoint()
        self.project.parts.pop(index)

    def paint(self, index: int, start: tuple[int, int], end: tuple[int, int],
              radius: int, erase: bool = False) -> None:
        """Caller checkpoints once per stroke, not once per mouse event."""
        if radius < 1:
            raise ValueError("ブラシ半径は1以上にしてください。")
        mask = self.project.parts[index].mask
        value = 0 if erase else 255
        cv2.line(mask, start, end, value, thickness=2 * radius + 1)
        cv2.circle(mask, start, radius, value, thickness=-1)
        cv2.circle(mask, end, radius, value, thickness=-1)

    def polygon(self, index: int, points: list[tuple[int, int]], erase: bool = False) -> None:
        if len(points) >= 3:
            self.checkpoint()
            cv2.fillPoly(self.project.parts[index].mask,
                         [np.asarray(points, dtype=np.int32)], 0 if erase else 255)

    def proposal(self, backend: SegmentationBackend) -> np.ndarray:
        mask = backend.propose(self.project.source)
        if mask.shape != self.project.source.shape[:2] or mask.dtype != np.uint8:
            raise ValueError("推論候補の寸法・形式が不正です。")
        return mask.copy()

    def accept_mask(self, index: int, mask: np.ndarray) -> None:
        if mask.shape != self.project.source.shape[:2] or mask.dtype != np.uint8:
            raise ValueError("マスクの寸法・形式が不正です。")
        self.checkpoint()
        self.project.parts[index].mask = mask.copy()
