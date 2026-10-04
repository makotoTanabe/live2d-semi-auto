"""Editing transactions and reversible manual/inference operations."""

from collections import deque
import cv2
import numpy as np

from .core import Part, Project, validate
from .inference import PartsProposal, SegmentationBackend
from .inpainting import RepairProposal


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
              radius: int, erase: bool = False, hidden: bool = False) -> None:
        """Caller checkpoints once per stroke, not once per mouse event."""
        if radius < 1:
            raise ValueError("ブラシ半径は1以上にしてください。")
        mask = self.edit_mask(index, hidden)
        value = 0 if erase else 255
        cv2.line(mask, start, end, value, thickness=2 * radius + 1)
        cv2.circle(mask, start, radius, value, thickness=-1)
        cv2.circle(mask, end, radius, value, thickness=-1)

    def polygon(self, index: int, points: list[tuple[int, int]], erase: bool = False,
                hidden: bool = False) -> None:
        if len(points) >= 3:
            self.checkpoint()
            cv2.fillPoly(self.edit_mask(index, hidden),
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

    def edit_mask(self, index: int, hidden: bool = False) -> np.ndarray:
        part = self.project.parts[index]
        if not hidden:
            return part.mask
        if part.hidden_mask is None:
            part.hidden_mask = np.zeros_like(part.mask)
        return part.hidden_mask

    def accept_parts(self, proposal: PartsProposal) -> None:
        # Add proposals; never discard existing manual parts.
        candidate = self.project.snapshot()
        names = {p.name.casefold() for p in candidate.parts}
        for part in proposal.parts:
            name = part.name
            suffix = 2
            while name.casefold() in names:
                name = f"{part.name}_{suffix}"
                suffix += 1
            names.add(name.casefold())
            candidate.parts.append(Part(name, part.mask.copy(), part.kind))
        errors = validate(candidate)
        if errors or not proposal.parts:
            raise ValueError("\n".join(errors) or "分割候補がありません。")
        self.checkpoint()
        candidate.history.append({"operation": "auto-parts", **proposal.metadata})
        self.project = candidate

    def repair_mask(self, index: int) -> np.ndarray:
        part = self.project.parts[index]
        mask = part.hidden_mask
        if mask is None or not np.any(mask):
            raise ValueError("補完領域モードで、隠れた領域を描いてください。")
        if np.any((mask > 0) & (part.mask > 0)):
            raise ValueError("補完領域が原画の可視マスクと重なっています。可視領域の外側だけを指定してください。")
        return mask.copy()

    def accept_repair(self, index: int, proposal: RepairProposal) -> None:
        requested = self.repair_mask(index)
        if (proposal.pixels.shape != self.project.source.shape or proposal.pixels.dtype != np.uint8
                or proposal.mask.shape != requested.shape or proposal.mask.dtype != np.uint8
                or not np.array_equal(proposal.mask, requested)):
            raise ValueError("補完候補の寸法・領域が不正です。")
        self.checkpoint()
        part = self.project.parts[index]
        # Retain previous repairs outside the newly selected target.
        generated = part.generated.copy() if part.generated is not None else np.zeros_like(self.project.source)
        coverage = part.generated_mask.copy() if part.generated_mask is not None else np.zeros_like(part.mask)
        generated[requested > 0] = proposal.pixels[requested > 0]
        coverage[requested > 0] = requested[requested > 0]
        generated.flags.writeable = coverage.flags.writeable = False
        part.generated, part.generated_mask = generated, coverage
        self.project.history.append({"operation": "inpaint", "part_id": part.id, **proposal.metadata})
