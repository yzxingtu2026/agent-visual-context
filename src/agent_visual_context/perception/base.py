"""感知层抽象：目标检测、目标跟踪与关系推理均可替换。

具体模型（YOLO-World、RelateAnything 等）只能在 `perception.adapters` 之类的适配器中实现，
领域层与流水线仅依赖本模块协议。
"""

from __future__ import annotations

from typing import Protocol

from ..domain import Detection, Frame, Relation, TrackedObject


class Detector(Protocol):
    """单帧目标检测器。"""

    @property
    def model_version(self) -> str: ...

    def detect(self, frame: Frame) -> list[Detection]: ...


class Tracker(Protocol):
    """跨帧目标跟踪器，为检测框分配稳定 track id。"""

    @property
    def model_version(self) -> str: ...

    def update(self, frame: Frame, detections: list[Detection]) -> list[TrackedObject]: ...

    def reset(self) -> None: ...


class RelationReasoner(Protocol):
    """开放词汇关系推理器。"""

    @property
    def model_version(self) -> str: ...

    def infer(self, frame: Frame, objects: list[TrackedObject]) -> list[Relation]: ...
