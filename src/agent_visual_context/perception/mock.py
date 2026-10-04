"""感知层的 Mock/脚本实现，用于离线 PoC 与测试注入。

这些实现不做任何真实推理，只按脚本产出结果，用来验证流水线接线、降级与时间线行为。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from ..domain import BBox, Detection, Frame, Relation, TrackedObject, TrackRef


class ScriptedDetector:
    """按 `frame_id` 回放预置检测结果的检测器。"""

    def __init__(
        self,
        script: Mapping[str, Sequence[Detection]] | None = None,
        *,
        default: Sequence[Detection] = (),
        model_version: str = "mock-detector-0.1",
    ) -> None:
        self._script: dict[str, list[Detection]] = {
            key: list(value) for key, value in (script or {}).items()
        }
        self._default = list(default)
        self._model_version = model_version
        self.calls = 0

    @property
    def model_version(self) -> str:
        return self._model_version

    def detect(self, frame: Frame) -> list[Detection]:
        self.calls += 1
        return list(self._script.get(frame.frame_id, self._default))


class MockTracker:
    """按标签顺序分配稳定 track id 的极简跟踪器。

    仅用于骨架与测试：真实跟踪（IoU/外观匹配）由 PoC-3 的适配器实现。
    """

    def __init__(self, *, model_version: str = "mock-tracker-0.1") -> None:
        self._model_version = model_version
        self._first_seen: dict[str, datetime] = {}

    @property
    def model_version(self) -> str:
        return self._model_version

    def update(self, frame: Frame, detections: list[Detection]) -> list[TrackedObject]:
        tracked: list[TrackedObject] = []
        seen_labels: dict[str, int] = {}
        for detection in detections:
            index = seen_labels.get(detection.label, 0)
            seen_labels[detection.label] = index + 1
            track_id = f"{detection.label}-{index + 1:02d}"
            first_seen = self._first_seen.setdefault(track_id, frame.captured_at)
            tracked.append(
                TrackedObject(
                    track_id=track_id,
                    label=detection.label,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    first_seen=first_seen,
                    last_seen=frame.captured_at,
                    model_version=self._model_version,
                )
            )
        return tracked

    def reset(self) -> None:
        self._first_seen.clear()


@dataclass(frozen=True, slots=True)
class MockTarget:
    """静态场景中的一个目标。"""

    label: str
    bbox: BBox
    confidence: float = 0.9


class StaticSceneDetector:
    """每帧输出相同目标集合的检测器替身，时间戳取自当前帧。"""

    def __init__(
        self,
        targets: Sequence[MockTarget] = (),
        *,
        model_version: str = "mock-detector-0.1",
    ) -> None:
        self._targets = list(targets)
        self._model_version = model_version
        self.calls = 0

    @property
    def model_version(self) -> str:
        return self._model_version

    def detect(self, frame: Frame) -> list[Detection]:
        self.calls += 1
        return [
            Detection(
                label=target.label,
                bbox=target.bbox,
                confidence=target.confidence,
                model_version=self._model_version,
                detected_at=frame.captured_at,
            )
            for target in self._targets
        ]


@dataclass(frozen=True, slots=True)
class RelationRule:
    """Mock 关系规则：按标签组合生成关系。"""

    subject_label: str
    predicate: str
    target_label: str
    confidence: float = 0.75


class MockRelationReasoner:
    """按标签规则生成关系的推理器替身。"""

    def __init__(
        self,
        rules: Sequence[RelationRule] = (),
        *,
        model_version: str = "mock-relate-0.1",
    ) -> None:
        self._rules = list(rules)
        self._model_version = model_version
        self.calls = 0

    @property
    def model_version(self) -> str:
        return self._model_version

    def infer(self, frame: Frame, objects: list[TrackedObject]) -> list[Relation]:
        self.calls += 1
        relations: list[Relation] = []
        for rule in self._rules:
            subjects = [obj for obj in objects if obj.label == rule.subject_label]
            targets = [obj for obj in objects if obj.label == rule.target_label]
            for subject in subjects:
                for target in targets:
                    if subject.track_id == target.track_id:
                        continue
                    relations.append(
                        Relation(
                            subject=TrackRef(track_id=subject.track_id, label=subject.label),
                            predicate=rule.predicate,
                            target=TrackRef(track_id=target.track_id, label=target.label),
                            confidence=rule.confidence,
                            model_version=self._model_version,
                            observed_at=frame.captured_at,
                        )
                    )
        return relations
