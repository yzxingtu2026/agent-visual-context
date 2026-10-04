"""迎宾候选策略（骨架版）。

首期只实现边界与最小规则：当窗口内存在面向交互区域且置信度达标的观察时，
产出一条 `greeting-candidate` 规则事件，并受冷却时间约束。
真实阈值、区域判定与大屏状态联动由后续 PoC Issue 决定，此处不实现任何播报动作。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

from ..domain import Event, Observation, Snapshot


class GreetingCandidatePolicy:
    """基于视觉观察提出迎宾候选，绝不直接触发业务动作。"""

    KIND = "greeting-candidate"

    def __init__(
        self,
        *,
        scene_id: str,
        predicates: Iterable[str] = ("facing", "looking_at"),
        target_labels: Iterable[str] = ("screen",),
        min_confidence: float = 0.6,
        cooldown: timedelta = timedelta(seconds=30),
        source_id: str = "policy:greeting",
    ) -> None:
        self._scene_id = scene_id
        self._predicates = set(predicates)
        self._target_labels = set(target_labels)
        self._min_confidence = min_confidence
        self._cooldown = cooldown
        self._source_id = source_id
        self._last_fired_at: datetime | None = None

    @property
    def name(self) -> str:
        return "greeting-candidate"

    def evaluate(self, snapshot: Snapshot, *, now: datetime) -> list[Event]:
        if not self._cooldown_passed(now):
            return []

        candidate = self._pick(snapshot.observations)
        if candidate is None:
            return []

        self._last_fired_at = now
        return [
            Event(
                scene_id=snapshot.scene_id or self._scene_id,
                kind=self.KIND,
                observed_at=now,
                expires_at=now + self._cooldown,
                confidence=candidate.confidence,
                source_id=self._source_id,
                payload={
                    "track_id": candidate.subject.track_id,
                    "predicate": candidate.predicate,
                    "target_track_id": candidate.target.track_id if candidate.target else None,
                    "observed_at": candidate.observed_at.isoformat(),
                },
            )
        ]

    def reset(self) -> None:
        self._last_fired_at = None

    def _cooldown_passed(self, now: datetime) -> bool:
        if self._last_fired_at is None:
            return True
        return (now - self._last_fired_at) >= self._cooldown

    def _pick(self, observations: list[Observation]) -> Observation | None:
        matches = [
            observation
            for observation in observations
            if observation.predicate in self._predicates
            and observation.confidence >= self._min_confidence
            and observation.target is not None
            and observation.target.label in self._target_labels
        ]
        if not matches:
            return None
        return max(matches, key=lambda item: (item.confidence, item.observed_at))
