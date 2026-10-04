"""场景摘要：把有界时间线压缩成可注入话轮上下文的快照。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

from ..domain import Event, Observation, Snapshot, utc_now
from ..temporal import BoundedTimeline


class SceneSummarizer:
    """按时间窗口生成场景摘要，输出必须是截断、去重且有界的。"""

    def __init__(
        self,
        *,
        window_seconds: float = 4.0,
        max_highlights: int = 6,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._window = timedelta(seconds=window_seconds)
        self._max_highlights = max_highlights
        self._clock = clock

    def build(
        self,
        timeline: BoundedTimeline,
        *,
        now: datetime | None = None,
        degraded: bool = False,
    ) -> Snapshot:
        moment = now if now is not None else self._clock()
        start = moment - self._window
        timeline.prune(now=moment)
        observations = _dedupe(timeline.observations(start=start, end=moment, now=moment))
        events = timeline.events(start=start, end=moment, now=moment)
        scene_id = observations[0].scene_id if observations else ""

        return Snapshot(
            scene_id=scene_id,
            generated_at=moment,
            window_start=start,
            window_end=moment,
            observations=observations,
            events=events,
            highlights=self._highlights(observations, events),
            degraded=degraded,
        )

    def _highlights(self, observations: list[Observation], events: list[Event]) -> list[str]:
        highlights = [
            f"最近 {self._window.total_seconds():.0f} 秒观察到 {len(observations)} 条有效视觉观察"
        ]
        for observation in observations[: self._max_highlights]:
            target = observation.target
            if target is None:
                highlights.append(
                    f"- {observation.subject.label} {observation.predicate}"
                    f"（置信度 {observation.confidence:.2f}）"
                )
            else:
                highlights.append(
                    f"- {observation.subject.label} {observation.predicate} {target.label}"
                    f"（置信度 {observation.confidence:.2f}）"
                )
        if events:
            highlights.append(f"- 规则事件 {len(events)} 条")
        return highlights[: self._max_highlights + 2]


def _dedupe(observations: list[Observation]) -> list[Observation]:
    """同一 (subject, predicate, target) 只保留最新一条，避免快照膨胀。"""
    latest: dict[tuple[str, str, str | None], Observation] = {}
    for observation in observations:
        key = (
            observation.subject.track_id,
            observation.predicate,
            observation.target.track_id if observation.target else None,
        )
        current = latest.get(key)
        if current is None or observation.observed_at >= current.observed_at:
            latest[key] = observation
    return sorted(latest.values(), key=lambda item: item.observed_at)
