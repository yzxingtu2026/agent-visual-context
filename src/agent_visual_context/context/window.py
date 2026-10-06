"""指定时间窗口快照：把有界时间线的某个 [start, end] 区间压缩成可注入的快照。

与 `SceneSummarizer`（“当前时刻往前 window_seconds”的场景摘要）互补：本模块面向
**调用方显式给定的时间窗口**（例如某次语音话轮的起止时间），同样保证截断、去重与过期过滤，
使 Agent 侧不必重复实现时间线查询逻辑。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from ..domain import Observation, Snapshot, utc_now
from ..temporal import BoundedTimeline


def dedupe_observations(observations: list[Observation]) -> list[Observation]:
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


class WindowSnapshotBuilder:
    """按显式时间窗口生成快照；只读时间线，不做写入或裁剪。"""

    def __init__(self, *, clock: Callable[[], datetime] = utc_now) -> None:
        self._clock = clock

    def build(
        self,
        timeline: BoundedTimeline,
        *,
        start: datetime,
        end: datetime | None = None,
        scene_id: str = "",
        label: str | None = None,
    ) -> Snapshot:
        """返回 `[start, end]` 窗口内截断、去重且过滤过期项后的快照。

        - `end` 缺省时取当前时钟；
        - `scene_id` 缺省时回退到窗口内首条观察的场景，仍为空则留空由调用方补齐；
        - `label` 用于生成可读摘要（例如话轮 id）。
        """
        moment = end if end is not None else self._clock()
        observations = dedupe_observations(
            timeline.observations(start=start, end=moment, now=moment)
        )
        events = timeline.events(start=start, end=moment, now=moment)
        persons = timeline.latest_person_summary(start=start, end=moment, now=moment)
        resolved_scene = scene_id or (
            observations[0].scene_id if observations else (persons.scene_id if persons else "")
        )
        scope = f"话轮 {label} 期间" if label else "指定窗口内"
        highlights = [f"{scope}观察到 {len(observations)} 条视觉观察、{len(events)} 条规则事件"]
        return Snapshot(
            scene_id=resolved_scene,
            generated_at=moment,
            window_start=start,
            window_end=moment,
            observations=observations,
            events=events,
            persons=persons,
            highlights=highlights,
        )
