"""Agent 侧查询/订阅接口的最小边界。

首期只提供进程内实现，保持与 README 中约定的接口形态一致：

    get_scene_snapshot()                 获取当前场景摘要
    get_observations(since, until)       查询时间窗口内的观察
    subscribe_events()                   订阅视觉事件
    get_turn_context(turn_id, ...)       获取某个语音话轮的视觉快照

后续接入 WebSocket / Agent 框架时只替换本模块的传输实现，接口签名保持稳定。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..context import SceneSummarizer, WindowSnapshotBuilder
from ..domain import Event, Observation, Snapshot, utc_now
from ..runtime.bus import EventBus, EventListener
from ..runtime.pipeline import Pipeline
from ..temporal import BoundedTimeline


@dataclass(slots=True)
class TurnContext:
    """一次语音话轮对应的有界视觉上下文。"""

    turn_id: str
    snapshot: Snapshot


class VisualContextApi:
    """查询与订阅入口；只暴露有界、带有效期的数据。"""

    def __init__(
        self,
        *,
        scene_id: str,
        timeline: BoundedTimeline,
        summarizer: SceneSummarizer,
        bus: EventBus,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._scene_id = scene_id
        self._timeline = timeline
        self._summarizer = summarizer
        self._bus = bus
        self._clock = clock
        self._window = WindowSnapshotBuilder(clock=clock)

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline) -> VisualContextApi:
        """基于已装配好的流水线创建查询入口，避免重复接线。"""
        return cls(
            scene_id=pipeline.config.scene_id,
            timeline=pipeline.timeline,
            summarizer=pipeline.summarizer,
            bus=pipeline.bus,
            clock=pipeline.clock,
        )

    def get_scene_snapshot(self, *, now: datetime | None = None) -> Snapshot:
        moment = now if now is not None else self._clock()
        snapshot = self._summarizer.build(self._timeline, now=moment)
        if not snapshot.scene_id:
            snapshot = snapshot.model_copy(update={"scene_id": self._scene_id})
        return snapshot

    def get_observations(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        include_expired: bool = False,
        now: datetime | None = None,
    ) -> list[Observation]:
        moment = now if now is not None else self._clock()
        start = since if since is not None else moment - timedelta(seconds=1)
        return self._timeline.observations(
            start=start,
            end=until,
            now=None if include_expired else moment,
        )

    def get_events(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        include_expired: bool = False,
        now: datetime | None = None,
    ) -> list[Event]:
        moment = now if now is not None else self._clock()
        return self._timeline.events(
            start=since,
            end=until,
            now=None if include_expired else moment,
        )

    def subscribe_events(self, listener: EventListener) -> Callable[[], None]:
        """订阅视觉事件，返回取消订阅的回调。"""
        return self._bus.subscribe(listener)

    def get_turn_context(
        self,
        turn_id: str,
        *,
        started_at: datetime,
        ended_at: datetime | None = None,
        now: datetime | None = None,
    ) -> TurnContext:
        """返回某个话轮时间范围内的视觉快照（截断、去重且过滤过期项）。"""
        moment = ended_at if ended_at is not None else (now if now is not None else self._clock())
        snapshot = self._window.build(
            self._timeline,
            start=started_at,
            end=moment,
            scene_id=self._scene_id,
            label=turn_id,
        )
        return TurnContext(turn_id=turn_id, snapshot=snapshot)
