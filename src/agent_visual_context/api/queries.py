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
from ..perception.person_identity import IdentityMatcher, IdentityStore, StoredIdentityMatcher
from ..runtime.bus import EventBus, EventListener
from ..runtime.live import LiveRuntime
from ..runtime.metrics import MetricsSnapshot
from ..runtime.pipeline import Pipeline
from ..runtime.status import PipelineStatus
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
        live: LiveRuntime | None = None,
        identity_store: IdentityStore | None = None,
        identity_matcher: IdentityMatcher | None = None,
    ) -> None:
        self._scene_id = scene_id
        self._timeline = timeline
        self._summarizer = summarizer
        self._bus = bus
        self._clock = clock
        self._live = live
        self._identity_store = identity_store
        self._identity_matcher = identity_matcher
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
            identity_store=pipeline.identity_store,
            identity_matcher=pipeline.identity_matcher,
        )

    @classmethod
    def from_live_runtime(cls, runtime: LiveRuntime) -> VisualContextApi:
        """基于实时 Sidecar 运行时创建查询入口，额外暴露健康状态与运行指标。"""
        pipeline = runtime.pipeline
        return cls(
            scene_id=pipeline.config.scene_id,
            timeline=pipeline.timeline,
            summarizer=pipeline.summarizer,
            bus=pipeline.bus,
            clock=pipeline.clock,
            live=runtime,
            identity_store=pipeline.identity_store,
            identity_matcher=pipeline.identity_matcher,
        )

    def register_anonymous_person(
        self,
        embedding: list[float] | tuple[float, ...],
        *,
        model_version: str,
        consent: bool = False,
        person_id: str | None = None,
    ) -> str:
        """登记匿名特征；向量不会进入快照、日志或事件载荷。"""
        if self._identity_store is None:
            raise RuntimeError("匿名人物匹配未配置")
        return self._identity_store.register(
            embedding,
            model_version=model_version,
            now=self._clock(),
            consent=consent,
            person_id=person_id,
        )

    def add_anonymous_person_sample(
        self,
        person_id: str,
        embedding: list[float] | tuple[float, ...],
        *,
        model_version: str,
        consent: bool = False,
    ) -> None:
        if self._identity_store is None:
            raise RuntimeError("匿名人物匹配未配置")
        self._identity_store.add_sample(
            person_id,
            embedding,
            model_version=model_version,
            now=self._clock(),
            consent=consent,
        )

    def delete_anonymous_person(self, person_id: str) -> None:
        if self._identity_store is not None:
            self._identity_store.delete(person_id)
        if isinstance(self._identity_matcher, StoredIdentityMatcher):
            self._identity_matcher.forget_person(person_id)

    def disable_anonymous_person(self, person_id: str) -> None:
        if self._identity_store is not None:
            self._identity_store.disable(person_id)
        if isinstance(self._identity_matcher, StoredIdentityMatcher):
            self._identity_matcher.forget_person(person_id)

    def get_health(self) -> PipelineStatus | None:
        """返回实时链路的健康状态（组件级 + 整体）；无实时运行时时返回 `None`。

        这是宿主链路（大屏/语音）判断视觉是否可用的唯一口径：视觉降级时读取到
        `degraded`/`failed` 状态即可跳过视觉增强，而不必等待或阻塞。
        """
        return self._live.status if self._live is not None else None

    def get_metrics(self, *, now: datetime | None = None) -> MetricsSnapshot | None:
        """返回实时链路运行指标快照（有效 FPS、延迟、丢帧率、失败率）；无实时运行时时返回 `None`。"""
        if self._live is None:
            return None
        moment = now if now is not None else self._clock()
        return self._live.metrics.snapshot(now=moment)

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
