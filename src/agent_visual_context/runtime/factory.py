"""流水线装配工厂。

统一在此完成组件接线，CLI、示例与测试都通过它构造流水线，避免各处重复组装。
需要替换组件时通过参数注入，不要在业务代码里直接 new 具体实现。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from pathlib import Path

from ..config import AppConfig
from ..context import SceneSummarizer
from ..domain import BBox, utc_now
from ..input import ScriptedFrameSource, frame_source_from_path, synthetic_frames
from ..input.base import FrameSource
from ..perception import (
    MockRelationReasoner,
    MockTarget,
    MockTracker,
    RelationRule,
    StaticSceneDetector,
)
from ..perception.base import Detector, RelationReasoner, Tracker
from ..policies import GreetingCandidatePolicy, Policy
from ..temporal import BoundedTimeline, PersistenceGate
from .bus import EventBus
from .pipeline import Pipeline

DEFAULT_TARGETS: tuple[MockTarget, ...] = (
    MockTarget(label="person", bbox=BBox(x=80, y=120, width=120, height=260), confidence=0.92),
    MockTarget(label="screen", bbox=BBox(x=300, y=80, width=280, height=200), confidence=0.88),
)

DEFAULT_RELATION_RULES: tuple[RelationRule, ...] = (
    RelationRule(
        subject_label="person", predicate="looking_at", target_label="screen", confidence=0.82
    ),
)


def build_mock_pipeline(
    config: AppConfig,
    *,
    clock: Callable[[], datetime] = utc_now,
    source: FrameSource | None = None,
    detector: Detector | None = None,
    tracker: Tracker | None = None,
    reasoner: RelationReasoner | None = None,
    policies: Sequence[Policy] | None = None,
    bus: EventBus | None = None,
    targets: Sequence[MockTarget] = DEFAULT_TARGETS,
    relation_rules: Sequence[RelationRule] = DEFAULT_RELATION_RULES,
) -> Pipeline:
    """构造一条全部由 Mock 组件组成的可运行流水线。"""
    now = clock()
    interval = timedelta(seconds=1 / config.target_fps)
    frame_count = config.max_frames or 4
    resolved_source = source or ScriptedFrameSource(
        synthetic_frames(
            frame_count,
            source_id=config.source_id,
            width=config.frame_width,
            height=config.frame_height,
            # 以“最后一帧即当前时刻”为锚点回放，保证时间窗口过滤符合真实语义
            start=now - interval * (frame_count - 1),
            interval=interval,
        ),
        source_id=config.source_id,
    )

    return Pipeline(
        config=config,
        source=resolved_source,
        detector=detector or StaticSceneDetector(targets),
        tracker=tracker or MockTracker(),
        reasoner=reasoner or MockRelationReasoner(relation_rules),
        timeline=BoundedTimeline(
            capacity=config.timeline_capacity,
            default_ttl=timedelta(seconds=config.observation_ttl_seconds),
            clock=clock,
        ),
        gate=PersistenceGate(
            min_hits=config.persistence_min_hits,
            min_seconds=config.persistence_min_seconds,
        ),
        summarizer=SceneSummarizer(window_seconds=config.window_seconds, clock=clock),
        policies=(
            policies
            if policies is not None
            else [
                GreetingCandidatePolicy(
                    scene_id=config.scene_id,
                    min_confidence=0.6,
                    cooldown=timedelta(seconds=config.greeting_cooldown_seconds),
                )
            ]
        ),
        bus=bus,
        clock=clock,
    )


def build_offline_pipeline(
    config: AppConfig,
    path: Path | str,
    *,
    clock: Callable[[], datetime] = utc_now,
    source_id: str | None = None,
    detector: Detector | None = None,
    tracker: Tracker | None = None,
    reasoner: RelationReasoner | None = None,
    policies: Sequence[Policy] | None = None,
    bus: EventBus | None = None,
    targets: Sequence[MockTarget] = DEFAULT_TARGETS,
    relation_rules: Sequence[RelationRule] = DEFAULT_RELATION_RULES,
) -> Pipeline:
    """构造以离线素材（图片/图片目录/视频）为输入的 Mock 流水线。

    与 `build_mock_pipeline` 复用同一套领域模型、时间线与策略组件，
    仅把输入源替换为真实文件适配器；感知组件仍为 Mock（PoC-3/PoC-4 接入真实模型）。
    `source_id` 未显式给出时取素材文件/目录名，并同步回配置。
    """
    source = frame_source_from_path(
        path, source_id=source_id, target_fps=config.target_fps, clock=clock
    )
    resolved_config = config.model_copy(update={"source_id": source.source_id})
    return build_mock_pipeline(
        resolved_config,
        clock=clock,
        source=source,
        detector=detector,
        tracker=tracker,
        reasoner=reasoner,
        policies=policies,
        bus=bus,
        targets=targets,
        relation_rules=relation_rules,
    )
