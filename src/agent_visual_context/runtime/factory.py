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
    RelateAnythingReasoner,
    RelationRule,
    StaticSceneDetector,
    YoloWorldDetector,
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


def build_detector(
    config: AppConfig,
    *,
    targets: Sequence[MockTarget] = DEFAULT_TARGETS,
) -> Detector:
    """按配置装配目标检测器。

    `detector_backend=yolo-world` 时返回真实 YOLO-World 适配器（首次检测惰性加载模型），
    否则返回 Mock `StaticSceneDetector`。两种实现都满足 `Detector` 协议，
    时间线与上层模块无需感知差异。
    """
    if config.detector_backend == "yolo-world":
        return YoloWorldDetector.from_config(config)
    return StaticSceneDetector(targets)


def build_reasoner(
    config: AppConfig,
    *,
    relation_rules: Sequence[RelationRule] = DEFAULT_RELATION_RULES,
) -> RelationReasoner:
    """按配置装配关系推理器。

    `reasoner_backend=relate-anything` 时返回真实 RelateAnything 适配器（首次推理惰性加载模型），
    否则返回 Mock `MockRelationReasoner`。两种实现都满足 `RelationReasoner` 协议，
    时间线与上层模块无需感知差异。
    """
    if config.reasoner_backend == "relate-anything":
        return RelateAnythingReasoner.from_config(config)
    return MockRelationReasoner(relation_rules)


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
    """构造一条由 Mock 组件组成的可运行流水线。

    检测器默认按 `config.detector_backend` 装配（默认 `mock`），其余感知组件为 Mock；
    需要固定替身时通过 `detector=` 显式注入。
    """
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
        detector=detector or build_detector(config, targets=targets),
        tracker=tracker or MockTracker(),
        reasoner=reasoner or build_reasoner(config, relation_rules=relation_rules),
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
    """构造以离线素材（图片/图片目录/视频）为输入的流水线。

    与 `build_mock_pipeline` 复用同一套领域模型、时间线与策略组件，
    仅把输入源替换为真实文件适配器。检测器默认由 `build_detector(config)` 按
    `detector_backend` 装配：`mock`（默认）用替身，`yolo-world` 用真实适配器；
    也可通过 `detector=` 显式注入（例如注入带假后端的 YOLO-World 做离线验证）。
    跟踪与关系推理仍为 Mock（PoC-3 后续 / PoC-4 接入真实模型）。
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
        detector=detector or build_detector(resolved_config, targets=targets),
        tracker=tracker,
        reasoner=reasoner,
        policies=policies,
        bus=bus,
        targets=targets,
        relation_rules=relation_rules,
    )
