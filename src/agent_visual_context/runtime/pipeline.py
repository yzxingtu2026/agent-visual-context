"""最小视觉处理流水线。

串联顺序：输入源 -> 检测 -> 跟踪 -> 关系推理 -> 去抖/持续性 -> 时间线 -> 摘要 -> 策略 -> 事件。

设计约束：
- 所有组件都通过协议注入，可被 Mock 或真实适配器替换；
- 任一感知组件异常只降级该组件，不中断整条流水线（除非 `fail_fast`）；
- 时间线有界，快照有窗口，输出带时间戳、置信度、来源、模型版本与有效期。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from time import monotonic
from typing import TypeVar

from ..config import AppConfig
from ..context import SceneSummarizer
from ..domain import (
    Detection,
    Event,
    FaceEstimate,
    FaceObservation,
    Frame,
    Observation,
    PersonCountQuality,
    PersonIdentityMatch,
    PersonSceneSummary,
    Relation,
    Snapshot,
    TrackedObject,
    utc_now,
)
from ..errors import PerceptionError
from ..input.base import FrameSource
from ..logging_setup import get_logger
from ..perception.base import Detector, FaceAnalyzer, RelationReasoner, Tracker
from ..perception.face_association import associate_faces
from ..perception.person_identity import IdentityMatcher, IdentityStore, StoredIdentityMatcher
from ..policies.base import Policy
from ..temporal import BoundedTimeline, PersistenceGate
from .bus import EventBus
from .status import ComponentState, PipelineState, PipelineStatus

logger = get_logger("runtime.pipeline")

_T = TypeVar("_T")


@dataclass(slots=True)
class FrameResult:
    """单帧处理结果，便于测试与调试。

    计数字段（`detections`/`tracked_objects`/`relations`）用于轻量统计；
    `*_items` 字段承载该帧实际的对象，供可视化/调试示例直接绘制检测框、
    跟踪目标与关系连线，避免上层为拿框而重复推理。默认为空，不影响既有行为。
    """

    frame_id: str
    detections: int = 0
    tracked_objects: int = 0
    relations: int = 0
    observations: list[Observation] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    degraded_components: tuple[str, ...] = ()
    detection_items: tuple[Detection, ...] = ()
    tracked_items: tuple[TrackedObject, ...] = ()
    relation_items: tuple[Relation, ...] = ()
    face_items: tuple[FaceObservation, ...] = ()
    identity_items: tuple[PersonIdentityMatch, ...] = ()
    persons: PersonSceneSummary | None = None


@dataclass(slots=True)
class RunResult:
    """一次流水线运行的汇总结果。"""

    frames_processed: int = 0
    observations: list[Observation] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    status: PipelineStatus = field(default_factory=PipelineStatus)
    snapshot: Snapshot | None = None


class Pipeline:
    """把可替换组件编排成一条有界、可降级的视觉上下文流水线。"""

    def __init__(
        self,
        *,
        config: AppConfig,
        source: FrameSource,
        detector: Detector,
        tracker: Tracker,
        face_analyzer: FaceAnalyzer | None = None,
        identity_store: IdentityStore | None = None,
        identity_matcher: IdentityMatcher | None = None,
        reasoner: RelationReasoner,
        timeline: BoundedTimeline | None = None,
        gate: PersistenceGate | None = None,
        summarizer: SceneSummarizer | None = None,
        policies: Sequence[Policy] = (),
        bus: EventBus | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._config = config
        self._source = source
        self._detector = detector
        self._tracker = tracker
        self._face_analyzer = face_analyzer
        self._identity_store = identity_store
        self._identity_matcher = identity_matcher
        self._reasoner = reasoner
        self._clock = clock
        self.timeline = (
            timeline
            if timeline is not None
            else BoundedTimeline(
                capacity=config.timeline_capacity,
                default_ttl=timedelta(seconds=config.observation_ttl_seconds),
                clock=clock,
            )
        )
        self.gate = (
            gate
            if gate is not None
            else PersistenceGate(
                min_hits=config.persistence_min_hits,
                min_seconds=config.persistence_min_seconds,
            )
        )
        self.summarizer = (
            summarizer
            if summarizer is not None
            else SceneSummarizer(
                window_seconds=config.window_seconds,
                clock=clock,
            )
        )
        self.policies = list(policies)
        self.bus = bus if bus is not None else EventBus()
        self.status = PipelineStatus(scene_id=config.scene_id)
        self._register_components()

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def source(self) -> FrameSource:
        """当前流水线的输入源；实时链路由 `LiveRuntime` 驱动其采集生命周期。"""
        return self._source

    @property
    def identity_store(self) -> IdentityStore | None:
        return self._identity_store

    @property
    def identity_matcher(self) -> IdentityMatcher | None:
        return self._identity_matcher

    @property
    def clock(self) -> Callable[[], datetime]:
        return self._clock

    def run(self, *, max_frames: int | None = None) -> RunResult:
        """运行流水线直到数据源耗尽或达到 `max_frames`。"""
        limit = self._config.max_frames if max_frames is None else max_frames
        self.status.state = PipelineState.RUNNING
        result = RunResult(status=self.status)

        self._source.open()
        try:
            while limit <= 0 or result.frames_processed < limit:
                frame = self._read_frame()
                if frame is None:
                    break
                frame_result = self.process_frame(frame)
                result.frames_processed += 1
                result.observations.extend(frame_result.observations)
                result.events.extend(frame_result.events)
        except Exception as exc:
            self.status.state = PipelineState.FAILED
            self.status.last_error = str(exc)
            logger.exception("流水线异常终止")
            if self._config.fail_fast:
                raise
        finally:
            self._source.close()

        now = self._clock()
        result.snapshot = self.summarizer.build(
            self.timeline, now=now, degraded=self.status.is_degraded
        )
        self.status.frames_processed = result.frames_processed
        self.status.dropped_items = self.timeline.dropped_total
        self.status.refresh(now=now)
        if self.status.state != PipelineState.FAILED:
            self.status.state = (
                PipelineState.DEGRADED if self.status.is_degraded else PipelineState.STOPPED
            )
        result.status = self.status
        logger.info(
            "流水线结束 frames=%s observations=%s events=%s state=%s",
            result.frames_processed,
            len(result.observations),
            len(result.events),
            self.status.state.value,
        )
        return result

    def process_frame(self, frame: Frame) -> FrameResult:
        """处理单帧，返回该帧产生的观察与事件。"""
        moment = frame.captured_at
        degraded: list[str] = []
        result = FrameResult(frame_id=frame.frame_id)

        detections, ok = self._call("detector", lambda: self._detector.detect(frame), now=moment)
        if not ok:
            degraded.append("detector")
        detections = [
            item for item in detections if item.confidence >= self._config.min_detection_confidence
        ]
        result.detections = len(detections)
        people = [item for item in detections if item.label == "person"]
        detector_ok = ok

        tracked, ok = self._call(
            "tracker",
            lambda: self._tracker.update(frame, detections),
            now=moment,
        )
        if not ok:
            degraded.append("tracker")
            tracked = []
        result.tracked_objects = len(tracked)

        faces: list[FaceObservation] = []
        face_ok = False
        face_analyzer = self._face_analyzer
        if face_analyzer is not None:
            face_started = monotonic()
            faces, face_ok = self._call(
                "face_analyzer", lambda: face_analyzer.analyze(frame), now=moment
            )
            self.status.register("face_analyzer").last_latency_ms = (
                monotonic() - face_started
            ) * 1000
            if face_ok:
                faces = associate_faces(faces, tracked, min_quality=self._config.face_min_quality)
                self.status.mark_ok("face_analyzer", now=moment)
            else:
                degraded.append("face_analyzer")
        identity_matches: tuple[PersonIdentityMatch, ...] = ()
        identity_matcher = self._identity_matcher
        if identity_matcher is not None and face_ok:
            matches, match_ok = self._call(
                "identity_matcher",
                lambda: identity_matcher.match_faces(
                    faces, source_id=frame.source_id, observed_at=moment,
                    ttl_seconds=self._config.observation_ttl_seconds,
                ),
                now=moment,
            )
            identity_matches = tuple(matches)
            if not match_ok:
                degraded.append("identity_matcher")
            else:
                self.status.mark_ok("identity_matcher", now=moment)
        elif isinstance(identity_matcher, StoredIdentityMatcher):
            identity_matcher.reset()
        result.face_items = tuple(face.model_copy(update={"embedding": None}) for face in faces)
        result.identity_items = identity_matches
        result.persons = self.timeline.add_person_summary(
            PersonSceneSummary(
                scene_id=self._config.scene_id,
                source_id=frame.source_id,
                sampled_at=moment,
                expires_at=moment + timedelta(seconds=self._config.observation_ttl_seconds),
                current_person_count=(
                    len(people) if detector_ok and "person" in self._config.detector_classes else None
                ),
                recognizable_face_count=len(faces) if face_ok else None,
                face_estimates=tuple(
                    FaceEstimate(
                        track_id=face.track_id, age_band=face.age_band,
                        age_confidence=face.age_confidence,
                        apparent_gender=face.apparent_gender,
                        gender_confidence=face.gender_confidence,
                    )
                    for face in faces if face.track_id is not None
                    and (face.age_band is not None or face.apparent_gender is not None)
                ),
                identity_matches=identity_matches,
                confidence=(
                    min(item.confidence for item in people)
                    if people and detector_ok and "person" in self._config.detector_classes
                    else None
                ),
                quality=(
                    PersonCountQuality.DEGRADED
                    if not detector_ok
                    else PersonCountQuality.UNAVAILABLE
                    if "person" not in self._config.detector_classes
                    else PersonCountQuality.DETECTED
                    if people
                    else PersonCountQuality.NO_DETECTION
                ),
            )
        )

        relations, ok = self._call(
            "reasoner",
            lambda: self._reasoner.infer(frame, tracked),
            now=moment,
        )
        if not ok:
            degraded.append("reasoner")
        relations = [
            item for item in relations if item.confidence >= self._config.min_relation_confidence
        ]
        result.relations = len(relations)

        stable_relations = self.gate.update(relations, now=moment)
        for relation in stable_relations:
            observation = Observation(
                scene_id=self._config.scene_id,
                subject=relation.subject,
                predicate=relation.predicate,
                target=relation.target,
                confidence=relation.confidence,
                observed_at=relation.observed_at,
                expires_at=relation.observed_at
                + timedelta(seconds=self._config.observation_ttl_seconds),
                source_id=self._source.source_id,
                model_version=relation.model_version,
            )
            stored = self.timeline.add_observation(observation)
            result.observations.append(stored)
            self.status.observations_emitted += 1

        snapshot = self.summarizer.build(
            self.timeline, now=moment, degraded=bool(degraded) or self.status.is_degraded
        )
        for policy in self.policies:
            for event in policy.evaluate(snapshot, now=moment):
                self.timeline.add_event(event)
                self.bus.publish(event)
                self.status.events_emitted += 1
                result.events.append(event)

        for name in ("detector", "tracker", "reasoner"):
            if name not in degraded:
                self.status.mark_ok(name, now=moment)
        result.degraded_components = tuple(degraded)
        # 承载本帧可绘制对象，供可视化/调试示例直接取用（避免重复推理）。
        result.detection_items = tuple(detections)
        result.tracked_items = tuple(tracked)
        result.relation_items = tuple(relations)
        self.status.frames_processed += 1
        self.status.refresh(now=moment)
        return result

    def _read_frame(self) -> Frame | None:
        try:
            return self._source.read()
        except Exception as exc:
            self.status.mark_failure(
                "source", error=str(exc), now=self._clock(), fatal=self._config.fail_fast
            )
            logger.error("输入源读取失败，降级为空帧：%s", exc)
            if self._config.fail_fast:
                raise
            return None

    def _call(
        self,
        component: str,
        action: Callable[[], Iterable[_T]],
        *,
        now: datetime,
    ) -> tuple[list[_T], bool]:
        """执行组件调用；失败时记录降级状态并返回空结果。"""
        try:
            return list(action()), True
        except Exception as exc:
            error = "身份匹配调用失败" if component == "identity_matcher" else str(exc)
            self.status.mark_failure(
                component, error=error, now=now, fatal=self._config.fail_fast
            )
            logger.warning("组件 %s 调用失败，已降级：%s", component, error)
            if self._config.fail_fast:
                if component == "identity_matcher":
                    raise PerceptionError(f"组件 {component} 调用失败：{error}") from None
                raise PerceptionError(f"组件 {component} 调用失败：{error}") from exc
            return [], False

    def _register_components(self) -> None:
        self.status.register("source", model_version=self._source.source_id)
        self.status.register("detector", model_version=self._detector.model_version)
        self.status.register("tracker", model_version=self._tracker.model_version)
        if self._face_analyzer is not None:
            self.status.register("face_analyzer", model_version=self._face_analyzer.model_version)
        else:
            self.status.disable("face_analyzer")
        if self._identity_matcher is not None:
            self.status.register("identity_matcher")
        else:
            self.status.disable("identity_matcher")
        self.status.register("reasoner", model_version=self._reasoner.model_version)
        if not self.policies:
            self.status.disable("policies")
        for component in self.status.components.values():
            if component.state != ComponentState.DISABLED and component.model_version == "unknown":
                component.model_version = "unspecified"
