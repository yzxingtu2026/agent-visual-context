"""集成测试：Mock 组件注入流水线后的端到端行为与降级能力。"""

from __future__ import annotations

from datetime import timedelta

import pytest

from agent_visual_context.config import AppConfig
from agent_visual_context.domain import Event, Frame, Observation, Relation, TrackedObject
from agent_visual_context.errors import PerceptionError
from agent_visual_context.input import ScriptedFrameSource, synthetic_frames
from agent_visual_context.policies import GreetingCandidatePolicy
from agent_visual_context.runtime import ComponentState, PipelineState, build_mock_pipeline
from tests.conftest import T0, FakeClock


class FailingReasoner:
    """模拟推理器超时/崩溃，用于验证降级路径。"""

    model_version = "broken-relate-0.1"

    def infer(self, frame: Frame, objects: list[TrackedObject]) -> list[Relation]:
        del frame, objects
        msg = "推理超时"
        raise PerceptionError(msg)


def test_mock_pipeline_produces_bounded_observations(config: AppConfig, clock: FakeClock) -> None:
    pipeline = build_mock_pipeline(config, clock=clock)

    result = pipeline.run()

    assert result.frames_processed == config.max_frames
    # 去抖要求至少 2 次命中且持续 1 秒，因此前若干帧不会产出观察
    assert 0 < len(result.observations) < result.frames_processed
    assert all(isinstance(item, Observation) for item in result.observations)
    assert all(item.expires_at > item.observed_at for item in result.observations)
    assert all(item.source_id == config.source_id for item in result.observations)
    assert pipeline.status.state is PipelineState.STOPPED
    assert not pipeline.status.is_degraded


def test_snapshot_is_deduped_and_window_bounded(config: AppConfig, clock: FakeClock) -> None:
    pipeline = build_mock_pipeline(config, clock=clock)

    result = pipeline.run()
    assert result.snapshot is not None

    snapshot = result.snapshot
    keys = {(item.subject.track_id, item.predicate) for item in snapshot.observations}
    assert len(keys) == len(snapshot.observations)
    assert snapshot.duration_seconds == pytest.approx(config.window_seconds)
    assert snapshot.window_end == clock()


def test_process_frame_exposes_drawable_items(config: AppConfig, clock: FakeClock) -> None:
    """可视化示例依赖 FrameResult 承载本帧检测框/跟踪目标/关系，无需重复推理。"""
    pipeline = build_mock_pipeline(config, clock=clock)
    frames = synthetic_frames(
        1,
        source_id=config.source_id,
        start=T0,
        interval=timedelta(seconds=1 / config.target_fps),
    )
    pipeline.source.open()

    frame_result = pipeline.process_frame(frames[0])

    assert len(frame_result.detection_items) == frame_result.detections == 2
    assert len(frame_result.tracked_items) == frame_result.tracked_objects == 2
    assert {item.label for item in frame_result.tracked_items} == {"person", "screen"}
    assert all(item.bbox.width > 0 for item in frame_result.tracked_items)
    # Mock 关系规则 person looking_at screen，关系两端引用稳定 track_id
    assert len(frame_result.relation_items) == frame_result.relations == 1
    relation = frame_result.relation_items[0]
    assert relation.predicate == "looking_at"
    assert relation.subject.track_id == "person-01"
    assert relation.target.track_id == "screen-01"
    pipeline.source.close()


def test_pipeline_degrades_when_reasoner_fails(config: AppConfig, clock: FakeClock) -> None:
    pipeline = build_mock_pipeline(config, clock=clock, reasoner=FailingReasoner())

    result = pipeline.run()

    assert result.frames_processed == config.max_frames
    assert result.observations == []
    assert pipeline.status.components["reasoner"].state is ComponentState.DEGRADED
    assert pipeline.status.state is PipelineState.DEGRADED
    assert "推理超时" in (pipeline.status.last_error or "")


def test_fail_fast_propagates_perception_error(clock: FakeClock) -> None:
    config = AppConfig(scene_id="scene-test", max_frames=4, fail_fast=True)
    pipeline = build_mock_pipeline(config, clock=clock, reasoner=FailingReasoner())

    with pytest.raises(PerceptionError):
        pipeline.run()

    assert pipeline.status.state is PipelineState.FAILED


def test_source_is_opened_and_closed_once(config: AppConfig, clock: FakeClock) -> None:
    frames = synthetic_frames(
        config.max_frames,
        source_id=config.source_id,
        start=T0,
        interval=timedelta(seconds=1 / config.target_fps),
    )
    source = ScriptedFrameSource(frames)
    pipeline = build_mock_pipeline(config, clock=clock, source=source)

    pipeline.run()

    assert source.open_count == 1
    assert source.close_count == 1
    assert not source.is_open


def test_greeting_policy_emits_single_candidate_within_cooldown(clock: FakeClock) -> None:
    config = AppConfig(
        scene_id="scene-test",
        max_frames=6,
        target_fps=2.0,
        persistence_min_hits=2,
        persistence_min_seconds=1.0,
    )
    policy = GreetingCandidatePolicy(
        scene_id=config.scene_id,
        predicates=("looking_at",),
        min_confidence=0.6,
        cooldown=timedelta(seconds=30),
    )
    pipeline = build_mock_pipeline(config, clock=clock, policies=[policy])

    result = pipeline.run()

    assert [event.kind for event in result.events] == ["greeting-candidate"]
    candidate: Event = result.events[0]
    assert candidate.payload["track_id"] == "person-01"
    assert candidate.epistemic_status.value == "rule-event"


def test_event_bus_delivers_events_to_subscribers(config: AppConfig, clock: FakeClock) -> None:
    pipeline = build_mock_pipeline(config, clock=clock)
    received: list[Event] = []
    unsubscribe = pipeline.bus.subscribe(received.append)

    pipeline.run()

    assert len(received) == 1
    unsubscribe()
    assert pipeline.bus.listener_count == 0


def test_broken_subscriber_does_not_break_pipeline(config: AppConfig, clock: FakeClock) -> None:
    pipeline = build_mock_pipeline(config, clock=clock)

    def broken_listener(event: Event) -> None:
        del event
        msg = "订阅者异常"
        raise RuntimeError(msg)

    pipeline.bus.subscribe(broken_listener)
    result = pipeline.run()

    assert result.frames_processed == config.max_frames
    assert pipeline.status.state is not PipelineState.FAILED
