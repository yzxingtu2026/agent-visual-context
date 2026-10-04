"""契约测试：Mock/替身实现必须满足可替换组件协议。

这些测试同时是 mypy 的类型契约样本——函数签名使用协议类型，
传入 Mock 实现若能通过类型检查，说明后续真实适配器可以按同样方式接入。
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from agent_visual_context.domain import BBox, Detection, Relation, TrackedObject
from agent_visual_context.errors import FrameSourceError
from agent_visual_context.input import ScriptedFrameSource, synthetic_frames
from agent_visual_context.input.base import FrameSource
from agent_visual_context.perception import (
    MockRelationReasoner,
    MockTarget,
    MockTracker,
    RelationRule,
    ScriptedDetector,
    StaticSceneDetector,
)
from agent_visual_context.perception.base import Detector, RelationReasoner, Tracker
from tests.conftest import T0


def consume_source(source: FrameSource) -> list[str]:
    source.open()
    frames: list[str] = []
    while (frame := source.read()) is not None:
        frames.append(frame.frame_id)
    source.close()
    return frames


def test_scripted_source_satisfies_frame_source_protocol() -> None:
    frames = synthetic_frames(3, source_id="mock-source", start=T0, interval=timedelta(seconds=1))
    source = ScriptedFrameSource(frames)

    assert consume_source(source) == [frame.frame_id for frame in frames]
    assert source.open_count == 1
    assert source.close_count == 1


def test_source_read_before_open_raises() -> None:
    source = ScriptedFrameSource(synthetic_frames(1, start=T0))

    with pytest.raises(FrameSourceError):
        source.read()


def test_source_close_is_idempotent() -> None:
    source = ScriptedFrameSource(synthetic_frames(1, start=T0))
    source.open()

    source.close()
    source.close()

    assert source.close_count == 1
    assert not source.is_open


def test_static_detector_satisfies_detector_protocol() -> None:
    detector: Detector = StaticSceneDetector(
        [MockTarget(label="person", bbox=BBox(x=0, y=0, width=10, height=10))]
    )
    frame = synthetic_frames(1, start=T0)[0]

    detections = detector.detect(frame)

    assert detector.model_version == "mock-detector-0.1"
    assert [item.label for item in detections] == ["person"]
    assert detections[0].detected_at == frame.captured_at


def test_scripted_detector_returns_per_frame_results() -> None:
    frame = synthetic_frames(1, start=T0)[0]
    detection = Detection(
        label="phone",
        bbox=BBox(x=1, y=2, width=3, height=4),
        confidence=0.7,
        detected_at=frame.captured_at,
    )
    detector: Detector = ScriptedDetector({frame.frame_id: [detection]})

    assert detector.detect(frame) == [detection]
    assert detector.detect(synthetic_frames(1, source_id="other", start=T0)[0]) == []


def test_mock_tracker_assigns_stable_track_ids() -> None:
    tracker: Tracker = MockTracker()
    frame = synthetic_frames(1, start=T0)[0]
    detections = [
        Detection(
            label="person",
            bbox=BBox(x=0, y=0, width=10, height=10),
            confidence=0.9,
            detected_at=frame.captured_at,
        ),
        Detection(
            label="screen",
            bbox=BBox(x=20, y=0, width=10, height=10),
            confidence=0.9,
            detected_at=frame.captured_at,
        ),
    ]

    tracked: list[TrackedObject] = tracker.update(frame, detections)
    next_frame = synthetic_frames(1, start=T0 + timedelta(seconds=1))[0]
    tracked_again = tracker.update(next_frame, detections)

    assert [item.track_id for item in tracked] == ["person-01", "screen-01"]
    assert [item.track_id for item in tracked_again] == ["person-01", "screen-01"]
    assert tracked_again[0].first_seen == frame.captured_at
    assert tracked_again[0].last_seen == next_frame.captured_at


def test_mock_reasoner_satisfies_relation_reasoner_protocol() -> None:
    reasoner: RelationReasoner = MockRelationReasoner(
        [RelationRule(subject_label="person", predicate="looking_at", target_label="screen")]
    )
    tracker = MockTracker()
    frame = synthetic_frames(1, start=T0)[0]
    detections = [
        Detection(
            label=label,
            bbox=BBox(x=offset, y=0, width=10, height=10),
            confidence=0.9,
            detected_at=frame.captured_at,
        )
        for offset, label in enumerate(("person", "screen"))
    ]
    tracked = tracker.update(frame, detections)

    relations: list[Relation] = reasoner.infer(frame, tracked)

    assert len(relations) == 1
    assert relations[0].key == ("person-01", "looking_at", "screen-01")
    assert relations[0].observed_at == frame.captured_at
