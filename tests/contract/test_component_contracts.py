"""契约测试：Mock/替身实现必须满足可替换组件协议。

这些测试同时是 mypy 的类型契约样本——函数签名使用协议类型，
传入 Mock 实现若能通过类型检查，说明后续真实适配器可以按同样方式接入。
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from agent_visual_context.domain import BBox, Detection, Frame, Relation, TrackedObject
from agent_visual_context.errors import FrameSourceError
from agent_visual_context.input import (
    ImageFrameSource,
    ScriptedFrameSource,
    VideoFrameSource,
    synthetic_frames,
)
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
from agent_visual_context.perception.yolo_world import (
    RawDetection,
    YoloWorldDetector,
    YoloWorldSettings,
)
from tests.conftest import SAMPLE_IMAGE, SAMPLE_VIDEO, T0


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


def test_image_source_satisfies_frame_source_protocol() -> None:
    source: FrameSource = ImageFrameSource([SAMPLE_IMAGE], start=T0)

    assert consume_source(source) == ["sample_image-00000"]


def test_video_source_satisfies_frame_source_protocol() -> None:
    source: FrameSource = VideoFrameSource(SAMPLE_VIDEO, target_fps=2.0, start=T0)

    assert consume_source(source) == [
        "sample_video-00000",
        "sample_video-00005",
        "sample_video-00010",
        "sample_video-00015",
    ]


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


class _StubBackend:
    """契约测试用的最小后端，返回固定的原始检测框。"""

    model_version = "yolo-world:stub"

    def predict(self, image: object, **kwargs: object) -> list[RawDetection]:
        return [
            RawDetection(x1=80, y1=120, x2=200, y2=380, confidence=0.9, label="person"),
            RawDetection(x1=300, y1=80, x2=580, y2=280, confidence=0.85, label="screen"),
        ]


def test_yolo_world_detector_satisfies_detector_protocol() -> None:
    """真实适配器与 Mock 满足同一 `Detector` 协议，可直接替换。"""
    detector: Detector = YoloWorldDetector(
        YoloWorldSettings(classes=("person", "screen")), backend=_StubBackend()
    )
    # 合成帧 data=None，需给一帧带像素数据的帧
    frame = Frame(
        frame_id="stub-00000",
        source_id="stub",
        captured_at=T0,
        width=640,
        height=480,
        data="PIXELS",
    )

    detections = detector.detect(frame)

    assert [item.label for item in detections] == ["person", "screen"]
    assert all(item.detected_at == frame.captured_at for item in detections)
    assert all(item.model_version == "yolo-world:stub" for item in detections)


def test_yolo_world_detector_feeds_mock_tracker_unchanged() -> None:
    """检测器换成 YOLO-World 后，跟踪与时间线上层无需修改。"""
    detector: Detector = YoloWorldDetector(backend=_StubBackend())
    tracker: Tracker = MockTracker()
    frame = Frame(
        frame_id="stub-00000",
        source_id="stub",
        captured_at=T0,
        width=640,
        height=480,
        data="PIXELS",
    )

    tracked: list[TrackedObject] = tracker.update(frame, detector.detect(frame))

    assert [item.track_id for item in tracked] == ["person-01", "screen-01"]
