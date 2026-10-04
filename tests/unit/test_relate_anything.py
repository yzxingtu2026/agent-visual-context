"""RelateAnything 关系推理器单元测试。

全部使用假后端，无需安装 relsgg/torch，也不联网下载权重；
覆盖归一化、白名单过滤、置信度过滤、空目标、初始化失败、推理异常、
推理超时、缺少像素数据与可观测信息记录等降级路径。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from agent_visual_context.config import AppConfig
from agent_visual_context.domain import BBox, Frame, TrackedObject
from agent_visual_context.errors import PerceptionError
from agent_visual_context.perception.base import RelationReasoner
from agent_visual_context.perception.relate_anything import (
    DEFAULT_VOCABULARY,
    RawTriplet,
    RelateAnythingBackend,
    RelateAnythingReasoner,
    RelateAnythingSettings,
)

T0 = datetime(2026, 10, 4, 12, 0, 0, tzinfo=UTC)


class FakeBackend:
    """返回预置原始关系三元组的假后端，并记录被调用时的参数。"""

    def __init__(
        self,
        results: Sequence[RawTriplet] = (),
        *,
        model_version: str = "relate-anything:fake",
        error: Exception | None = None,
        sleep_seconds: float = 0.0,
    ) -> None:
        self._results = list(results)
        self._model_version = model_version
        self._error = error
        self._sleep_seconds = sleep_seconds
        self.calls: list[dict[str, Any]] = []

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(
        self,
        image: Any,
        boxes: Any,
        box_labels: Sequence[str],
        *,
        vocabulary: Sequence[str],
        conf_threshold: float,
        topk: int,
    ) -> Sequence[RawTriplet]:
        self.calls.append(
            {
                "box_labels": list(box_labels),
                "vocabulary": list(vocabulary),
                "conf_threshold": conf_threshold,
                "topk": topk,
            }
        )
        if self._sleep_seconds:
            time.sleep(self._sleep_seconds)
        if self._error is not None:
            raise self._error
        return list(self._results)


def make_frame(
    *,
    frame_id: str = "img-00000",
    width: int = 640,
    height: int = 480,
    data: Any = "PIXELS",
    captured_at: datetime = T0,
) -> Frame:
    return Frame(
        frame_id=frame_id,
        source_id="img",
        captured_at=captured_at,
        width=width,
        height=height,
        data=data,
    )


def make_objects() -> list[TrackedObject]:
    """构造两个典型跟踪目标：person 和 screen。"""
    return [
        TrackedObject(
            track_id="person-01",
            label="person",
            bbox=BBox(x=80, y=120, width=120, height=260),
            confidence=0.92,
            first_seen=T0,
            last_seen=T0,
        ),
        TrackedObject(
            track_id="screen-01",
            label="screen",
            bbox=BBox(x=300, y=80, width=280, height=200),
            confidence=0.88,
            first_seen=T0,
            last_seen=T0,
        ),
    ]


def test_reasoner_satisfies_protocol_and_normalizes_output() -> None:
    backend = FakeBackend(
        [
            RawTriplet(subject_index=0, predicate="looking_at", object_index=1, confidence=0.85),
            RawTriplet(subject_index=1, predicate="near", object_index=0, confidence=0.72),
        ]
    )
    reasoner: RelationReasoner = RelateAnythingReasoner(
        RelateAnythingSettings(vocabulary=("looking_at", "near")), backend=backend
    )
    frame = make_frame()
    objects = make_objects()

    relations = reasoner.infer(frame, objects)

    assert reasoner.model_version == "relate-anything:fake"
    assert len(relations) == 2
    first = relations[0]
    assert first.subject.track_id == "person-01"
    assert first.subject.label == "person"
    assert first.predicate == "looking_at"
    assert first.target.track_id == "screen-01"
    assert first.target.label == "screen"
    assert first.confidence == pytest.approx(0.85)
    assert first.observed_at == frame.captured_at
    assert first.model_version == "relate-anything:fake"


def test_backend_receives_configured_settings() -> None:
    backend = FakeBackend([])
    settings = RelateAnythingSettings(
        vocabulary=("holding", "touching"),
        conf_threshold=0.5,
        topk=5,
        device="cpu",
    )
    reasoner = RelateAnythingReasoner(settings, backend=backend)

    reasoner.infer(make_frame(), make_objects())

    assert backend.calls[0] == {
        "box_labels": ["person", "screen"],
        "vocabulary": ["holding", "touching"],
        "conf_threshold": 0.5,
        "topk": 5,
    }


def test_vocabulary_whitelist_filters_unknown_predicates() -> None:
    backend = FakeBackend(
        [
            RawTriplet(subject_index=0, predicate="looking_at", object_index=1, confidence=0.9),
            RawTriplet(subject_index=0, predicate="eating", object_index=1, confidence=0.8),
            RawTriplet(subject_index=1, predicate="near", object_index=0, confidence=0.7),
        ]
    )
    reasoner = RelateAnythingReasoner(
        RelateAnythingSettings(vocabulary=("looking_at", "near")), backend=backend
    )

    relations = reasoner.infer(make_frame(), make_objects())

    predicates = [r.predicate for r in relations]
    assert "eating" not in predicates
    assert "looking_at" in predicates
    assert "near" in predicates


def test_self_reference_skipped() -> None:
    backend = FakeBackend(
        [RawTriplet(subject_index=0, predicate="looking_at", object_index=0, confidence=0.9)]
    )
    reasoner = RelateAnythingReasoner(
        RelateAnythingSettings(vocabulary=("looking_at",)), backend=backend
    )

    relations = reasoner.infer(make_frame(), make_objects())

    assert relations == []


def test_out_of_bounds_index_skipped() -> None:
    backend = FakeBackend(
        [
            RawTriplet(subject_index=0, predicate="looking_at", object_index=99, confidence=0.9),
            RawTriplet(subject_index=-1, predicate="near", object_index=1, confidence=0.8),
        ]
    )
    reasoner = RelateAnythingReasoner(
        RelateAnythingSettings(vocabulary=("looking_at", "near")), backend=backend
    )

    relations = reasoner.infer(make_frame(), make_objects())

    assert relations == []


def test_empty_objects_returns_empty_without_calling_backend() -> None:
    backend = FakeBackend(
        [RawTriplet(subject_index=0, predicate="looking_at", object_index=1, confidence=0.9)]
    )
    reasoner = RelateAnythingReasoner(backend=backend)

    relations = reasoner.infer(make_frame(), [])

    assert relations == []
    assert backend.calls == []


def test_missing_pixel_data_raises_perception_error() -> None:
    reasoner = RelateAnythingReasoner(backend=FakeBackend([]))

    with pytest.raises(PerceptionError, match="像素数据"):
        reasoner.infer(make_frame(data=None), make_objects())


def test_backend_factory_error_degrades_and_is_not_retried() -> None:
    attempts = {"count": 0}

    def failing_factory(_settings: RelateAnythingSettings) -> RelateAnythingBackend:
        attempts["count"] += 1
        msg = "无法加载权重"
        raise RuntimeError(msg)

    reasoner = RelateAnythingReasoner(backend_factory=failing_factory)

    with pytest.raises(PerceptionError, match="初始化失败"):
        reasoner.infer(make_frame(), make_objects())
    # 初始化失败被缓存，后续帧不再重复加载
    with pytest.raises(PerceptionError, match="此前初始化失败"):
        reasoner.infer(make_frame(), make_objects())
    assert attempts["count"] == 1


def test_backend_factory_perception_error_propagates() -> None:
    def missing_dependency(_settings: RelateAnythingSettings) -> RelateAnythingBackend:
        msg = "relsgg 未安装"
        raise PerceptionError(msg)

    reasoner = RelateAnythingReasoner(backend_factory=missing_dependency)

    with pytest.raises(PerceptionError, match="relsgg 未安装"):
        reasoner.infer(make_frame(), make_objects())


def test_inference_error_raises_perception_error() -> None:
    backend = FakeBackend([], error=RuntimeError("CUDA out of memory"))
    reasoner = RelateAnythingReasoner(backend=backend)

    with pytest.raises(PerceptionError, match="推理失败"):
        reasoner.infer(make_frame(), make_objects())


def test_inference_timeout_raises_perception_error() -> None:
    backend = FakeBackend([], sleep_seconds=0.3)
    reasoner = RelateAnythingReasoner(
        RelateAnythingSettings(timeout_seconds=0.02), backend=backend
    )

    with pytest.raises(PerceptionError, match="推理超过"):
        reasoner.infer(make_frame(), make_objects())
    reasoner.close()


def test_confidence_clamped_to_0_1() -> None:
    backend = FakeBackend(
        [
            RawTriplet(subject_index=0, predicate="looking_at", object_index=1, confidence=1.5),
            RawTriplet(subject_index=1, predicate="near", object_index=0, confidence=-0.2),
        ]
    )
    reasoner = RelateAnythingReasoner(
        RelateAnythingSettings(vocabulary=("looking_at", "near")), backend=backend
    )

    relations = reasoner.infer(make_frame(), make_objects())

    assert relations[0].confidence == pytest.approx(1.0)
    assert relations[1].confidence == pytest.approx(0.0)


def test_metrics_and_stats_recorded() -> None:
    backend = FakeBackend(
        [
            RawTriplet(subject_index=0, predicate="looking_at", object_index=1, confidence=0.9),
            RawTriplet(subject_index=1, predicate="near", object_index=0, confidence=0.7),
        ]
    )
    reasoner = RelateAnythingReasoner(
        RelateAnythingSettings(vocabulary=("looking_at", "near")), backend=backend
    )

    reasoner.infer(make_frame(frame_id="f0"), make_objects())
    reasoner.infer(make_frame(frame_id="f1"), make_objects())

    metrics = reasoner.last_metrics
    assert metrics is not None
    assert metrics.frame_id == "f1"
    assert metrics.input_objects == 2
    assert metrics.relation_count == 2
    assert metrics.predicate_counts == {"looking_at": 1, "near": 1}
    assert metrics.latency_seconds >= 0.0
    assert reasoner.stats.frames == 2
    assert reasoner.stats.relations == 4
    assert reasoner.stats.mean_latency_seconds >= 0.0
    assert reasoner.stats.min_latency_seconds is not None
    assert reasoner.stats.max_latency_seconds is not None


def test_from_config_maps_settings() -> None:
    config = AppConfig(
        reasoner_backend="relate-anything",
        reasoner_vocabulary=["looking_at", "holding", "near"],
        reasoner_conf_threshold=0.4,
        reasoner_topk=8,
        reasoner_device="cpu",
        reasoner_timeout_seconds=20.0,
        reasoner_model_name="maelic/relsgg-vits16plus",
    )
    backend = FakeBackend([])

    reasoner = RelateAnythingReasoner.from_config(config, backend=backend)

    assert reasoner.settings.vocabulary == ("looking_at", "holding", "near")
    assert reasoner.settings.conf_threshold == pytest.approx(0.4)
    assert reasoner.settings.topk == 8
    assert reasoner.settings.timeout_seconds == pytest.approx(20.0)
    reasoner.infer(make_frame(), make_objects())
    assert backend.calls[0]["topk"] == 8


def test_default_vocabulary_covers_issue_requirements() -> None:
    """确认默认白名单覆盖 Issue #7 要求的首批关系词。"""
    required = {"looking_at", "facing", "holding", "pointing_at", "touching", "near"}
    assert required.issubset(set(DEFAULT_VOCABULARY))


def test_predicate_case_insensitive_matching() -> None:
    """模型输出 'Looking At' 应能匹配白名单中的 'looking_at'。"""
    backend = FakeBackend(
        [RawTriplet(subject_index=0, predicate="Looking At", object_index=1, confidence=0.8)]
    )
    reasoner = RelateAnythingReasoner(
        RelateAnythingSettings(vocabulary=("looking_at",)), backend=backend
    )

    relations = reasoner.infer(make_frame(), make_objects())

    assert len(relations) == 1
    assert relations[0].predicate == "looking_at"
