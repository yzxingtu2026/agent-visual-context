"""YOLO-World 检测器单元测试。

全部使用假后端，无需安装 ultralytics/torch，也不联网下载权重；
覆盖归一化、bbox 裁剪、置信度过滤、空结果、初始化失败、推理异常、
推理超时、缺少像素数据与可观测信息记录等降级路径。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from agent_visual_context.config import AppConfig
from agent_visual_context.domain import Frame
from agent_visual_context.errors import PerceptionError
from agent_visual_context.perception.base import Detector
from agent_visual_context.perception.yolo_world import (
    RawDetection,
    YoloWorldBackend,
    YoloWorldDetector,
    YoloWorldSettings,
)

T0 = datetime(2026, 10, 4, 12, 0, 0, tzinfo=UTC)


class FakeBackend:
    """返回预置原始检测的假后端，并记录被调用时的参数。"""

    def __init__(
        self,
        results: Sequence[RawDetection] = (),
        *,
        model_version: str = "yolo-world:fake",
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
        *,
        classes: Sequence[str],
        conf_threshold: float,
        iou_threshold: float,
        imgsz: int,
        device: str,
    ) -> Sequence[RawDetection]:
        self.calls.append(
            {
                "classes": tuple(classes),
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "imgsz": imgsz,
                "device": device,
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
    width: int = 320,
    height: int = 240,
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


def test_detector_satisfies_protocol_and_normalizes_output() -> None:
    backend = FakeBackend(
        [
            RawDetection(x1=10, y1=20, x2=110, y2=220, confidence=0.82, label="person"),
            RawDetection(x1=200, y1=30, x2=300, y2=130, confidence=0.44, label="phone"),
        ]
    )
    detector: Detector = YoloWorldDetector(
        YoloWorldSettings(classes=("person", "phone")), backend=backend
    )
    frame = make_frame()

    detections = detector.detect(frame)

    assert detector.model_version == "yolo-world:fake"
    assert [item.label for item in detections] == ["person", "phone"]
    person = detections[0]
    assert (person.bbox.x, person.bbox.y) == (10, 20)
    assert (person.bbox.width, person.bbox.height) == (100, 200)
    assert person.confidence == pytest.approx(0.82)
    assert person.detected_at == frame.captured_at
    assert person.model_version == "yolo-world:fake"


def test_backend_receives_configured_settings() -> None:
    backend = FakeBackend([])
    settings = YoloWorldSettings(
        classes=("person", "hand"), conf_threshold=0.1, iou_threshold=0.5, imgsz=512, device="cpu"
    )
    detector = YoloWorldDetector(settings, backend=backend)

    detector.detect(make_frame())

    assert backend.calls[0] == {
        "classes": ("person", "hand"),
        "conf_threshold": 0.1,
        "iou_threshold": 0.5,
        "imgsz": 512,
        "device": "cpu",
    }


def test_bbox_clamped_to_frame_bounds() -> None:
    backend = FakeBackend(
        [RawDetection(x1=-30, y1=-10, x2=999, y2=999, confidence=0.9, label="screen")]
    )
    detector = YoloWorldDetector(backend=backend)

    detections = detector.detect(make_frame(width=320, height=240))

    box = detections[0].bbox
    assert (box.x, box.y) == (0, 0)
    assert (box.width, box.height) == (320, 240)


def test_degenerate_box_dropped() -> None:
    backend = FakeBackend(
        [
            RawDetection(x1=50, y1=50, x2=50, y2=80, confidence=0.9, label="person"),
            RawDetection(x1=400, y1=400, x2=500, y2=500, confidence=0.9, label="hand"),
        ]
    )
    detector = YoloWorldDetector(backend=backend)

    # 第一个框宽度为 0，第二个完全在帧外裁剪后退化，均被丢弃
    assert detector.detect(make_frame(width=320, height=240)) == []


def test_low_confidence_filtered() -> None:
    backend = FakeBackend(
        [
            RawDetection(x1=0, y1=0, x2=50, y2=50, confidence=0.02, label="person"),
            RawDetection(x1=0, y1=0, x2=50, y2=50, confidence=0.3, label="phone"),
        ]
    )
    detector = YoloWorldDetector(YoloWorldSettings(conf_threshold=0.05), backend=backend)

    detections = detector.detect(make_frame())

    assert [item.label for item in detections] == ["phone"]


def test_empty_result_is_not_a_failure() -> None:
    detector = YoloWorldDetector(backend=FakeBackend([]))

    assert detector.detect(make_frame()) == []


def test_missing_pixel_data_raises_perception_error() -> None:
    detector = YoloWorldDetector(backend=FakeBackend([]))

    with pytest.raises(PerceptionError, match="像素数据"):
        detector.detect(make_frame(data=None))


def test_backend_factory_error_degrades_and_is_not_retried() -> None:
    attempts = {"count": 0}

    def failing_factory(_settings: YoloWorldSettings) -> YoloWorldBackend:
        attempts["count"] += 1
        msg = "无法加载权重"
        raise RuntimeError(msg)

    detector = YoloWorldDetector(backend_factory=failing_factory)

    with pytest.raises(PerceptionError, match="初始化失败"):
        detector.detect(make_frame())
    # 初始化失败被缓存，后续帧不再重复加载
    with pytest.raises(PerceptionError, match="此前初始化失败"):
        detector.detect(make_frame())
    assert attempts["count"] == 1


def test_backend_factory_perception_error_propagates() -> None:
    def missing_dependency(_settings: YoloWorldSettings) -> YoloWorldBackend:
        msg = "ultralytics 未安装"
        raise PerceptionError(msg)

    detector = YoloWorldDetector(backend_factory=missing_dependency)

    with pytest.raises(PerceptionError, match="ultralytics 未安装"):
        detector.detect(make_frame())


def test_inference_error_raises_perception_error() -> None:
    backend = FakeBackend([], error=RuntimeError("CUDA out of memory"))
    detector = YoloWorldDetector(backend=backend)

    with pytest.raises(PerceptionError, match="推理失败"):
        detector.detect(make_frame())


def test_inference_timeout_raises_perception_error() -> None:
    backend = FakeBackend([], sleep_seconds=0.3)
    detector = YoloWorldDetector(YoloWorldSettings(timeout_seconds=0.02), backend=backend)

    with pytest.raises(PerceptionError, match="推理超过"):
        detector.detect(make_frame())
    detector.close()


def test_metrics_and_stats_recorded() -> None:
    backend = FakeBackend(
        [
            RawDetection(x1=0, y1=0, x2=40, y2=40, confidence=0.9, label="person"),
            RawDetection(x1=0, y1=0, x2=40, y2=40, confidence=0.8, label="person"),
            RawDetection(x1=0, y1=0, x2=40, y2=40, confidence=0.7, label="phone"),
        ]
    )
    detector = YoloWorldDetector(backend=backend)

    detector.detect(make_frame(frame_id="f0", width=320, height=240))
    detector.detect(make_frame(frame_id="f1", width=320, height=240))

    metrics = detector.last_metrics
    assert metrics is not None
    assert metrics.frame_id == "f1"
    assert metrics.input_width == 320
    assert metrics.input_height == 240
    assert metrics.detection_count == 3
    assert metrics.label_counts == {"person": 2, "phone": 1}
    assert metrics.latency_seconds >= 0.0
    assert detector.stats.frames == 2
    assert detector.stats.detections == 6
    assert detector.stats.mean_latency_seconds >= 0.0
    assert detector.stats.min_latency_seconds is not None
    assert detector.stats.max_latency_seconds is not None


def test_from_config_maps_settings() -> None:
    config = AppConfig(
        detector_backend="yolo-world",
        detector_classes=["person", "hand", "phone", "screen"],
        detector_conf_threshold=0.08,
        detector_iou_threshold=0.5,
        detector_imgsz=512,
        detector_device="cpu",
        detector_timeout_seconds=5.0,
        detector_weights="yolov8s-worldv2.pt",
    )
    backend = FakeBackend([])

    detector = YoloWorldDetector.from_config(config, backend=backend)

    assert detector.settings.classes == ("person", "hand", "phone", "screen")
    assert detector.settings.conf_threshold == pytest.approx(0.08)
    assert detector.settings.imgsz == 512
    assert detector.settings.timeout_seconds == pytest.approx(5.0)
    detector.detect(make_frame())
    assert backend.calls[0]["imgsz"] == 512
