"""集成测试：YOLO-World 检测器接入离线图片/视频流水线。

使用假后端注入，避免安装 torch/ultralytics 与联网下载权重；
验证真实适配器能替换 Mock 跑通闭环、失败时降级不崩溃，以及工厂按配置装配。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from agent_visual_context.config import AppConfig
from agent_visual_context.perception import StaticSceneDetector
from agent_visual_context.perception.yolo_world import (
    RawDetection,
    YoloWorldBackend,
    YoloWorldDetector,
    YoloWorldSettings,
)
from agent_visual_context.runtime import (
    ComponentState,
    PipelineState,
    build_detector,
    build_offline_pipeline,
)
from tests.conftest import SAMPLE_IMAGE, SAMPLE_VIDEO, FakeClock

# person + screen 组合可被默认关系规则 (person looking_at screen) 消费，产出观察；
# 坐标落在最小素材尺寸（视频帧 160x120）内，避免裁剪后退化被丢弃。
_STUB_RESULTS: tuple[RawDetection, ...] = (
    RawDetection(x1=10, y1=20, x2=70, y2=110, confidence=0.9, label="person"),
    RawDetection(x1=80, y1=10, x2=150, y2=90, confidence=0.85, label="screen"),
)


class StubBackend:
    """忽略像素、返回固定检测框的假后端。"""

    model_version = "yolo-world:stub"

    def __init__(self, results: Sequence[RawDetection] = _STUB_RESULTS) -> None:
        self._results = list(results)
        self.calls = 0

    def predict(self, image: Any, **kwargs: Any) -> Sequence[RawDetection]:
        self.calls += 1
        return list(self._results)


def test_offline_image_with_yolo_world_produces_observation(clock: FakeClock) -> None:
    config = AppConfig(
        scene_id="scene-yolo-image",
        max_frames=4,
        persistence_min_hits=1,
        persistence_min_seconds=0.0,
    )
    detector = YoloWorldDetector(
        YoloWorldSettings(classes=("person", "screen")), backend=StubBackend()
    )

    pipeline = build_offline_pipeline(config, SAMPLE_IMAGE, clock=clock, detector=detector)
    result = pipeline.run()

    assert result.frames_processed == 1
    assert pipeline.status.state is PipelineState.STOPPED
    assert pipeline.status.components["detector"].model_version == "yolo-world:stub"
    assert len(result.observations) == 1
    observation = result.observations[0]
    assert observation.predicate == "looking_at"
    assert observation.subject.track_id == "person-01"
    # 观察的 model_version 来自关系推理器（仍为 Mock），检测器版本记录在组件状态上
    assert observation.source_id == "sample_image"
    assert [event.kind for event in result.events] == ["greeting-candidate"]


def test_offline_video_with_yolo_world_produces_bounded_timeline(clock: FakeClock) -> None:
    config = AppConfig(scene_id="scene-yolo-video", max_frames=8)
    backend = StubBackend()
    detector = YoloWorldDetector(backend=backend)

    pipeline = build_offline_pipeline(config, SAMPLE_VIDEO, clock=clock, detector=detector)
    result = pipeline.run()

    assert result.frames_processed == 4
    assert backend.calls == 4
    assert pipeline.status.state is PipelineState.STOPPED
    assert not pipeline.status.is_degraded
    assert 0 < len(result.observations) < result.frames_processed
    assert detector.stats.frames == 4
    assert detector.stats.detections == 8  # 每帧 person + screen


def test_failing_backend_degrades_without_crash(clock: FakeClock) -> None:
    config = AppConfig(scene_id="scene-yolo-degraded", max_frames=4)

    def broken_factory(_settings: YoloWorldSettings) -> YoloWorldBackend:
        msg = "权重下载失败"
        raise RuntimeError(msg)

    detector = YoloWorldDetector(backend_factory=broken_factory)
    pipeline = build_offline_pipeline(config, SAMPLE_IMAGE, clock=clock, detector=detector)

    result = pipeline.run()

    assert result.frames_processed == 1
    assert pipeline.status.state is PipelineState.DEGRADED
    assert pipeline.status.is_degraded
    assert pipeline.status.components["detector"].state is ComponentState.DEGRADED
    assert result.observations == []
    assert result.snapshot is not None
    assert result.snapshot.degraded is True


def test_build_detector_selects_backend_by_config() -> None:
    mock_detector = build_detector(AppConfig(detector_backend="mock"))
    yolo_detector = build_detector(AppConfig(detector_backend="yolo-world"))

    assert isinstance(mock_detector, StaticSceneDetector)
    assert isinstance(yolo_detector, YoloWorldDetector)
    # 真实后端惰性加载：构造检测器不触发 ultralytics 导入
    assert yolo_detector.model_version.startswith("yolo-world:")


def test_cli_run_with_yolo_world_detector_flag(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI --detector yolo-world 走真实适配器装配路径。

    通过 monkeypatch 让 ultralytics 视为缺失，确定性地触发降级，避免联网下载权重；
    验证 flag 被正确路由到 YOLO-World 检测器，且模型不可用时流水线仍产出合法 JSON。
    """
    import json

    from agent_visual_context import cli
    from agent_visual_context.errors import PerceptionError
    from agent_visual_context.perception import yolo_world

    def missing_dependency() -> Any:
        msg = "ultralytics 未安装（测试）"
        raise PerceptionError(msg)

    monkeypatch.setattr(yolo_world, "require_ultralytics", missing_dependency)

    exit_code = cli.main(
        [
            "--log-level",
            "ERROR",
            "run",
            "--input",
            str(SAMPLE_IMAGE),
            "--detector",
            "yolo-world",
            "--scene-id",
            "scene-cli-yolo",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scene_id"] == "scene-cli-yolo"
    # 模型不可用 -> 无检测 -> 无观察，但不崩溃
    assert payload["observations"] == []
