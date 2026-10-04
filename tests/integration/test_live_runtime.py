"""集成测试：实时摄像头 Sidecar 链路（假后端 + Mock 感知，不依赖真实设备/GPU/网络）。

覆盖 Issue #8 的关键验收标准：
- 摄像头可持续运行并按有界队列处理，避免无限堆积；
- 摄像头打开失败 / 读帧失败 / 模型异常时返回 degraded，不抛异常；
- 视觉降级不阻塞调用方（宿主链路可立即拿到状态继续）。
"""

from __future__ import annotations

import time

from agent_visual_context.api import VisualContextApi
from agent_visual_context.config import AppConfig
from agent_visual_context.domain import Detection, Frame
from agent_visual_context.perception.base import Detector
from agent_visual_context.runtime import PipelineState, build_live_runtime
from tests.conftest import FakeCameraBackend


def _live_config(**overrides: object) -> AppConfig:
    base = AppConfig(
        scene_id="scene-live",
        source_id="cam",
        target_fps=10.0,
        # 单帧即提升为观察，便于确定性断言
        persistence_min_hits=1,
        persistence_min_seconds=0.0,
        window_seconds=10.0,
        observation_ttl_seconds=10.0,
        live_poll_interval=0.01,
        live_max_capture_failures=2,
    )
    return base.model_copy(update=overrides)


def test_live_runtime_processes_frames_and_produces_observations() -> None:
    config = _live_config()
    runtime = build_live_runtime(config, backend=FakeCameraBackend(), sleeper=lambda _s: None)

    result = runtime.run(max_frames=3)

    assert result.frames_processed == 3
    assert result.metrics is not None
    assert result.metrics.frames_processed == 3
    assert result.metrics.frames_captured >= 3
    # Mock 检测 person+screen -> looking_at 关系被提升为观察
    assert len(result.observations) >= 1
    assert result.snapshot is not None
    assert result.status.state in (PipelineState.STOPPED, PipelineState.DEGRADED)
    assert not runtime.is_running  # 运行结束后已停止


def test_live_runtime_degrades_when_camera_open_fails() -> None:
    """摄像头打不开时返回 degraded，绝不抛异常，且不阻塞调用方。"""
    config = _live_config()
    runtime = build_live_runtime(
        config, backend=FakeCameraBackend(fail_on_open=True), sleeper=lambda _s: None
    )

    started = time.monotonic()
    result = runtime.run(max_frames=5)
    elapsed = time.monotonic() - started

    assert runtime.capture_failed is True
    assert result.status.is_degraded is True
    assert result.status.state == PipelineState.DEGRADED
    assert result.status.components["source"].state.value in ("degraded", "failed")
    assert result.frames_processed == 0
    assert elapsed < 5.0  # 快速返回，未阻塞宿主


def test_live_runtime_degrades_on_persistent_read_failure() -> None:
    """读帧持续失败达到上限后，采集线程停止并降级，主循环安全退出。"""
    config = _live_config()
    runtime = build_live_runtime(
        config,
        backend=FakeCameraBackend(fail_reads=1000),
        sleeper=lambda _s: None,
    )

    result = runtime.run(duration_seconds=3.0, max_frames=10)

    assert runtime.capture_failed is True
    assert result.status.is_degraded is True
    assert result.status.state == PipelineState.DEGRADED


class _FailingDetector:
    """始终抛异常的检测器，用于验证模型失败只降级不中断实时循环。"""

    model_version = "failing-detector"

    def detect(self, frame: Frame) -> list[Detection]:
        msg = "模型推理崩溃（测试模拟）"
        raise RuntimeError(msg)


def test_live_runtime_survives_model_failure() -> None:
    config = _live_config()
    failing: Detector = _FailingDetector()
    runtime = build_live_runtime(
        config,
        backend=FakeCameraBackend(),
        sleeper=lambda _s: None,
        detector=failing,
    )

    result = runtime.run(max_frames=3)

    # 检测器每帧失败，但循环继续处理完 3 帧并标记降级
    assert result.frames_processed == 3
    assert result.status.is_degraded is True
    assert result.status.components["detector"].state.value in ("degraded", "failed")
    assert result.metrics is not None
    assert result.metrics.model_failures >= 1


def test_live_runtime_start_stop_lifecycle_is_idempotent() -> None:
    config = _live_config()
    runtime = build_live_runtime(config, backend=FakeCameraBackend(), sleeper=lambda _s: None)

    runtime.start()
    runtime.start()  # 重复 start 不应抛异常
    running_after_start = runtime.is_running
    assert running_after_start
    # 让采集/消费短暂运转
    time.sleep(0.05)
    runtime.stop()
    runtime.stop()  # 幂等

    running_after_stop = runtime.is_running
    assert not running_after_stop
    assert runtime.status.state in (PipelineState.STOPPED, PipelineState.DEGRADED)


def test_context_manager_runs_and_stops() -> None:
    config = _live_config()
    runtime = build_live_runtime(config, backend=FakeCameraBackend(), sleeper=lambda _s: None)

    with runtime:
        assert runtime.is_running
        time.sleep(0.05)

    assert not runtime.is_running


def test_api_exposes_health_and_metrics() -> None:
    config = _live_config()
    runtime = build_live_runtime(config, backend=FakeCameraBackend(), sleeper=lambda _s: None)
    api = VisualContextApi.from_live_runtime(runtime)

    result = runtime.run(max_frames=2)
    health = api.get_health()
    metrics = api.get_metrics()

    assert health is not None
    assert health is runtime.status
    assert metrics is not None
    assert metrics.frames_processed == result.frames_processed
    # 场景快照可正常查询（有界、带有效期）
    snapshot = api.get_scene_snapshot()
    assert snapshot.scene_id == "scene-live"
