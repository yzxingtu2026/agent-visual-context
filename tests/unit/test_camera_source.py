"""单元测试：本地摄像头输入源（假后端，不依赖真实设备/GPU/网络）。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta

import pytest

from agent_visual_context.errors import FrameSourceError
from agent_visual_context.input import CameraSource
from tests.conftest import FakeCameraBackend, FakeClock, make_image


def _recorder_sleeper() -> tuple[list[float], Callable[[float], None]]:
    sleeps: list[float] = []

    def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    return sleeps, _sleep


def test_camera_emits_frames_with_metadata(clock: FakeClock) -> None:
    backend = FakeCameraBackend(image=make_image(320, 240))
    source = CameraSource(
        device_index=1, backend=backend, target_fps=2.0, clock=clock, sleeper=lambda _s: None
    )

    with source:
        first = source.read()
        clock.advance(0.5)
        second = source.read()

    assert first is not None and second is not None
    assert first.frame_id == "camera-1-00000"
    assert second.frame_id == "camera-1-00001"
    assert first.source_id == "camera-1"
    assert second.captured_at - first.captured_at == timedelta(seconds=0.5)
    assert (first.width, first.height) == (320, 240)
    assert first.data is not None
    assert first.metadata["kind"] == "camera"
    assert first.metadata["device_index"] == 1
    assert first.metadata["target_fps"] == 2.0
    assert source.frames_emitted == 2


def test_camera_default_source_id_uses_device_index() -> None:
    source = CameraSource(device_index=2, backend=FakeCameraBackend())
    assert source.source_id == "camera-2"


def test_camera_throttles_to_target_fps(clock: FakeClock) -> None:
    """时钟未推进时，连续读取应按 target_fps 间隔等待（低频采样）。"""
    sleeps, sleeper = _recorder_sleeper()
    backend = FakeCameraBackend()
    source = CameraSource(
        device_index=0, backend=backend, target_fps=2.0, clock=clock, sleeper=sleeper
    )

    with source:
        source.read()  # 首帧不节流
        source.read()  # 距上次 0s < 0.5s，应等待约 0.5s

    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(0.5, abs=1e-6)


def test_camera_skips_sleep_when_interval_elapsed(clock: FakeClock) -> None:
    sleeps, sleeper = _recorder_sleeper()
    source = CameraSource(
        device_index=0, backend=FakeCameraBackend(), target_fps=2.0, clock=clock, sleeper=sleeper
    )

    with source:
        source.read()
        clock.advance(1.0)  # 已超过 0.5s 间隔
        source.read()

    assert sleeps == []


def test_camera_open_failure_raises_and_marks_degraded() -> None:
    backend = FakeCameraBackend(fail_on_open=True)
    source = CameraSource(device_index=0, backend=backend)

    with pytest.raises(FrameSourceError, match="被占用"):
        source.open()
    assert not source.is_open


def test_camera_read_failure_raises() -> None:
    backend = FakeCameraBackend(fail_reads=1)
    source = CameraSource(device_index=0, backend=backend, sleeper=lambda _s: None)

    with source, pytest.raises(FrameSourceError, match="读帧失败"):
        source.read()


def test_camera_close_releases_backend() -> None:
    backend = FakeCameraBackend()
    source = CameraSource(device_index=0, backend=backend)

    source.open()
    source.close()
    source.close()  # 幂等

    assert backend.open_count == 1
    assert backend.release_count == 1


def test_camera_reopen_reacquires_device() -> None:
    backend = FakeCameraBackend()
    source = CameraSource(device_index=0, backend=backend)

    source.open()
    source.close()
    source.open()
    source.close()

    assert backend.open_count == 2
    assert backend.release_count == 2


def test_camera_rejects_invalid_params() -> None:
    with pytest.raises(FrameSourceError, match="target_fps"):
        CameraSource(target_fps=0.0, backend=FakeCameraBackend())
    with pytest.raises(FrameSourceError, match="device_index"):
        CameraSource(device_index=-1, backend=FakeCameraBackend())
