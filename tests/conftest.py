"""测试公共夹具：确定性时钟、隔离环境与配置。"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from agent_visual_context.config import AppConfig
from agent_visual_context.errors import FrameSourceError

T0 = datetime(2026, 10, 4, 12, 0, 0, tzinfo=UTC)

# 固定离线素材：一张静态图片（320x240 PNG）与一段短视频（2s/10fps/20 帧 MP4）
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_IMAGE = FIXTURES_DIR / "sample_image.png"
SAMPLE_VIDEO = FIXTURES_DIR / "sample_video.mp4"


class FakeClock:
    """可手动推进的时钟，避免测试依赖真实时间。"""

    def __init__(self, start: datetime = T0) -> None:
        self._now = start

    def __call__(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> datetime:
        self._now += timedelta(seconds=seconds)
        return self._now


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """清除宿主环境中的 AVC_* 变量，保证配置类测试可重复。"""
    for key in list(os.environ):
        if key.startswith("AVC_"):
            monkeypatch.delenv(key, raising=False)
    yield


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(T0 + timedelta(seconds=2.5))


@pytest.fixture
def config() -> AppConfig:
    return AppConfig(
        scene_id="scene-test",
        source_id="mock-source",
        max_frames=6,
        target_fps=2.0,
        observation_ttl_seconds=4.0,
        window_seconds=4.0,
        persistence_min_hits=2,
        persistence_min_seconds=1.0,
        min_relation_confidence=0.4,
    )


def make_image(width: int = 640, height: int = 480) -> Any:
    """生成一张纯色 numpy 图像，作为摄像头/离线源的像素数据替身。"""
    import numpy as np

    return np.zeros((height, width, 3), dtype=np.uint8)


class FakeCameraBackend:
    """测试用摄像头后端：产出固定 numpy 帧，可模拟打开失败与读帧失败。

    满足 `input.camera.CameraBackend` 协议（open/read/release），
    让摄像头链路的测试无需真实设备、torch 或网络。
    """

    def __init__(
        self,
        *,
        image: Any = None,
        fail_on_open: bool = False,
        fail_reads: int = 0,
        max_reads: int | None = None,
    ) -> None:
        self._image = make_image() if image is None else image
        self._fail_on_open = fail_on_open
        self._fail_reads = fail_reads
        self._max_reads = max_reads
        self.open_count = 0
        self.release_count = 0
        self.read_count = 0

    def open(self) -> None:
        self.open_count += 1
        if self._fail_on_open:
            msg = "摄像头被占用（测试模拟）"
            raise FrameSourceError(msg)

    def read(self) -> Any | None:
        self.read_count += 1
        if self.read_count <= self._fail_reads:
            return None
        if self._max_reads is not None and self.read_count > self._max_reads:
            return None
        return self._image

    def release(self) -> None:
        self.release_count += 1
