"""测试公共夹具：确定性时钟、隔离环境与配置。"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from agent_visual_context.config import AppConfig

T0 = datetime(2026, 10, 4, 12, 0, 0, tzinfo=UTC)


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
