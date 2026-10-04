"""单元测试：实时链路运行指标（LiveMetrics）。"""

from __future__ import annotations

import pytest

from agent_visual_context.runtime import LiveMetrics
from tests.conftest import T0, FakeClock


def test_counts_and_derived_rates() -> None:
    clock = FakeClock(T0)
    metrics = LiveMetrics(clock=clock)
    metrics.start()

    for _ in range(10):
        metrics.record_captured()
    metrics.record_dropped(3)
    metrics.record_processed(latency_seconds=0.2)
    metrics.record_processed(latency_seconds=0.4, degraded=True)
    metrics.record_processed(latency_seconds=0.3)
    metrics.record_processed(latency_seconds=0.5, degraded=True)
    clock.advance(2.0)

    snap = metrics.snapshot()
    assert snap.uptime_seconds == pytest.approx(2.0)
    assert snap.frames_captured == 10
    assert snap.frames_processed == 4
    assert snap.frames_dropped == 3
    assert snap.model_failures == 2
    assert snap.effective_fps == pytest.approx(2.0)  # 4 帧 / 2s
    assert snap.drop_rate == pytest.approx(0.3)  # 3 / 10
    assert snap.failure_rate == pytest.approx(0.5)  # 2 / 4
    assert snap.mean_latency_seconds == pytest.approx(0.35)
    assert snap.min_latency_seconds == pytest.approx(0.2)
    assert snap.max_latency_seconds == pytest.approx(0.5)


def test_uptime_zero_before_start() -> None:
    metrics = LiveMetrics(clock=FakeClock(T0))
    snap = metrics.snapshot()

    assert snap.uptime_seconds == 0.0
    assert snap.effective_fps == 0.0
    assert snap.drop_rate == 0.0


def test_start_resets_counters() -> None:
    clock = FakeClock(T0)
    metrics = LiveMetrics(clock=clock)
    metrics.start()
    metrics.record_captured(5)
    metrics.record_processed(latency_seconds=1.0)

    metrics.start()  # 重新计时应清零
    snap = metrics.snapshot()

    assert snap.frames_captured == 0
    assert snap.frames_processed == 0


def test_resource_usage_defaults_none_until_filled() -> None:
    metrics = LiveMetrics(clock=FakeClock(T0))
    metrics.start()

    assert metrics.snapshot().cpu_percent is None
    assert metrics.snapshot().rss_mb is None

    metrics.set_resource_usage(cpu_percent=42.5, rss_mb=880.0)
    snap = metrics.snapshot()
    assert snap.cpu_percent == pytest.approx(42.5)
    assert snap.rss_mb == pytest.approx(880.0)
