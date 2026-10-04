"""单元测试：有界保最新帧缓冲（LatestFrameBuffer）。"""

from __future__ import annotations

import threading

from agent_visual_context.domain import Frame
from agent_visual_context.input import synthetic_frames
from agent_visual_context.runtime import LatestFrameBuffer
from tests.conftest import T0


def _frames(count: int) -> list[Frame]:
    return synthetic_frames(count, source_id="cam", start=T0)


def test_keeps_latest_when_capacity_one() -> None:
    buffer = LatestFrameBuffer(maxsize=1)
    f1, f2 = _frames(2)

    buffer.put(f1)
    buffer.put(f2)

    assert buffer.latest() is f2
    assert buffer.dropped_total == 1
    assert buffer.put_total == 2
    # 取走后缓冲清空
    assert buffer.latest() is None


def test_stale_backlog_dropped_on_take() -> None:
    buffer = LatestFrameBuffer(maxsize=3)
    frames = _frames(3)

    for frame in frames:
        buffer.put(frame)
    assert len(buffer) == 3

    latest = buffer.latest()
    assert latest is frames[-1]
    # 消费只取最新，更早的两帧积压被丢弃
    assert buffer.dropped_total == 2


def test_overflow_drops_oldest() -> None:
    buffer = LatestFrameBuffer(maxsize=2)
    frames = _frames(3)

    for frame in frames:
        buffer.put(frame)

    # 容量 2，第三帧挤掉第一帧
    assert buffer.dropped_total == 1
    assert len(buffer) == 2
    assert buffer.latest() is frames[-1]


def test_get_returns_none_on_timeout() -> None:
    buffer = LatestFrameBuffer()
    assert buffer.get(timeout=0.01) is None


def test_get_returns_put_frame() -> None:
    buffer = LatestFrameBuffer()
    frame = _frames(1)[0]

    buffer.put(frame)

    assert buffer.get(timeout=0.01) is frame


def test_get_blocks_until_producer_puts() -> None:
    buffer = LatestFrameBuffer()
    frame = _frames(1)[0]
    received: list[Frame | None] = []

    def _produce() -> None:
        buffer.put(frame)

    producer = threading.Thread(target=_produce)
    producer.start()
    received.append(buffer.get(timeout=2.0))
    producer.join(timeout=2.0)

    assert received[0] is frame


def test_concurrent_capture_never_grows_unbounded() -> None:
    """采集线程高频写入、消费缓慢时，缓冲长度始终不超过容量。"""
    buffer = LatestFrameBuffer(maxsize=1)
    frames = _frames(200)

    def _capture() -> None:
        for frame in frames:
            buffer.put(frame)

    capture = threading.Thread(target=_capture)
    capture.start()
    max_len = 0
    processed = 0
    while capture.is_alive() or len(buffer) > 0:
        frame = buffer.get(timeout=0.01)
        max_len = max(max_len, len(buffer))
        if frame is not None:
            processed += 1
    capture.join(timeout=2.0)

    assert max_len <= 1
    assert buffer.put_total == 200
    # 处理数 + 丢弃数 == 采集数（保最新，其余丢弃）
    assert processed + buffer.dropped_total == 200
