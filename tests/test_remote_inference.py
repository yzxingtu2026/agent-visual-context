"""单机 WebRTC 示例的推理侧无需相机和网络即可测试。"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from agent_visual_context.domain import Frame, utc_now
from agent_visual_context.runtime.pipeline import FrameResult
from agent_visual_context.runtime.remote import PushFrameSource, RemoteInference


class SlowPipeline:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def process_frame(self, frame: Frame) -> FrameResult:
        if frame.frame_id == "1":
            self.started.set()
            if not self.release.wait(timeout=3):
                raise TimeoutError("test worker timeout")
        return FrameResult(frame_id=frame.frame_id)


def make_frame(number: int) -> Frame:
    return Frame(
        frame_id=str(number), source_id="remote", captured_at=utc_now(),
        width=16, height=16, metadata={"received_mono": time.monotonic()},
    )


def test_remote_inference_drops_old_pending_frames() -> None:
    async def scenario() -> None:
        source = PushFrameSource("remote")
        pipeline = SlowPipeline()
        output: list[dict[str, Any]] = []
        inference = RemoteInference(
            pipeline, source, target_fps=1000,
            on_result=output.append,
        )
        inference.start()
        try:
            assert inference.offer(make_frame(1))
            assert await asyncio.to_thread(pipeline.started.wait, 2)
            await asyncio.sleep(.002)
            assert inference.offer(make_frame(2))
            await asyncio.sleep(.002)
            assert inference.offer(make_frame(3))
            pipeline.release.set()
            async with asyncio.timeout(3):
                while len(output) < 2:
                    await asyncio.sleep(.01)
            assert [item["frame_id"] for item in output] == ["1", "3"]
            assert output[-1]["queue_dropped"] == 1
            assert output[-1]["sampled"] == 3
            assert output[-1]["processed"] == 2
            assert output[-1]["server_ms"] >= 0
        finally:
            pipeline.release.set()
            await inference.stop()
        assert source.is_open is False

    asyncio.run(scenario())


def test_push_source_requires_open_and_close_is_idempotent() -> None:
    source = PushFrameSource("remote")
    try:
        source.push(make_frame(1))
    except RuntimeError:
        pass
    else:
        raise AssertionError("closed source accepted frame")
    source.open()
    source.push(make_frame(1))
    assert source.read().frame_id == "1"  # type: ignore[union-attr]
    assert source.read() is None
    source.close()
    source.close()
