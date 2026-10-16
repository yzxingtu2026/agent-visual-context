"""WebRTC 等远端输入共用的单连接低频推理控制器。

视频接收/回传必须在事件循环中持续进行；只在目标频率采样，推理在线程中
串行执行，待处理缓冲容量为 1，永不让旧帧无限排队。每个连接独占状态。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any, Protocol

from ..domain import Frame
from ..input.base import AbstractFrameSource
from .buffer import LatestFrameBuffer
from .pipeline import FrameResult

logger = logging.getLogger("agent_visual_context.runtime.remote")


class PushFrameSource(AbstractFrameSource):
    """由传输适配器推入帧；推理侧沿用 FrameSource 契约。"""

    def __init__(self, source_id: str) -> None:
        super().__init__(source_id)
        self.buffer = LatestFrameBuffer(maxsize=1)

    def push(self, frame: Frame) -> None:
        if not self.is_open:
            raise RuntimeError("远端输入源未打开")
        self.buffer.put(frame)

    def _do_open(self) -> None:
        pass

    def _do_read(self) -> Frame | None:
        return self.buffer.get(timeout=0)

    def _do_close(self) -> None:
        self.buffer.latest()


class FrameProcessor(Protocol):
    def process_frame(self, frame: Frame) -> FrameResult: ...


class RemoteInference:
    """只保存最新待处理帧；输出耗时属于服务端，不冒充跨机端到端延迟。"""

    def __init__(
        self,
        pipeline: FrameProcessor,
        source: PushFrameSource,
        *,
        target_fps: float,
        on_result: Callable[[dict[str, Any]], None],
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps 必须大于零")
        self.pipeline = pipeline
        self.source = source
        self.interval = 1 / target_fps
        self.on_result = on_result
        self.monotonic = monotonic
        self._last_sample = float("-inf")
        self._pending = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self.received = 0
        self.sampled = 0
        self.processed = 0

    def start(self) -> None:
        self.source.open()
        self._task = asyncio.create_task(self._run())

    def observe(self) -> bool:
        """统计每个视频帧；仅到达采样间隔时才需要解码成推理输入。"""
        self.received += 1
        return self.monotonic() - self._last_sample >= self.interval

    def offer(self, frame: Frame) -> bool:
        """在视频轨接收路径调用；未到采样时间时不复制/入队。"""
        now = self.monotonic()
        if now - self._last_sample < self.interval:
            return False
        self._last_sample = now
        self.sampled += 1
        self.source.push(frame)
        self._pending.set()
        return True

    async def _run(self) -> None:
        while True:
            await self._pending.wait()
            self._pending.clear()
            frame = self.source.read()
            if frame is None:
                continue
            started = self.monotonic()
            try:
                result = await asyncio.to_thread(self.pipeline.process_frame, frame)
                payload: dict[str, Any] = {
                    "type": "inference",
                    "frame_id": frame.frame_id,
                    "sampled_at": frame.captured_at.isoformat(),
                    "server_ms": round((self.monotonic() - started) * 1000, 1),
                    "queue_ms": round(max(0, started - frame.metadata.get("received_mono", started)) * 1000, 1),
                    "received": self.received,
                    "sampled": self.sampled,
                    "queue_dropped": self.source.buffer.dropped_total,
                    "detections": [item.model_dump(mode="json") for item in result.detection_items],
                    "relations": [item.model_dump(mode="json") for item in result.relation_items],
                    "observations": [item.model_dump(mode="json") for item in result.observations],
                    "degraded": list(result.degraded_components),
                }
            except Exception as exc:
                payload = {"type": "inference_error", "frame_id": frame.frame_id, "error": str(exc)}
            self.processed += 1
            payload["processed"] = self.processed
            try:
                self.on_result(payload)
            except Exception:
                logger.exception("远端推理结果发送失败，继续处理下一帧")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self.source.close()
